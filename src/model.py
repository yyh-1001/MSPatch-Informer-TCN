"""MSPatch-Informer-TCN 主模型。

输入形状为 ``[批大小, 时间窗口, 传感器数]``。模型先把每个传感器视为一条独立序列，
在细/粗两个尺度上切分为 patch；每个尺度并行提取长程注意力和短程 TCN 特征，再逐级
门控融合，最后预测下一时刻的所有传感器值。
"""

import math

import torch
from torch import nn


def stable_softmax(logits: torch.Tensor, dim: int) -> torch.Tensor:
    """以 FP32 执行 softmax，兼顾 AMP 速度与数值稳定性。

    注意力分数出现极端值或非有限值会在这里被截断，避免一次异常 batch 直接让训练
    产生 NaN；输出再转换回原精度以节省显存。
    """
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    logits = logits.clamp(min=-1e4, max=1e4)
    return torch.softmax(logits.float(), dim=dim).to(dtype=logits.dtype)


def patch_count(length: int, patch_len: int, stride: int) -> int:
    """右侧固定复制 S 个观测后，N=floor((w-P)/S)+2。"""
    if min(length, patch_len, stride) <= 0 or length < patch_len:
        raise ValueError("patching requires length >= patch_len > 0 and stride > 0")
    return (length - patch_len) // stride + 2


def unfold_patches(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """把 ``[批, 长度]`` 序列切成 ``[批, patch数, patch长度]``。

    按论文定义，无论能否整除，都在右边界重复最后一个观测 stride 次。
    """
    length = x.size(1)
    patch_count(length, patch_len, stride)
    pad_value = x[:, -1:].expand(-1, stride)
    x = torch.cat([x, pad_value], dim=1)
    return x.unfold(dimension=1, size=patch_len, step=stride)


class ProbSparseSelfAttention(nn.Module):
    """Informer 风格的 ProbSparse 自注意力。

    它先对每个 query 从抽取的 key 中估计稀疏度，只为最重要的少量 query 计算完整
    注意力；未选中的 query 使用 value 的全局均值上下文，从而减少长序列开销。训练时
    使用随机抽样，评估时使用确定性的均匀抽样，保证同一输入重复推理得到相同结果。
    """

    def __init__(self, d_model: int, n_heads: int, factor: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.factor = max(1, int(factor))
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对 ``[批, token数, d_model]`` token 序列提取稀疏全局依赖。"""
        batch, length, _ = x.shape
        if length == 1:
            return self.output(self.value(x))

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        query = split_heads(self.query(x))
        key = split_heads(self.key(x))
        value = split_heads(self.value(x))
        sample_size = min(length, max(1, self.factor * int(math.ceil(math.log(length)))))
        top_queries = sample_size

        # 训练阶段随机估计 query 稀疏度；评估阶段使用循环均匀采样，避免 eval() 下
        # 仍因 torch.randint 而产生不稳定的预测/异常分数。
        if self.training:
            sampled_key_indices = torch.randint(length, (length, sample_size), device=x.device)
        else:
            query_offsets = torch.arange(length, device=x.device).unsqueeze(1)
            sample_offsets = torch.div(
                torch.arange(sample_size, device=x.device) * length,
                sample_size,
                rounding_mode="floor",
            ).unsqueeze(0)
            sampled_key_indices = (query_offsets + sample_offsets).remainder(length)
        sampled_keys = key[:, :, sampled_key_indices, :]
        sampled_scores = (query.unsqueeze(-2) * sampled_keys).sum(dim=-1)
        sparsity = sampled_scores.max(dim=-1).values - sampled_scores.mean(dim=-1)
        top_indices = sparsity.topk(top_queries, dim=-1, sorted=False).indices

        gather_indices = top_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        top_query = query.gather(dim=2, index=gather_indices)
        scores = torch.matmul(top_query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = self.dropout(stable_softmax(scores, dim=-1))
        updates = torch.matmul(weights, value)

        # 未入选 query 采用全局 value 均值上下文，这是 Informer 的近似核心。
        context = value.mean(dim=2, keepdim=True).expand(-1, -1, length, -1).clone()
        context.scatter_(dim=2, index=gather_indices, src=updates)
        context = context.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.output(context)


class MultiheadSelfAttention(nn.Module):
    """PyTorch 完整多头自注意力，用于与 ProbSparse 的加速实验对照。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # 通道独立建模会把“样本数 × 传感器数”展平为 batch；按块调用以免超出 CUDA 网格限制。
        self.max_attention_batch = 8192

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """完整计算所有 token 对之间的注意力，必要时沿 batch 维分块。"""
        if x.size(0) <= self.max_attention_batch:
            output, _ = self.attention(x, x, x, need_weights=False)
            return output

        outputs = []
        for chunk in x.split(self.max_attention_batch, dim=0):
            output, _ = self.attention(chunk, chunk, chunk, need_weights=False)
            outputs.append(output)
        return torch.cat(outputs, dim=0)


class InformerEncoderLayer(nn.Module):
    """包含注意力残差块和前馈残差块的单层编码器。"""

    def __init__(self, d_model: int, n_heads: int, factor: int, dropout: float, attention_type: str) -> None:
        super().__init__()
        if attention_type == "probsparse":
            self.attn = ProbSparseSelfAttention(d_model, n_heads, factor, dropout)
        elif attention_type == "multihead":
            self.attn = MultiheadSelfAttention(d_model, n_heads, dropout)
        else:
            raise ValueError("attention_type must be 'probsparse' or 'multihead'")
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """依次完成注意力融合、归一化、前馈映射和第二次残差归一化。"""
        x = self.norm1(x + self.dropout(self.attn(x)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class InformerEncoder(nn.Module):
    """堆叠多层 Informer 编码器以逐步提炼长程 token 表示。"""

    def __init__(
        self, d_model: int, n_heads: int, layers: int, factor: int, dropout: float, attention_type: str
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [InformerEncoderLayer(d_model, n_heads, factor, dropout, attention_type) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """让 token 依次通过所有编码层。"""
        for layer in self.layers:
            x = layer(x)
        return x


class Chomp1d(nn.Module):
    """移除因果卷积右侧填充，确保每个位置不窥视未来时间点。"""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """裁剪卷积结果末尾的填充长度。"""
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """两层膨胀因果卷积构成的 TCN 残差块。"""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对 ``[批, token数, d_model]`` 做短期局部时序建模。"""
        # Conv1d 以 [批, 通道, 长度] 接收输入，因此进入和离开卷积时各转置一次。
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + y)


class TCNEncoder(nn.Module):
    """按 1、2、4... 膨胀率堆叠 TCN 块，扩大短期分支的感受野。"""

    def __init__(self, d_model: int, levels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TemporalBlock(d_model, kernel_size, dilation=2**i, dropout=dropout)
                for i in range(levels)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """依次应用所有膨胀卷积残差块。"""
        for block in self.blocks:
            x = block(x)
        return x


class TokenGate(nn.Module):
    """对两组同形状特征学习逐 token 的软门控权重。"""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model * 2, 2)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """拼接两路特征，得到权重和为 1 的两路融合结果。"""
        weights = stable_softmax(self.gate(torch.cat([a, b], dim=-1)), dim=-1)
        return weights[..., 0:1] * a + weights[..., 1:2] * b


class ScaleBranch(nn.Module):
    """单一 patch 尺度上的长短时特征提取与摘要。

    ``InformerEncoder`` 负责跨 patch 的长程关系，``TCNEncoder`` 负责邻近 patch 的
    短程变化，随后通过门控融合并汇总为该尺度下的一个向量。
    """

    def __init__(
        self,
        window_size: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        informer_layers: int,
        factor: int,
        tcn_levels: int,
        conv_kernel_size: int,
        dropout: float,
        attention_type: str,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        max_patches = patch_count(window_size, patch_len, stride)
        # 论文的 patch 投影为 W_emb X，不含偏置项。
        self.patch_proj = nn.Linear(patch_len, d_model, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, d_model))
        self.informer = InformerEncoder(d_model, n_heads, informer_layers, factor, dropout, attention_type)
        self.tcn = TCNEncoder(d_model, tcn_levels, conv_kernel_size, dropout)
        self.long_short_gate = TokenGate(d_model)
        self.summary_norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将一批单传感器窗口编码为 ``[批, d_model]`` 尺度摘要。"""
        patches = unfold_patches(x, self.patch_len, self.stride)
        tokens = self.patch_proj(patches)
        pos = self.pos_embed[:, : tokens.size(1), :]
        long_feat = self.informer(tokens + pos)
        short_feat = self.tcn(tokens)
        fused = self.long_short_gate(long_feat, short_feat)
        # 末端 token 强调近期状态，均值池化保留整窗趋势；两者等权汇总。
        summary = 0.5 * fused[:, -1, :] + 0.5 * fused.mean(dim=1)
        return self.summary_norm(summary)


class MSPatchInformerTCN(nn.Module):
    """双尺度、长短时融合、通道独立预测器。"""

    def __init__(
        self,
        window_size: int = 512,
        patch_len_fine: int = 16,
        stride_fine: int = 8,
        patch_len_coarse: int = 64,
        stride_coarse: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        informer_layers: int = 3,
        factor: int = 5,
        tcn_levels: int = 3,
        conv_kernel_size: int = 4,
        dropout: float = 0.1,
        attention_type: str = "probsparse",
    ) -> None:
        super().__init__()
        branch_kwargs = dict(
            window_size=window_size,
            d_model=d_model,
            n_heads=n_heads,
            informer_layers=informer_layers,
            factor=factor,
            tcn_levels=tcn_levels,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            attention_type=attention_type,
        )
        self.fine = ScaleBranch(patch_len=patch_len_fine, stride=stride_fine, **branch_kwargs)
        self.coarse = ScaleBranch(patch_len=patch_len_coarse, stride=stride_coarse, **branch_kwargs)
        self.scale_gate = TokenGate(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """预测输入窗口之后一个时间点的所有传感器值。"""
        # 将每个传感器拆成独立样本，共享分支参数以控制模型规模。
        batch, length, channels = x.shape
        x_ch = x.permute(0, 2, 1).reshape(batch * channels, length)
        fine = self.fine(x_ch)
        coarse = self.coarse(x_ch)
        fused = self.scale_gate(fine, coarse)
        pred = self.head(fused).view(batch, channels)
        return pred


def build_model(config: dict, channels: int = None) -> nn.Module:
    """按配置构造 MSPatch 主模型，并注入窗口长度。"""
    data_cfg = config["data"]
    model_cfg = dict(config["model"])
    architecture = model_cfg.pop("architecture", "mspatch_informer_tcn")
    if architecture != "mspatch_informer_tcn":
        raise ValueError("仅支持 MSPatch 模型 architecture='mspatch_informer_tcn'，当前为 {!r}".format(architecture))
    model_cfg["window_size"] = data_cfg["window_size"]
    return MSPatchInformerTCN(**model_cfg)
