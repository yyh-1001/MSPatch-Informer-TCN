# MSPatch-Informer-TCN

Code for **“Multivariate Time Series Anomaly Detection via Dual-Scale Heterogeneous Encoding and Hierarchical Gated Fusion.”**

MSPatch-Informer-TCN is a prediction-based method for multivariate time series anomaly detection. It combines temporal representations from different patch scales and encoding mechanisms.

## Method

The framework consists of:

- **Dual-scale patching:** constructs fine- and coarse-grained representations using a channel-independent strategy.
- **Heterogeneous encoding:** applies parallel Informer and TCN encoders at each scale.
- **Hierarchical gated fusion:** combines encoder outputs within each scale, then aggregates the summarized representations across scales.
- **Anomaly scoring:** detects deviations using standardized next-step prediction errors and causal smoothing.

Channels are processed independently; cross-variable dependencies are not explicitly modeled.

## Datasets

Experiments use:

- **SWaT:** industrial water treatment monitoring data, available through the dataset provider’s access procedures.
- **SMD:** server monitoring data available from the [OmniAnomaly repository](https://github.com/NetManAIOps/OmniAnomaly).

Datasets are provided by their respective owners.

## Reported Results

Results reported in the manuscript are percentages, expressed as mean ± standard deviation.

| Dataset | F1 with PA | Point-wise F1 without PA |
|---|---:|---:|
| SWaT | 89.78 ± 0.22 | 68.63 ± 0.13 |
| SMD | 91.07 ± 0.18 | 55.56 ± 0.31 |

PA denotes **point adjustment**. PA-based results should be interpreted separately from unadjusted point-wise detection performance.

## Evaluation Protocol

The official training data are divided chronologically into model fitting, validation, and threshold calibration subsets with proportions of 70%, 10%, and 20%.

Thresholds are determined from the 99th percentile of calibration scores and remain fixed during testing. For SMD, calibration is performed separately for each machine. Test labels are used only for evaluation.
