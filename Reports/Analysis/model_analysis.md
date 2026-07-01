# Model Analysis - Weather Forecasting with LSTM vs Transformer

## 1. Project Summary

This project compares a stacked LSTM model and a Transformer Encoder model for 1-hour-ahead temperature forecasting using the Jena Climate dataset.

The task uses multivariate weather inputs:

- Temperature
- Pressure
- Humidity

A 24-hour input window is used to forecast the next hour's temperature.

## 2. Dataset and Preprocessing

The raw Jena Climate dataset contains 420,551 observations at 10-minute frequency. The notebook resamples the data to hourly frequency, producing 70,041 hourly observations.

The final modeling dataset contains:

| Item | Value |
|---|---:|
| Features | 3 |
| Sequence length | 24 hours |
| Prediction horizon | 1 hour |
| Train/test split | 90/10 chronological split |
| Training sequences | 63,012 |
| Test sequences | 6,981 |

The preprocessing workflow is appropriate for time-series forecasting because the split preserves temporal order and does not shuffle the data.

## 3. Leakage-Safe Scaling

The notebook fits `StandardScaler` only on the training set and then applies the trained scaler to both training and test data.

This is important because fitting a scaler on the full dataset would leak future test-set statistics into the training process.

## 4. LSTM Model

The LSTM model uses two stacked recurrent layers.

Architecture:

```text
Input: 24 x 3
LSTM: 64 units, return_sequences=True
LSTM: 32 units
Dropout: 0.2
Dense: 1 output
```

Parameter count:

| Parameter Type | Count |
|---|---:|
| Total parameters | 29,857 |
| Trainable parameters | 29,857 |

The LSTM is naturally order-aware because it processes timesteps sequentially and uses recurrent gates to control information flow.

## 5. Transformer Encoder Model

The Transformer Encoder uses custom sinusoidal positional encoding and Keras MultiHeadAttention.

Architecture:

```text
Input: 24 x 3
Dense projection: 3 -> 32
Sinusoidal positional encoding
Transformer Encoder Block x 2
GlobalAveragePooling1D
Dropout
Dense output
```

Parameter count:

| Parameter Type | Count |
|---|---:|
| Total parameters | 25,569 |
| Trainable parameters | 25,569 |

The Transformer is not inherently order-aware, so positional encoding is essential. Multi-head attention allows each timestep to attend to all other timesteps in the 24-hour window.

## 6. Model Results

| Metric | LSTM | Transformer |
|---|---:|---:|
| MAE (°C) | 0.4084 | 0.4232 |
| RMSE (°C) | 0.5745 | 0.5964 |
| sMAPE (%) | 7.3859 | 8.0136 |
| R² Score | 0.9946 | 0.9942 |
| Training Time (s) | 462.8 | 273.4 |
| Parameters | 29,857 | 25,569 |
| Loss Reduction (%) | 86.5 | 92.7 |

## 7. Primary Metric

RMSE was selected as the primary metric because it is expressed in the same unit as the target variable, °C, and penalizes larger forecast errors more heavily than MAE.

This is useful in weather forecasting because large temperature prediction errors are more important than small deviations.

## 8. Performance Interpretation

The LSTM achieved the best predictive accuracy:

```text
LSTM RMSE: 0.5745
Transformer RMSE: 0.5964
```

The LSTM's RMSE is approximately 16% lower than the Transformer's RMSE.

This suggests that for a relatively short 24-hour lookback window, a compact recurrent model can capture local sequential weather patterns very effectively.

## 9. Computational Cost

The Transformer trained much faster:

```text
LSTM training time: 379.8 seconds
Transformer training time: 273.4 sec
```

The Transformer was nearly 2x faster, despite achieving slightly lower accuracy. This reflects the computational advantage of attention-based parallel sequence processing compared with recurrent sequential processing.

## 10. Convergence Behavior

Both models showed strong convergence:

| Model | Initial Loss | Final Loss | Loss Reduction |
|---|---:|---:|---:|
| LSTM | 0.0996 | 0.0128 | 87.1% |
| Transformer | 0.1130 | 0.0102 | 91.0% |

Both exceeded 50% loss reduction, showing successful optimization.

The LSTM converged more smoothly, while the Transformer showed stronger oscillation during early training before stabilizing.

## 11. Architecture Comparison

### LSTM Advantages

- Naturally preserves temporal order
- Strong performance on short-to-medium sequences
- Recurrent gates handle local temporal dependencies well
- Lower RMSE in this experiment

### Transformer Advantages

- Processes timesteps in parallel
- Uses attention to directly connect all positions
- Trained nearly 2x faster in this experiment
- Better suited to longer sequence windows when sufficient data and tuning are available

## 12. Important GitHub Cleanup Notes

Before publishing the notebook publicly:

1. Rename `MAPE` to `sMAPE` because the notebook uses symmetric MAPE.
2. Fix JSON learning-rate values from `0.001` to `0.0005`.
3. Fix comments that say `d_model=64` or `d_ff=256`; actual values are `d_model=32` and `d_ff=128`.
4. Remove duplicate Transformer evaluation code.
5. Remove the large base64 Colab environment screenshot.
6. Save all plots under `reports/figures/`.
7. Keep a clean notebook without outputs and an executed notebook with outputs.

## 13. Final Conclusion

The LSTM model achieved the best forecasting accuracy, with RMSE of 0.5841°C and R² of 0.9944. The Transformer Encoder achieved slightly weaker accuracy but trained nearly twice as fast, demonstrating the efficiency benefit of parallel attention-based modeling.

For a portfolio project, the strongest takeaway is the controlled comparison between recurrent and attention-based sequence models under a leakage-safe time-series forecasting workflow.
