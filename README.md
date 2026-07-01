[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SrivastavAryan/weather-forecasting-lstm-transformer/blob/main/Notebooks/weather_forecasting_lstm_transformer_clean.ipynb)
# Weather Forecasting - LSTM vs Transformer Encoder

A deep learning time-series forecasting project that compares a stacked LSTM model with a Transformer Encoder model for 1-hour-ahead temperature prediction using the Jena Climate dataset.

The project demonstrates multivariate time-series preprocessing, temporal train/test splitting, leakage-safe scaling, sequence-window creation, recurrent modeling, positional encoding, multi-head self-attention, and performance comparison using regression metrics.

## Open in Google Colab

After pushing this repository to GitHub, replace `<your-github-username>` in the link below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-github-username>/weather-forecasting-lstm-transformer/blob/main/notebooks/weather_forecasting_lstm_transformer_clean.ipynb)

## Project Objective

The objective is to forecast temperature one hour ahead using the previous 24 hours of weather conditions.

The project compares two deep learning architectures:

1. A stacked LSTM model
2. A Transformer Encoder model with sinusoidal positional encoding and multi-head self-attention

The project focuses on:

- Multivariate time-series forecasting
- Weather data preprocessing
- Chronological train/test split
- Leakage-safe feature scaling
- Sliding-window sequence creation
- LSTM vs Transformer architecture comparison
- Positional encoding for temporal order
- Multi-head attention for sequence modeling
- MAE, RMSE, sMAPE, and R² evaluation
- Training-time, parameter-count, and convergence analysis

## Repository Structure

```text
weather-forecasting-lstm-transformer/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- notebooks/
|   |-- weather_forecasting_lstm_transformer_clean.ipynb
|   |-- weather_forecasting_lstm_transformer_executed.ipynb
|
|-- reports/
|   |-- figures/
|   |   |-- weather_feature_trends.png
|   |   |-- train_test_split.png
|   |   |-- lstm_predictions.png
|   |   |-- transformer_predictions.png
|   |   |-- model_error_comparison.png
|   |   |-- actual_vs_predicted_comparison.png
|   |   |-- training_loss_comparison.png
|   |   |-- positional_encoding_heatmap.png
|   |
|   |-- Analysis/
|       |-- model_analysis.md
|
|-- data/
    |-- README.md
```

## Dataset

This project uses the Jena Climate dataset downloaded through TensorFlow/Keras utilities.

Dataset details from the executed notebook:

| Item | Value |
|---|---:|
| Raw dataset shape | 420,551 rows x 15 columns |
| Final hourly dataset shape | 70,041 rows x 3 columns |
| Features used | Temperature, Pressure, Humidity |
| Target | Temperature |
| Frequency used | Hourly |
| Sequence length | 24 hours |
| Prediction horizon | 1 hour ahead |
| Train/test split | 90/10 chronological split |
| Training sequences | 63,012 |
| Test sequences | 6,981 |

The raw dataset is not stored in this repository because it is downloaded programmatically.

## Features Used

| Feature | Description |
|---|---|
| `temperature` | Air temperature in °C |
| `pressure` | Atmospheric pressure in mbar |
| `humidity` | Relative humidity in % |

The target variable is future temperature:

```text
target = temperature at t + 1 hour
```

## Preprocessing

The notebook follows a leakage-safe time-series preprocessing workflow:

1. Downloads the Jena Climate CSV
2. Parses the `Date Time` column
3. Selects temperature, pressure, and humidity
4. Resamples 10-minute observations into hourly averages
5. Applies a strict 90/10 chronological split
6. Fits `StandardScaler` only on the training data
7. Transforms train and test data using the training-fitted scaler
8. Creates 24-hour sliding windows
9. Predicts the next 1-hour temperature value
10. Inverse-transforms predictions back to °C before metric calculation

## Model 1: Stacked LSTM

The LSTM model uses two stacked recurrent layers.

Architecture summary:

```text
Input: 24 timesteps x 3 features
LSTM layer 1: 64 units, return_sequences=True
LSTM layer 2: 32 units
Dropout: 0.2
Dense output: 1 value
```

Key details:

| Item | Value |
|---|---:|
| Model type | LSTM |
| Recurrent layers | 2 |
| Hidden units | 64 / 32 |
| Total parameters | 29,857 |
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Loss | MSE |
| Batch size | 32 |
| Epochs trained | 26 |
| Loss reduction | 87.1% |

## Model 2: Transformer Encoder

The Transformer model uses custom sinusoidal positional encoding and Keras MultiHeadAttention.

Architecture summary:

```text
Input: 24 timesteps x 3 features
Dense projection: 3 -> d_model
Sinusoidal positional encoding
Transformer Encoder Block x 2:
  - MultiHeadAttention
  - Add + LayerNorm
  - Feed Forward Network
  - Add + LayerNorm
GlobalAveragePooling1D
Dropout
Dense output: 1 value
```

Key details:

| Item | Value |
|---|---:|
| Model type | Transformer Encoder |
| Encoder layers | 2 |
| Attention heads | 4 |
| d_model | 32 |
| d_ff | 128 |
| Total parameters | 25,569 |
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Loss | MSE |
| Batch size | 32 |
| Epochs trained | 22 |
| Loss reduction | 91.0% |

## Results

Metrics are calculated on inverse-transformed predictions in the original Celsius scale.

Because near-zero winter temperatures can make standard MAPE unstable, the notebook uses symmetric MAPE, reported below as `sMAPE`.

| Metric | LSTM | Transformer |
|---|---:|---:|
| MAE (°C) | 0.4100 | 0.5208 |
| RMSE (°C) | 0.5841 | 0.6947 |
| sMAPE (%) | 7.4266 | 9.7856 |
| R² Score | 0.9944 | 0.9921 |
| Training Time (s) | 379.8 | 191.6 |
| Total Parameters | 29,857 | 25,569 |
| Loss Reduction (%) | 87.1 | 91.0 |

## Key Findings

- LSTM achieved the best forecasting accuracy, with approximately 16% lower RMSE than the Transformer.
- Transformer trained nearly 2x faster while maintaining a strong R² score above 0.99.
- LSTM performed better for this 24-hour lookback window because recurrent gating is well suited to short sequential weather patterns.
- Transformer demonstrated the benefit of parallel sequence processing and multi-head attention, especially in training speed.
- Both models exceeded 50% training-loss reduction and generalized well on the chronological test set.
- RMSE was selected as the primary metric because it is expressed in °C and penalizes larger forecast errors.

## Visual Outputs

Recommended figures to save under `reports/figures/`:

```text
weather_feature_trends.png
train_test_split.png
lstm_predictions.png
transformer_predictions.png
model_error_comparison.png
actual_vs_predicted_comparison.png
training_loss_comparison.png
positional_encoding_heatmap.png
```

## How to Run

### Option 1: Run in Google Colab

Open the clean notebook using the Colab badge at the top of this README.

The dataset will be downloaded automatically using:

```python
tf.keras.utils.get_file(...)
```

### Option 2: Run Locally

Clone the repository:

```bash
git clone https://github.com/<your-github-username>/weather-forecasting-lstm-transformer.git
cd weather-forecasting-lstm-transformer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the clean notebook:

```bash
jupyter notebook notebooks/weather_forecasting_lstm_transformer_clean.ipynb
```

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab
- Jupyter Notebook

## Skills Demonstrated

- Multivariate time-series forecasting
- Temporal train/test splitting
- Leakage-safe preprocessing
- Sliding-window sequence generation
- StandardScaler normalization
- LSTM modeling
- Transformer Encoder modeling
- Sinusoidal positional encoding
- Multi-head self-attention
- Regression metric evaluation
- Training-time benchmarking
- Model comparison and interpretation
- Forecast visualization

## Important Notes Before Public GitHub Upload

Before publishing the notebook, make these cleanup changes:

1. Rename the notebook files to portfolio-friendly names:
   - `weather_forecasting_lstm_transformer_clean.ipynb`
   - `weather_forecasting_lstm_transformer_executed.ipynb`
2. Clear outputs in the clean notebook.
3. Keep outputs in the executed notebook.
4. Remove the large embedded Google Colab environment screenshot from the GitHub version.
5. Save plots to `reports/figures/`.
6. Rename `MAPE` to `sMAPE` in the notebook because the current function uses symmetric MAPE.
7. Fix comments that mention `d_model=64` or `d_ff=256`; the actual notebook uses `d_model=32` and `d_ff=128`.
8. Fix JSON learning-rate values from `0.001` to `0.0005`, matching the actual model compile setting.
9. Remove duplicate Transformer evaluation code in the evaluation cell.
10. Keep the assignment JSON only if you label it as an academic/autograder summary.

## Future Improvements

- Add a naive persistence baseline for stronger benchmark context
- Try longer lookback windows such as 48 or 72 hours
- Compare LSTM, GRU, Transformer, and TCN models
- Add hyperparameter tuning logs
- Use multiple forecast horizons such as 3-hour, 6-hour, and 12-hour ahead prediction
- Add prediction intervals or uncertainty estimates
- Deploy an interactive Streamlit demo showing actual vs predicted temperature

## Disclaimer

This project is for educational and portfolio purposes only.
