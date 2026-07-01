# Dataset Instructions

This project uses the Jena Climate dataset for multivariate time-series forecasting.

The dataset is downloaded automatically in the notebook using TensorFlow/Keras utilities, so the raw CSV file is not committed to this repository.

## Dataset Source

The notebook downloads the dataset using:

```python
tf.keras.utils.get_file(
    origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
    fname="jena_climate_2009_2016.csv.zip",
    extract=True
)
```

## Dataset Description

The Jena Climate dataset contains weather observations recorded at 10-minute intervals.

The raw dataset contains:

| Item | Value |
|---|---:|
| Rows | 420,551 |
| Columns | 15 |
| Original frequency | 10 minutes |
| Date range | 2009-2016 |
| Domain | Weather / climate time series |

## Data Used in This Project

The notebook uses three weather variables:

```text
T (degC)  -> temperature
p (mbar)  -> pressure
rh (%)    -> humidity
```

The data is resampled from 10-minute frequency to hourly frequency using mean aggregation.

Final modeling dataset:

| Item | Value |
|---|---:|
| Rows | 70,041 |
| Features | 3 |
| Frequency | Hourly |
| Target | 1-hour-ahead temperature |

## Expected Data Flow

No manual dataset download is required for normal notebook execution.

The notebook performs:

```text
Download zip
Extract CSV
Load CSV
Parse Date Time
Select weather features
Resample to hourly
Create chronological train/test split
Create sequence windows
```

## Manual Data Option

If you choose to store the dataset manually for local experimentation, use:

```text
data/raw/jena_climate_2009_2016.csv
```

Do not commit the raw file to GitHub.

## Why Raw Data Is Not Uploaded

The raw dataset is excluded because:

- It can be downloaded programmatically
- It keeps the repository lightweight
- It avoids storing large files in GitHub
- It makes the notebook easier to reproduce in Colab

## Recommended `.gitignore` Rule

Use:

```text
data/raw/
*.zip
```

This prevents accidentally uploading the dataset archive or raw CSV.

## Google Colab Usage

In Google Colab, simply run the notebook from the beginning. The dataset download cell will automatically retrieve and cache the dataset.

## Local Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run the notebook. The dataset will be downloaded automatically to the local Keras datasets cache.

## Note

Only use public or properly licensed datasets for portfolio projects.
