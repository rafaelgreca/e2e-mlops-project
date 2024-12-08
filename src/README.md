# API

To launch the API, execute the following command:

```python
fastapi run api/main.py
```

Use your preferred browser to open `http://0.0.0.0:8000/` to access the documentation. Change it to match your unique address, such as the AWS EC2 URL, if you're not running locally.

## Endpoints

### Data Drift

Uses the reference data — the data used to train the model — and the current data to create a data drift monitoring report.

URL: `http://0.0.0.0:8000/monitor-data`

Entry: a window size (the number of current data samples that will be used.)

Requistion Example (using CURL):

```bash
curl -X 'GET' \
  'http://0.0.0.0:8000/monitor-data?window_size=300' \
  -H 'accept: application/json'
```

Output Example: a HTML page of the generated report. Will also be saved inside the `reports` folder.

### Data Quality

Uses the reference data — the data used to train the model — and the current data to create a data quality monitoring report.

URL: `http://0.0.0.0:8000/monitor-data-quality`

Entry: a window size (the number of current data samples that will be used.)

Requistion Example (using CURL):

```bash
curl -X 'GET' \
  'http://0.0.0.0:8000/monitor-data-quality?window_size=300' \
  -H 'accept: application/json'
```

Output Example: a HTML page of the generated report. Will also be saved inside the `reports` folder.

### Model Performance

Uses the reference data — the data used to train the model — and the current data to create a model performance monitoring report.

URL: `http://0.0.0.0:8000/monitor-model`

Entry: a window size (the number of current data samples that will be used.)

Requistion Example (using CURL):

```bash
curl -X 'GET' \
  'http://0.0.0.0:8000/monitor-model?window_size=300' \
  -H 'accept: application/json'
```

Output Example: a HTML page of the generated report. Will also be saved inside the `reports` folder.

### Predict

Returns the prediction for a given entry.

URL: `http://0.0.0.0:8000/predict`

Entry:

```python
{
  "Age": 24.443011,
  "CAEC": "Sometimes",
  "CALC": "Sometimes",
  "CH2O": 2.763573,
  "FAF": 0,
  "FAVC": "yes",
  "FCVC": 2,
  "Gender": "Male",
  "Height": 1.699998,
  "MTRANS": "Public_Transportation",
  "NCP": 2.983297,
  "SCC": "no",
  "TUE": 1,
  "Weight": 81.66995,
  "family_history_with_overweight": "yes"
}
```

Requistion Example (using CURL):

```bash
curl -X 'POST' \
  'http://0.0.0.0:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "Age": 24.443011,
  "CAEC": "Sometimes",
  "CALC": "Sometimes",
  "CH2O": 2.763573,
  "FAF": 0,
  "FAVC": "yes",
  "FCVC": 2,
  "Gender": "Male",
  "Height": 1.699998,
  "MTRANS": "Public_Transportation",
  "NCP": 2.983297,
  "SCC": "no",
  "TUE": 1,
  "Weight": 81.66995,
  "family_history_with_overweight": "yes"
}'
```

Output Example:

```python
{
  "predictions": [
    "Overweight_Level_II"
  ]
}
```

### Target Drift

Uses the reference data — the data used to train the model — and the current data to create a target drift monitoring report.

URL: `http://0.0.0.0:8000/target-drift`

Entry: a window size (the number of current data samples that will be used.)

Requistion Example (using CURL):

```bash
curl -X 'GET' \
  'http://0.0.0.0:8000/target-drift?window_size=300' \
  -H 'accept: application/json'
```

Output Example: a HTML page of the generated report. Will also be saved inside the `reports` folder.

### Versions

Returns both the code and model versions.

URL: `http://0.0.0.0:8000/version`

Entry: None

Requistion Example (using CURL):

```bash
curl -X 'GET' \
  'http://0.0.0.0:8000/version' \
  -H 'accept: application/json'
```

Output Example:

```python
{
  "code_version": "2.0.0",
  "model_version": "2.0"
}
```
