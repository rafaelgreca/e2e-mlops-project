# Notebooks (Research Environment)

Here go the notebooks used for research and development. The main idea is to try to simulate a real-world environment where data scientists use Jupyter Notebooks to explore the available data by doing Exploratory Data Analysis (EDA), data processing, testing different Machine Learning models for the determined task they are trying to solve, testing different hyperparameters for each model, and doing some feature engineering and selection (experimentations).

## Setup Credentials

If you haven't your credentials yet, please check the `docs` folder first before following along.

1. Set your `AWS Credentials` and `Kaggle API Credentials` (used to download the dataset) in the `credentials.yaml` file.

## Running the Code

1. Get the DB endpoint through the AWS user interface (RDS > Databases > Select the database > Connectivy & Security)

2. Run the following command in the AWS EC2 console:

```bash
mlflow server -h 0.0.0.0 -p 5000 \
              --backend-store-uri postgresql://postgres:postgres@<AWS_RDS_URL> \
              --default-artifact-root s3://<AWS_BUCKET_NAME>
```

P.S.: To run the code locally, use the following command:

```bash
mlflow server -h 0.0.0.0 -p 5000
```

3. Run the following code to build the image and run the container using docker compose (**you have to run this command in the root folder**):

```bash
sudo docker-compose up -d
```

4. Read the logs of the container so you can get the link to activate the Jupyter Notebook with the command bellow (copy and paste the link into your favorite web browser):

```bash
sudo docker log <CONTAINER_ID>
```

5. Execute the Jupyter notebooks in the following order:

- Download the data using the script (read the `README` file inside the `data` folder).
- Run the `EDA` notebook.
- Run the `Data Processing` notebook.
- Run the `Experimentations` notebook (will test different Machine Learning models, different hyperparameters for each model, and do some feature engineering and selection).
- Register the best models to the MLflow model registry using the `Experimentations` notebook (last cell) or the MLflow's user interface.
