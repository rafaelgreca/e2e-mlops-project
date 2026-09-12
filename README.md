# End-to-end MLOps Project



- [End-to-end MLOps Project](#end-to-end-mlops-project)
  - [1. About The Project](#1-about-the-project)
    - [1.1. A brief explanation of the primary files and folders](#11-a-brief-explanation-of-the-primary-files-and-folders)
  - [2. Setup | Using VSCODE devcontainer extension as a dev environment](#2-setup--using-vscode-devcontainer-extension-as-a-dev-environment)
    - [2.1 Requirements](#21-requirements)
    - [2.2 If first time setup](#22-if-first-time-setup)
    - [2.3 to open the project in an existing vscode devcontainer](#23-to-open-the-project-in-an-existing-vscode-devcontainer)
    - [2.4 Running tests](#24-running-tests)
  - [3. Pipelines](#3-pipelines)
    - [3.1. Research Environment](#31-research-environment)
    - [3.2. Production Environment](#32-production-environment)
    - [3.3 Built With](#33-built-with)
  - [5. Roadmap](#5-roadmap)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

## 1. About The Project

This project was forked from [Original Repository Name](link-to-original-repository).  
Objective of this repo is to showcase and test the various methodologies I can use to deploy the exact same generic solution. Please Note that all the code in this repository that is not modified by me is from the original repository, and thus do not take credit for it.  

### 1.1. A brief explanation of the primary files and folders

- `.devcontainer`: Contains the ML development environment, includes tools for linting, formatting, type checking, etc. Also includes .env, which is meant to contain environment variables specific for the devcontainer (e.g. git credentials), as well as ny environment variable usually initialized inside the ci/cd pipeline for its execution, but relative to the local devcontainer. Also includes docker-compose overrides specific for the devcontainer.
- `data`: where the script used to download both sets via Kaggle's API and the cleaned version of the training data as well as the evaluation and training datasets will be saved. Note: This folder is primarily used as a temporary folder to install the datasets and when the research environment is not operating locally.
- `models`: this is where the features (like training and validation arrays) and artifacts (like encoders and scalers) will be stored. Note: This folder is primarily used as a temporary location to install features and artifacts when the research environment is not operating locally.
- `notebooks`: these are used to simulate a real-life research work environment by conducting exploratory data analysis, data processing, model training and evaluation, and experiment tracking. Additionally, where the Docker file and isolated requirements for the development environment are stored.
- `reports`: this is where the coverage tests, model performance, data drift, data quality, and target drift monitoring tests will be stored.
- `src`: The main functions, including data processing, loading the trained model, model inference, configuration files, Pydantic's schemas, and the API code source.
- `tests`: this is where the model, data, and API unit and integration tests are created.
- `.pre-commit-config.yaml`: the configuration file for pre-commit.
- `.pylintrc`: the configuration file for Pylint.
- The file `docker-compose.yaml` is used to define and execute multi-container applications in both production and research settings. `Dockerfile`: the production environment's Dockerfile.
- `LICENSE`: the MIT license for the project.
- `requirements.txt`: the environment's requirements for production.

In-depth explanation of the files within the `src` folder:

- `api/`:
  - `main.py`: contains the pipeline and key functions of the API.
  - `utils.py`: contains auxiliary functions for the API, like generating monitoring reports and organizing data to precisely match Evidently AI's requirements.
- `config/`:
  - `aws.py`: handles the credentials for AWS specified in the credentials file.
  - `credentials.yaml`: credentials configuration file.
  - `kaggle.py`: deals with Kaggle's credentials defined inside the credentials file.
  - `log.py`: handles the logs setting specified in the configuration file.
  - `logs.yaml`: logs configuration file.
  - `metadata.yaml`: The metadata file for Cookiecutter.
  - `model.py`: handles the model setting specified in the configuration file.
  - `model.yaml`: model configuration file.
  - `reports.py`: handles the reports settings specified in the configuration file.
  - `reports.yaml`: reports configuration file.
  - `settings.py`: handles general setting specified in the configuration file.
  - `settings.yaml`: general settings configuration file.
- `data/`:
  - `processing.py`: the functions for processing the data, including loading a dataset, generating the desired features, scaling and encoding the features, and more,
  - `utils.py`: contains auxiliary functions for pre-processing and data processing tasks, like loading features and downloading datasets.
- `model/`:
  - `inference.py`: makes an inference for a given data set with the trained model.
- `schema/`:
  - `monitoring.py`: the Pydantic schema that verifies monitoring endpoint entries in the API.
  - `person.py`: the Pydantic schema used to verify the entries of the inference endpoint of the API.

## 2. Setup | Using VSCODE devcontainer extension as a dev environment

Vscode devcontainers enables development environments to emulate production environments, where one can develop code in the same software architecture than the one used in production.

### 2.1 Requirements

Note that the following instructions assume a Windows host PC equipped with a Nvidia Cuda enabled GPU.

- Up to date nvidia graphics driver
- Have Repo pulled (ideally using https, not ssh, for compatibility with further instructions), with core.autocrlf set to False

- Docker Desktop >= 4.34.1 installed : [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
- Ubuntu-WSL 22.04 distribution (download in `start-menu -> microsoft store -> search for Ubuntu22.04 -> Ubuntu 22.04.x LTS (find latest v22.04 version) -> install`)
  - Have it started once so credentials are set up (no specific credentials needed, just input your own)
- Visual studio code : [https://code.visualstudio.com/Download](https://code.visualstudio.com/Download)
- Visual code extensions (install within vscode using extension tab on the left-hand toolbar)
  - Python (ms-python.python)
  - Docker Extension (ms-azuretools.vscode-docker)
  - Dev Containers extension (ms-vscode-remote.remote-containers)
  - WSL (ms-vscode-remote.remote-wsl)
- gcloud CLI [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)

### 2.2 If first time setup

1) Complete listed requirements
2) Start the Docker Desktop application and configure with the following
    - Under `settings/resources/wsl integration` enable `Ubuntu-22.04` as additionnal distro
    - Under settings/general enable "Use the WSL 2 based engine"
3) In windows cmd, set default wsl distro by running `wsl --set-default Ubuntu-22.04`
4) Setup git identity
    - Inside project repo, copy the `./.devcontainer/.env.template` file and rename it to `.env`
    - edit the `GIT_USERNAME` and `GIT_EMAIL` environment variables so they match your identity
        - Ask Jonathan Boilard for the missing fields values
5) Step can currently be skipped ~~Setup devenv user credentials ~~
    - ~~Generate your gcp user-credentials~~
        - ~~User credentials can be generated using the following command `gcloud auth application-default login ~~--disable-quota-project`.
        - ~~Find the file created by the script (path is found in the script's output).~~
        - ~~Copy this file to `./.devcontainer/credentials/`~~
        - ~~Rename file to `user-credentials.json`~~

### 2.3 to open the project in an existing vscode devcontainer

1) Open Docker desktop in windows
2) open vscode
3) Open dev container environment
    - Either :
        - Open the command prompt (ctrl-shift-p) and run the `Run "Dev Containers: Open folder in container"`, then select the directory to which this project was cloned
        - Select `open in devcontainer` on the bottom-right pop-up if it appears
        - If you are reopening this project after already having opened it in a devcontainer, vscode should automatically open it properly.
            - If an `ECONNRESET` error occurs, try building devcontainer with the canam vpn disabled. This is only necessary for bulding the environment for the first time.
    - To validate, make sure that the bottom left blue window footing reads `Dev Container : {project_name}`, and that any terminal opened includes `(_dev)` preceding shell terminal inputs
        - It can take a few seconds for the environment to initialize itself properly when opening container. If `(_dev)` is not preceding terminal inputs, try reopening a terminal (ctrl + shift + `) after a few seconds

### 2.4 Running tests

- To test all docker compose services, run `pytest tests`

## 3. Pipelines

<details>
  <summary> Section bound to change : Click to expand</summary>
    The research and production environments served as the foundation for this project. In addition to trying to improve the data quality by developing, designing, and testing new features and data cleaning procedures, the research environment seeks to create a space designed by Data Scientists to test, train, evaluate, and draw new experiments for new Machine Learning model candidates. This environment is entirely dependent on Jupyter Notebooks, which are not yet ready for production because they are very difficult to automate due to their manual dependency. Following the completion of all experiments in the research environment and the selection of the model to be used, the production environment was intended to be the next stage. In order to be more production-ready, the entire workflow—which only includes the steps that were used in the research environment—will be optimized and structured in this way to get around the drawbacks of the research environment, like its manual dependence and poor optimization, while also deploying the solution to the end-user.
</details>

### 3.1. Research Environment

<details>
  <summary> Note : While I am proficient with the methods in the section below, I do not take credit of them : Click to expand__ </summary>
  As briefly introducted earlier, the research environment was designed to mimic a real-world scenario where Data Scientists conduct experiments for different steps throughout the Machine Learning Lifecycle workflow (e.g., model training and evaluation, data cleaning, feature construction and selection, and so on) using Jupyter Notebooks. The general overview of the research's workflow can be seen in the figure below.

  ![Research Workflow](images/research-workflow.png)

  First, a deep learning model trained on the [Obesity or CVD risk](https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster) dataset was used to create the [Multi-Class Prediction of Obesity Risk](https://www.kaggle.com/competitions/playground-series-s4e2) dataset, which was ingested using Kaggle's API and used to train and validate the models developed throught the pipeline. It is important to note that throughout the entire process, all datasets (raw and cleaned) and artifacts (such as scalers and encoders) were kept in an AWS S3 bucket (or locally within the `data` folder). In order to better understand the data and its characteristics and to learn what can be done in the next steps to improve the quality of the data, an Exploratory Data Analysis (EDA) is performed using the raw data.

  Following the EDA step, a number of data and feature transformations are carried out, including data cleaning (deleting duplicate rows and dropping features that aren't useful), log transformation, height unit to centimeter conversion, numerical column to categorical transformation, standard scale application to the numerical columns, and one hot encoder to the target and categorical columns. Body Mass Index (BMI), Physical Activity Level (PAL), Body Surface Area (BSA), Ideal Body Weight (IBW), the difference between IBW and actual weight (DIFF_IBW), Basal Metabolic Rate (BMR), Total Daily Energy Expenditure (TDEE), Sufficient Water Consumption (SWC), Is Sedentary? (IS), Healthy Habits? (HH), Ideal Number of Main Meals? (INMM), and, lastly, Eat Vegetables Every Main Meal? (EVEMM) are some of the new features that are then manually constructed in the feature construction step.

  Given that a range was established to retain at least 10% of the initial number of features and a maximum of 40%, experiments were created using MLflow and Sequential Feature Selector (SFS) to save the best feature combination. It's important to note that a few machine learning models, such as Decision Tree (DT), XGBoost, Random Forest (RF), LightGBM, and CatBoost, were used with their default parameters in order to assess that feature selection technique. Ultimately, each model has its own best feature combination, and the top model for each type (e.g., one for DT, one for RF, one for XGBoost, and so on) that produced the best outcome are kept in MLflow's model registry. The best models for each kind of model are then used in the hyperparameter tuning experiment, and they are also kept in MLflow's model registry. The figure below illustrates the model training pipeline used for both experiments (feature selection and model tuning) conducted using AWS EC2 (or locally), which divides the raw, preprocessed data into two sets: one for model training and another for performance validation.

  ![Model Training Pipeline](images/model-training-pipeline.png)

  It's important to emphasize that there is much space for improvement because the project's objective was not to place a strong emphasis on the research phase, but rather in the MLOps step (production environment).
</details>

### 3.2. Production Environment

<details>
  <summary> Section bound to change : Click to expand</summary>
  Following the research phase, a production-ready solution is created using the best model, its required artifacts (such as scalers and encoders), the chosen features, and all necessary data processing steps. Software Engineering (SWE) best practices for code quality are used to create a new pipeline using the existing code developed in the research environment (note: we are not using all the features, so we are not using every step; we are not selecting features or training a new model).

  The figure below shows the workflow for the production. The pipeline starts by loading the [Obesity or CVD risk](https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster) dataset, which we refer to as the "current" dataset, and the artifacts acquired during the model training in the research environment. In order to track target and data drift as well as the decline in the model's performance on the test set, this dataset was selected because it contains a variety of distributions for every feature that is available. The current data undergoes the same data transformation procedures as the training data. This also holds true for the feature selection and feature scaling processes (using the same artifacts). The selected model is loaded from the MLflow's model registry following preprocessing of the current data.

  When everything is prepared, they can be used in the deployment process via an API made with FastAPI, or in the Continuous Integration (CI) and Continuous Delivery (CD) stages. During the CI/CD phase, Pylint will be used to test the code quality and unit and integration tests will be used to evaluate everything we have built. To ensure that everything is functioning and the model is attaining the performance indicated in MLflow, the trained model will also be evaluated and tested.

  ![Production Workflow](images/production-workflow.png)

  The figure below shows the design of the API if you decide to use it instead. Continuous Monitoring (CM) features, like detecting target or data drift, determining whether the model's performance is declining, and assessing the quality of the data, were built into the API. Generally speaking, the backend of the API will obtain the reference data (the data used to train the model) and the current data, compare the model's predictions on both, and compare the two datasets. Evidently AI is used to generate the monitoring reports, which are subsequently stored locally or in an AWS S3 bucket.

  ![API Design](images/api_design.png)
</details>

### 3.3 Built With

Please note:
**elements that are striken-through are bound to change**

- ~~**API Framework**: FastAPI, Pydantic~~
- ~~**Cloud Server**: AWS EC2~~
- **Containerization**: Docker, Docker Compose
- **Continuous Integration (CI) and Continuous Delivery (CD)**: GitHub Actions
- ~~**Data Version Control**: AWS S3~~
- **Experiment Tracking**: MLflow, ~~AWS RDS~~
- **Exploratory Data Analysis (EDA)**: Matplotlib, Seaborn
- ~~**Feature and Artifact Store**: AWS S3~~
- **Feature Preprocessing**: Pandas, Numpy
- **Feature Selection**: Optuna
- **Hyperparameter Tuning**: Optuna
- **Logging**: Loguru
- **Model Registry**: MLflow
- ~~**Monitoring**: Evidently AI~~
- **Programming Language**: Python 3
- **Testing**: PyTest
- **Virtual Environment**: Conda Environment, Pip

## 5. Roadmap

Local development or CI/CD features :

- [ ] Add end-to-end test cases.
- [X] Automatically Register a model in mlflow instead of having to manually modify model.yaml file
- [ ] Finalize test containers for CI

Cloud development features :

- [ ] Define cloud CD github actions
- [ ] Implement feature engineering notebooks as a data pipeline
- [ ] implement model training as a model pipeline
- [ ] Make a cloud architecture terraform
- [ ] Replace striked-through elements in section 3.3 by GCP equivalents

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Forked repo author : Jonathan Boilard - [jboilard1994@gmail.com](jboilard1994@gmail.com) | [GitHub](https://github.com/jboilard1994) - [LinkedIn](https://www.linkedin.com/in/jboilard1994)
