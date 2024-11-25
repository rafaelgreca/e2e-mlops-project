# using the lastest version of Ubuntu 22.04 as a base for the Docker image
FROM ubuntu:22.04

# installing Python and Unzip
RUN apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip libgomp1

# creating the root folder
RUN mkdir -p /e2e-mlops-project

# copying all the files within this folder to the newly created folder
COPY . /e2e-mlops-project

# setting the curreting directory to be the folder containing the files
WORKDIR /e2e-mlops-project

# deleting the notebooks folder and keeping only the code version, which will
# be used by the API
RUN rm -r /e2e-mlops-project/notebooks/

COPY ./notebooks/VERSION /e2e-mlops-project/notebooks/VERSION

# updating pip
RUN pip install --no-cache-dir -U pip

# installing requirements
RUN pip install -r requirements.txt
