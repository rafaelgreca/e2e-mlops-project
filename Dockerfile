# using the lastest version of Ubuntu 22.04 as a base for the Docker image
FROM python:3.9-slim
ENV VENV_DIR=/opt/venv/_dev
ENV PATH=$VENV_DIR/bin:$PATH
SHELL ["/bin/bash", "-c"]

COPY ./requirements.txt /.
RUN python -m venv $VENV_DIR \
    && $VENV_DIR/bin/pip install --upgrade pip \
    && source $VENV_DIR/bin/activate \
    && pip install -r /requirements.txt

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
