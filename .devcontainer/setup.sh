#!/bin/sh
# init pre-commit hooks
pre-commit install

# git config
git config --global user.name $GIT_USERNAME
git config --global user.email $GIT_EMAIL
git config --global credential.useHttpPath true
git config --global core.filemode false
