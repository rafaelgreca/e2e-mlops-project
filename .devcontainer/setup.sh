#!/bin/sh
apt-get update
apt-get install -y locales
locale-gen en_US.UTF-8
update-locale LANG=en_US.UTF-8

# init pre-commit hooks
pre-commit install

# git config
git config --global user.name $GIT_USERNAME
git config --global user.email $GIT_EMAIL
git config --global credential.useHttpPath true
git config --global core.filemode false
