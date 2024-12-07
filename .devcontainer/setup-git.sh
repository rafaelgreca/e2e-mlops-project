#!/bin/sh
apt-get update -qqy
apt-get install -y git

git config --global user.name "${GIT_USERNAME}"
git config --global user.email "${GIT_EMAIL}"
git config --global credential.useHttpPath true
git config --global core.filemode false