#!/bin/sh

# function responsible for parsing the credentials yaml file
function parse_yaml {
    local prefix=$2
    local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
    sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
    awk -F$fs '{
        indent = length($1)/2;
        vname[indent] = $2;
        for (i in vname) {if (i > indent) {delete vname[i]}}
        if (length($3) > 0) {
            vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
            printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
        }
    }'
}

# setting important variables
eval $(parse_yaml ../credentials.yaml "CONFIG_")

export KAGGLE_USERNAME="$CONFIG_KAGGLE_USERNAME"
export KAGGLE_KEY="$CONFIG_KAGGLE_KEY"
s3_bucket="$CONFIG_S3"


# creating a folder within the temporary folder where the dataset will be temporarily saved
mkdir /tmp/e2e-mlops-project/ && cd /tmp/e2e-mlops-project/

# downloading the dataset using the kaggle's api
kaggle datasets download -d aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster

# unzipping the compressed file
unzip obesity-or-cvd-risk-classifyregressorcluster.zip

# deleting the zip file
rm -f obesity-or-cvd-risk-classifyregressorcluster.zip

# renaming the csv file
mv ObesityDataSet.csv Original_ObesityDataSet.csv

# copying the csv file to the s3 bucket
aws s3 cp Original_ObesityDataSet.csv s3://$s3_bucket

# deleting the create folder
cd ~ && rm -rf /tmp/e2e-mlops-project