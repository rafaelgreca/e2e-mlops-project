#!/usr/bin/env bash

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
eval $(parse_yaml ../config/credentials.yaml "CONFIG_")

# defining important variables
export KAGGLE_USERNAME="$CONFIG_KAGGLE_USERNAME"
export KAGGLE_KEY="$CONFIG_KAGGLE_KEY"
export AWS_ACCESS_KEY_ID="$CONFIG_AWS_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$CONFIG_AWS_SECRET_KEY"

if [ "$1" == "current" ]; then
    file_name=Current_ObesityDataSet.csv

    # downloading the raw dataset using the kaggle's api
    kaggle datasets download -d aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster --unzip

    # renaming the csv file
    mv ObesityDataSet.csv "$file_name"

elif [ "$1" == "raw" ]; then
    file_name=Original_ObesityDataSet.csv

    # downloading the current dataset using the kaggle's api
    kaggle competitions download -c playground-series-s4e2

    # unzip the file
    unzip playground-series-s4e2.zip

    # deleting the zip file, the sample submission file, and
    # the test file, as we are only using the training data for now
    rm playground-series-s4e2.zip sample_submission.csv test.csv

    # renaming the csv file
    mv train.csv "$file_name"
fi

if [[ "$CONFIG_S3" != "YOUR_S3_BUCKET_URL" ]]; then
    # copying the csv file to the s3 bucket
    aws s3 cp "$file_name" s3://$"$CONFIG_S3"

    # deleting the create folder
    rm "$file_name"
fi
