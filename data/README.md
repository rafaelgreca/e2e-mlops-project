# Data

The data will not be stored locally but rather in an AWS S3 bucket to simulate a real-world scenario with different dataset versions (data versioning) and make it easier for other members of the group to download it.

Before downloading the data, you need to do one prerequisite step:

1. Set your `AWS Credentials` and `Kaggle API Credentials` (used to download the dataset) in the `credentials.yaml` file.

Finally, you can download both current (testing) and raw (training and validation) using the following commands:

```bash
bash download_data.sh 'raw'
```

```bash
bash download_data.sh 'current'
```

The datasets will be temporarily saved locally (inside the `data` folder) and transferred to your AWS S3 bucket. After that, the dataset will be deleted. If you choose to not use an AWS S3 Bucket, then the dataset will be stored into the `data` folder.
