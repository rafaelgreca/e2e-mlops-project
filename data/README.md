# Data

The data will not be stored locally but rather in an AWS S3 bucket to simulate a real-world scenario with different dataset versions (data versioning) and make it easier for other members of the group to download it.

Before downloading the data, you need to do two steps:

1. Configure your AWS settings (Access Key and Secret Key) using the following command:

```bash
aws configure
```

2. Set your `AWS Credentials` and `Kaggle API Credentials` (used to download the dataset) on the `credentials.yaml` file.

Finally, you can download the dataset using the following command:

```bash
bash download_data.sh
```

A folder will be created inside the `/tmp/` folder in your computer to temporarily save the dataset locally and then transfer it to your AWS S3 bucket. After that, the newly created folder will be deleted.