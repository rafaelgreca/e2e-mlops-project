# Models

The models and artifacts will not be stored locally but rather in an AWS S3 Bucket to simulate a real-world scenario where models will have different versions (model versioning).

This folder will be used temporarily to save the models and artifacts locally and then transfer it to your AWS S3 bucket. After that, the files will be deleted. If you choose to not use an AWS S3 Bucket and an AWS RDS Databaset, then the `artifacts` and the `features` will be stored into the `models` folder.
