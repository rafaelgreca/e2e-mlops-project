# Setting up AWS's Account

## Prerequisites

Before proceding to the next steps, you **must** have a user attached to a group with the `AdministratorAccess` policy in order to procede. The page will look like something like this:

![Creating an Admin User](../images/admin.png)

After creating this user, you need to create a **Access Key** (Select the created user > Security Credentials > Create ccess key) and use it to configure the AWS CLI using the following command:

```bash
aws configure
```

## Creatings Instances

### Create User, Group, and Configure Access Permisions

1. Create a new user using the following command:

```bash
aws iam create-user --user-name e2e-user
```

The output should be something like this:

```bash
{
    "User": {
        "Path": "/",
        "UserName": "e2e-user",
        "UserId": "AIDATMIB4AESSSIYHA5GY",
        "Arn": "arn:aws:iam::232469037349:user/e2e-user",
        "CreateDate": "2024-08-24T14:27:08Z"
    }
}
```

2. Create a new group (will be used to attached to the new user) using the following command:

```bash
aws iam create-group --group-name e2e-group
```

The output should be something like this:

```bash
{
    "Group": {
        "Path": "/",
        "GroupName": "e2e-group",
        "GroupId": "AGPATMIB4AES6YQ4FOMD3",
        "Arn": "arn:aws:iam::232469037349:group/e2e-group",
        "CreateDate": "2024-08-24T14:30:44Z"
    }
}
```

3. Attach permissions to the new group. We must add full access policies to EC2, RD2, and S3. Use the command bellow:

```bash
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonEC2FullAccess --group-name e2e-group && \
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess --group-name e2e-group && \
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonRDSFullAccess --group-name e2e-group
```

Run the following command to check if the policies has been added to the desired group. The output should be something like this:

```bash
{
    "AttachedPolicies": [
        {
            "PolicyName": "AmazonEC2FullAccess",
            "PolicyArn": "arn:aws:iam::aws:policy/AmazonEC2FullAccess"
        },
        {
            "PolicyName": "AmazonRDSFullAccess",
            "PolicyArn": "arn:aws:iam::aws:policy/AmazonRDSFullAccess"
        },
        {
            "PolicyName": "AmazonS3FullAccess",
            "PolicyArn": "arn:aws:iam::aws:policy/AmazonS3FullAccess"
        }
    ]
}
```

4. Use the following command to attach the newly created user to the new group:

```bash
aws iam add-user-to-group --user-name e2e-user --group-name e2e-group
```

Run the following command to check if the user has been added to the desired group. The output should be something like this:

```bash
aws iam list-groups-for-user --user-name e2e-user
```

The output should be something like this:

```bash
{
    "Groups": [
        {
            "Path": "/",
            "GroupName": "e2e-group",
            "GroupId": "AGPATMIB4AES6YQ4FOMD3",
            "Arn": "arn:aws:iam::232469037349:group/e2e-group",
            "CreateDate": "2024-08-24T14:30:44Z"
        }
    ]
}
```

Validating the commands above using the AWS user interface:

![Validating the Group Creation](../images/iam-1.png)

![Validating the Group Policies](../images/iam-2.png)

5. Creating a **Access Key** through the AWS user interface (go to IAM > Users > select the created user > Security Credentials > Create ccess key) and save its credentials in a safe location. The page will look like something like this:

![Getting Access Key](../images/iam-3.png)

6. Update AWS CLI configuration with the new user's Access Keys using the following command:

```bash
aws configure
```

### Create S3 Bucket

1. Run the command below to create a AWS S3 Bucket:

```bash
aws s3api create-bucket --bucket e2e-bucket --region us-east-1
```

The output should look like something like this:

```bash
{
    "Location": "/e2e-aws-bucket"
}
```

2. Allow public access to the newly create S3 bucket with the following command:

```bash
aws s3api put-public-access-block \
    --bucket e2e-aws-bucket \
    --public-access-block-configuration "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"
```

Validating the commands above using the AWS user interface:

![Validating AWS S3 Creation](../images/s3.png)

### Create EC2 Instance

1. Get the default VPC ID through the AWS user interface (you can find it in EC2 > Default VPC in the top right corner)

2. Get the one of the subnets available in your VPC through AWS user interface (EC2 > Default VPC > VPC > Choose your VPC > Resource Net or EC2 > Default VPC > VPC > Subnets)

3. Get the AMI ID that will be used in the EC2 through the AWS user interface (EC2 > Images > AMI Catalog)

4. Create a Security Group that will be attached to the EC2 instance using the following command:

```bash
aws ec2 create-security-group \
    --group-name e2e-sg \
    --description "AWS EC2 SG" \
    --tag-specifications 'ResourceType=security-group, Tags=[{Key=Name, Value=e2e-sg}]' \
    --vpc-id "vpc-083c00fc36a7e57df"
```

The output should look like this:

```bash
{
    "GroupId": "sg-0613261580cd87115",
    "Tags": [
        {
            "Key": "Name",
            "Value": "e2e-sg"
        }
    ]
}
```

5. Add an inbound ingress firewall with port 5000 (will be used by MLflow) to the security group using the next command:

```bash
aws ec2 authorize-security-group-ingress \
    --group-id "sg-0613261580cd87115" \
    --protocol tcp \
    --port 5000 \
    --cidr "0.0.0.0/0"
```

The output should look like this:

```bash
{
    "Return": true,
    "SecurityGroupRules": [
        {
            "SecurityGroupRuleId": "sgr-04796116590d9b8e8",
            "GroupId": "sg-0613261580cd87115",
            "GroupOwnerId": "232469037349",
            "IsEgress": false,
            "IpProtocol": "tcp",
            "FromPort": 5000,
            "ToPort": 5000,
            "CidrIpv4": "0.0.0.0/0"
        }
    ]
}
```

```bash
aws ec2 authorize-security-group-ingress \
    --group-id "sg-0613261580cd87115" \
    --protocol tcp \
    --port 22 \
    --cidr "18.206.107.24/29"
```

The output should look like this:

```bash
{
    "Return": true,
    "SecurityGroupRules": [
        {
            "SecurityGroupRuleId": "sgr-0f7c473f544d6fa43",
            "GroupId": "sg-0613261580cd87115",
            "GroupOwnerId": "232469037349",
            "IsEgress": false,
            "IpProtocol": "tcp",
            "FromPort": 22,
            "ToPort": 22,
            "CidrIpv4": "18.206.107.24/29"
        }
    ]
}
```

6. Run the command below to create a SSH key pair:

```bash
aws ec2 create-key-pair --key-name e2e-kp
```

The output should look like this:

```bash
{
    "KeyFingerprint": "04:ef:b6:74:80:1d:e6:6e:a8:d1:fd:18:41:07:5a:af:e1:63:51:4c",
    "KeyMaterial": "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAx/++hsoiTb+QeIm9nh/qg8Tzo24iYUH/pBvOwnGGb0Pi/Pgp\noO7rENKEFiEpRnxuiRBmzhkexYIPxVtHZRgkUSomj7IAQ+oWUznIqafZwnmhZbdV\nmQUy3YI3QqyECQ9qVZyPYPy4iZmmz9Bq+9y5xjH3jxjof5NnNWxGZ9aD0zK7oBnV\nOqNQX92JCpQ/xol9QKwoLqrBoVWli2KkdKUSqTl6NQ2PxlvSuj20GE6O5aqEDK/5\n/0ONElfM2VKRKlvTt4NFv1SkfSZCrUM6gE03Cjdyvl+Kofz/CjsKF7SjPJsEk2Yn\nwUsHBssvhtclmWD5g3rE66CngdXxlLzPmi5myQIDAQABAoIBAEkLnwqbyOuci6d6\np0aZYY2+4T1PfD8JW9UUUu+xhugcY0sBWDLvD+uU96kdc3XK/JQS6Xlcs2BvoFL0\ng57oslryiiLxYCdt17mN4TwVkfL1eX0s+JZqz6kXZhdz6sVkW/HovHUv2DZpaAN+\nm5RhiXhZx5D8B+6QZ6GbahuCvaLkvj9ysmLvYpl4dOKs+6bc4D1xD/YV5IW2Lp+S\n4a+dOisRpTlhe88AQBWWfYdzy9YUeJ5mcarVfhZKkJCauIa5f7F+ocNL1TFWWh0u\nBrcB6bRPBNbVrQWUrHxi3oIyYDQAZYgLTdqonB6dlS6slFVMTzFq3RFWO1g9jXPC\nhLBCDgECgYEA/e3ZkZfy8Fu2QOicOQ2/VwnkmxxHRmo/GTsg1tLOs+yaLcdE+2ax\n0c8DQSBRICSPTXUoRy3BcEVgiZggFDffA5sXTHDV7ivw8eIEbmlnuaicypLSYa7j\nimcME3+ps3hGbgjhVQPX/H0PIbNQ17b/BLxKorcr8AvZDkq3wJkjNJ8CgYEAyaFM\nvI8peJAvV5X4Yup+x3j9e4p0TxzKsbpWUxTX+gcbBHBzIjq+YIOg6x56fQXk+y54\nd8RtI9yd3PmMDxIfp0Qj5W+MI92/dePMyaPPPHWIP7j1KVeznYZFQ4uKkZTAdvbU\nrK/cjTLfF/ZqNTj2yP6yCDwfBP72UwS3G3djg5cCgYEA44sKxru5uAcG511J+JpI\nFo/H6fLKL+AvUMW6ZKadEX/4nQMO3pUkzRTLHsX9zGbWfxt4D2iCCcAxhLrsLBHw\n5wNIOnh2S6nWjHnzbJlJmIXuklxE58HXJoO2v5ZLlIY33oxYl10NY7URJSrRS6wx\nE6ZC31JyrfV/+J2d0ZmogZMCgYBtUsXWnxDvS84/gRMSHu7qnNFjwKhblNLaFpxj\no7OO33doqsfiUVOT5QrFMNtdGmsTPhkXZyt9/Eqj27JzIXKEMM/9wMONVthOxVru\nIaFluJxNYK2c4u6ChKDdhLV3Rd4tIpwhnaDJ9kE8LuFbRi2YwXAqQt7WyJ8qWC4n\n+E2+tQKBgCfHBnUZLV4pt2UPKJdwab9OiofyaZfpqnYC+zzmp4t5c+QWZv1zBvl8\nyXYkwPA5D7jqMxAgdWX/HhTGgsP99LvKpwb0CmC5PunzMdAGttrI7aI7KnnATJgY\nrT1LaYOEZZ68xwcDSTfgMIXtFyM3wW/LlDu4pcZaLjzozuMITHiP\n-----END RSA PRIVATE KEY-----",
    "KeyName": "e2e-kp",
    "KeyPairId": "key-0aecf38c24e23b45c"
}
```

7. Finally, use the following command to create an EC2 instance:

```bash
aws ec2 run-instances \
    --image-id ami-0a0e5d9c7acc336f1 \
    --count 1 \
    --instance-type t2.micro \
    --key-name e2e-kp \
    --security-group-ids sg-0613261580cd87115 \
    --subnet-id subnet-02cc9e1bdee86f95e \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sdf\",\"Ebs\":{\"VolumeSize\":30,\"DeleteOnTermination\":true}}]" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name, Value=e2e-ec2}]' 'ResourceType=volume,Tags=[{Key=Name, Value=e2e-ec2-disk}]'
```

The output should look like this:

```bash
{
    "Groups": [],
    "Instances": [
        {
            "AmiLaunchIndex": 0,
            "ImageId": "ami-0a0e5d9c7acc336f1",
            "InstanceId": "i-0993f40f563bedd8d",
            "InstanceType": "t2.micro",
            "KeyName": "e2e-kp",
            "LaunchTime": "2024-08-24T20:43:20.000Z",
            "Monitoring": {
                "State": "disabled"
            },
            "Placement": {
                "AvailabilityZone": "us-east-1f",
                "GroupName": "",
                "Tenancy": "default"
            },
            "PrivateDnsName": "ip-172-31-70-18.ec2.internal",
            "PrivateIpAddress": "172.31.70.18",
            "ProductCodes": [],
            "PublicDnsName": "",
            "State": {
                "Code": 0,
                "Name": "pending"
            },
            "StateTransitionReason": "",
            "SubnetId": "subnet-02cc9e1bdee86f95e",
            "VpcId": "vpc-083c00fc36a7e57df",
            "Architecture": "x86_64",
            "BlockDeviceMappings": [],
            "ClientToken": "162fc794-1a0d-4c18-b91e-1b07d994db76",
            "EbsOptimized": false,
            "EnaSupport": true,
            "Hypervisor": "xen",
            "NetworkInterfaces": [
                {
                    "Attachment": {
                        "AttachTime": "2024-08-24T20:43:20.000Z",
                        "AttachmentId": "eni-attach-03a74f106e9997aa0",
                        "DeleteOnTermination": true,
                        "DeviceIndex": 0,
                        "Status": "attaching",
                        "NetworkCardIndex": 0
                    },
                    "Description": "",
                    "Groups": [
                        {
                            "GroupName": "e2e-sg",
                            "GroupId": "sg-0613261580cd87115"
                        }
                    ],
                    "Ipv6Addresses": [],
                    "MacAddress": "16:ff:da:2e:97:9d",
                    "NetworkInterfaceId": "eni-0df36f03dab8d68da",
                    "OwnerId": "232469037349",
                    "PrivateDnsName": "ip-172-31-70-18.ec2.internal",
                    "PrivateIpAddress": "172.31.70.18",
                    "PrivateIpAddresses": [
                        {
                            "Primary": true,
                            "PrivateDnsName": "ip-172-31-70-18.ec2.internal",
                            "PrivateIpAddress": "172.31.70.18"
                        }
                    ],
                    "SourceDestCheck": true,
                    "Status": "in-use",
                    "SubnetId": "subnet-02cc9e1bdee86f95e",
                    "VpcId": "vpc-083c00fc36a7e57df",
                    "InterfaceType": "interface"
                }
            ],
            "RootDeviceName": "/dev/sda1",
            "RootDeviceType": "ebs",
            "SecurityGroups": [
                {
                    "GroupName": "e2e-sg",
                    "GroupId": "sg-0613261580cd87115"
                }
            ],
            "SourceDestCheck": true,
            "StateReason": {
                "Code": "pending",
                "Message": "pending"
            },
            "Tags": [
                {
                    "Key": "Name",
                    "Value": "e2e-ec2"
                }
            ],
            "VirtualizationType": "hvm",
            "CpuOptions": {
                "CoreCount": 1,
                "ThreadsPerCore": 1
            },
            "CapacityReservationSpecification": {
                "CapacityReservationPreference": "open"
            },
            "MetadataOptions": {
                "State": "pending",
                "HttpTokens": "optional",
                "HttpPutResponseHopLimit": 1,
                "HttpEndpoint": "enabled",
                "HttpProtocolIpv6": "disabled",
                "InstanceMetadataTags": "disabled"
            },
            "EnclaveOptions": {
                "Enabled": false
            },
            "BootMode": "uefi-preferred",
            "PrivateDnsNameOptions": {
                "HostnameType": "ip-name",
                "EnableResourceNameDnsARecord": false,
                "EnableResourceNameDnsAAAARecord": false
            },
            "MaintenanceOptions": {
                "AutoRecovery": "default"
            },
            "CurrentInstanceBootMode": "legacy-bios"
        }
    ],
    "OwnerId": "232469037349",
    "ReservationId": "r-0a55512a761014798"
}
```

Validating the commands above using the AWS user interface:

![Validating AWS EC2 Creation](../images/ec2-1.png)

![Validating AWS EC2 Creation](../images/ec2-2.png)

### Create RDS (Postgresql) Database

1. Create the RDS database instance using the command below:

```bash
aws rds create-db-instance \
    --db-instance-identifier e2e-db \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --master-username postgres \
    --master-user-password postgres \
    --allocated-storage 20
```

The output should look like this:

```bash
{
    "DBInstance": {
        "DBInstanceIdentifier": "e2e-db",
        "DBInstanceClass": "db.t3.micro",
        "Engine": "postgres",
        "DBInstanceStatus": "creating",
        "MasterUsername": "postgres",
        "AllocatedStorage": 20,
        "PreferredBackupWindow": "03:01-03:31",
        "BackupRetentionPeriod": 1,
        "DBSecurityGroups": [],
        "VpcSecurityGroups": [
            {
                "VpcSecurityGroupId": "sg-0478a92f650f4b413",
                "Status": "active"
            }
        ],
        "DBParameterGroups": [
            {
                "DBParameterGroupName": "default.postgres16",
                "ParameterApplyStatus": "in-sync"
            }
        ],
        "DBSubnetGroup": {
            "DBSubnetGroupName": "default",
            "DBSubnetGroupDescription": "default",
            "VpcId": "vpc-083c00fc36a7e57df",
            "SubnetGroupStatus": "Complete",
            "Subnets": [
                {
                    "SubnetIdentifier": "subnet-02cc9e1bdee86f95e",
                    "SubnetAvailabilityZone": {
                        "Name": "us-east-1f"
                    },
                    "SubnetOutpost": {},
                    "SubnetStatus": "Active"
                },
                {
                    "SubnetIdentifier": "subnet-062430a4ed05a4fc8",
                    "SubnetAvailabilityZone": {
                        "Name": "us-east-1d"
                    },
                    "SubnetOutpost": {},
                    "SubnetStatus": "Active"
                },
                {
                    "SubnetIdentifier": "subnet-017dce77c4194ddc8",
                    "SubnetAvailabilityZone": {
                        "Name": "us-east-1e"
                    },
                    "SubnetOutpost": {},
                    "SubnetStatus": "Active"
                },
                {
                    "SubnetIdentifier": "subnet-0f53b4e2ad78aef64",
                    "SubnetAvailabilityZone": {
                        "Name": "us-east-1a"
                    },
                    "SubnetOutpost": {},
                    "SubnetStatus": "Active"
                },
                {
                    "SubnetIdentifier": "subnet-09f71b1c17553f67c",
                    "SubnetAvailabilityZone": {
                        "Name": "us-east-1b"
                    },
                    "SubnetOutpost": {},
                    "SubnetStatus": "Active"
                },
                {
                    "SubnetIdentifier": "subnet-0f8704ef9d991b750",
                    "SubnetAvailabilityZone": {
                        "Name": "us-east-1c"
                    },
                    "SubnetOutpost": {},
                    "SubnetStatus": "Active"
                }
            ]
        },
        "PreferredMaintenanceWindow": "fri:07:14-fri:07:44",
        "PendingModifiedValues": {
            "MasterUserPassword": "****"
        },
        "MultiAZ": false,
        "EngineVersion": "16.3",
        "AutoMinorVersionUpgrade": true,
        "ReadReplicaDBInstanceIdentifiers": [],
        "LicenseModel": "postgresql-license",
        "OptionGroupMemberships": [
            {
                "OptionGroupName": "default:postgres-16",
                "Status": "in-sync"
            }
        ],
        "PubliclyAccessible": true,
        "StorageType": "gp2",
        "DbInstancePort": 0,
        "StorageEncrypted": false,
        "DbiResourceId": "db-ZXEAGWVZLG2LKSEX2DIGZGM2MY",
        "CACertificateIdentifier": "rds-ca-rsa2048-g1",
        "DomainMemberships": [],
        "CopyTagsToSnapshot": false,
        "MonitoringInterval": 0,
        "DBInstanceArn": "arn:aws:rds:us-east-1:232469037349:db:e2e-db",
        "IAMDatabaseAuthenticationEnabled": false,
        "PerformanceInsightsEnabled": false,
        "DeletionProtection": false,
        "AssociatedRoles": [],
        "TagList": [],
        "CustomerOwnedIpEnabled": false,
        "BackupTarget": "region",
        "NetworkType": "IPV4",
        "StorageThroughput": 0,
        "CertificateDetails": {
            "CAIdentifier": "rds-ca-rsa2048-g1"
        },
        "DedicatedLogVolume": false,
        "EngineLifecycleSupport": "open-source-rds-extended-support"
    }
}
```

2. Change the Postgresql type inbound ingress' IP to be the same as the EC2's security group

Validating the commands above using the AWS user interface:

![Validating AWS RDS Creation](../images/rds-1.png)

## Configuring EC2

1. Go to EC2 > Instances > Select the Instance > Connect through the AWS user interface

2. Run the following commands:

```bash
sudo apt update

sudo apt install python3-pip

sudo pip3 install pipenv

sudo pip3 install virtualenv

pipenv install mlflow boto3 psycopg2-binary awscli

pipenv shell

aws configure
```
