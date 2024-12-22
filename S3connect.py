import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the variables
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")

# Print to verify (only for testing; avoid printing sensitive info in production)
print(f"AWS Access Key: {aws_access_key}")
print(f"AWS Region: {aws_region}")

import boto3

session = boto3.session.Session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region
)

# Create an S3 client
s3_client = session.client('s3')

# List S3 buckets
response = s3_client.list_buckets()
print("Buckets:", [bucket['Name'] for bucket in response['Buckets']])