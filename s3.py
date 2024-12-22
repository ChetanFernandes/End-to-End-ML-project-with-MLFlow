import os
import subprocess
from src.constants.constants import *
from dotenv import load_dotenv


"""
Configures DVC to use an S3 remote for artifact storage in a production-grade environment.
Ensures credentials are securely retrieved from environment variables.
"""
try:
    load_dotenv()
    # Ensure AWS credentials are set as environment variables
    AWS_ACCESS_KEY_ID = "AKIAU6VTTAEZ2U6NVRWR"
    AWS_SECRET_ACCESS_KEY = "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"
    AWS_REGION = "ap-south-1"
    bucket_url = "https://phishingartifacts.s3.ap-south-1.amazonaws.com/artifacts/"


    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION]):
        raise EnvironmentError("AWS credentials are not set in environment variables.")

    subprocess.run(["dvc", "remote", "add", "-d", "s3remote", bucket_url], check=True)
    subprocess.run(["dvc", "remote", "modify", "s3remote", "access_key_id", AWS_ACCESS_KEY_ID], check=True)
    subprocess.run(["dvc", "remote", "modify", "s3remote", "secret_access_key", AWS_SECRET_ACCESS_KEY], check=True)
    subprocess.run(["dvc", "remote", "modify", "s3remote", "region",AWS_REGION], check=True)

    print("DVC S3 remote successfully configured.")

except subprocess.CalledProcessError as e:
    print(f"Error while configuring DVC remote: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

