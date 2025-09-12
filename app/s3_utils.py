"""
This module provides utility functions for interacting with AWS S3.
"""
import streamlit as st
import boto3
from botocore.exceptions import ClientError
import pandas as pd
import io

def get_s3_client():
    """
    Initializes and returns a boto3 client for S3 using Streamlit secrets.
    """
    try:
        aws_creds = st.secrets["aws"]
        return boto3.client(
            's3',
            aws_access_key_id=aws_creds["iam_user_access_key_id"],
            aws_secret_access_key=aws_creds["iam_user_secret_access_key"],
            region_name=aws_creds.get("region_name", "us-east-1")
        )
    except (KeyError, FileNotFoundError):
        st.error("AWS S3 credentials or secrets file not found. Please ensure `.streamlit/secrets.toml` is configured correctly.")
        return None

@st.cache_data(ttl=300)
def list_available_years(username):
    """
    Lists available "year" directories for a given user in the S3 bucket.
    Assumes a structure of: <bucket>/<username>/<year>/
    """
    s3_client = get_s3_client()
    if not s3_client:
        return []

    bucket_name = st.secrets["aws"]["s3_bucket_name"]
    prefix = f"{username}/"

    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        result = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/')

        years = []
        for page in result:
            for common_prefix in page.get('CommonPrefixes', []):
                # Extract the year from the prefix (e.g., 'company-a/2024/')
                year_str = common_prefix.get('Prefix').split('/')[-2]
                if year_str.isdigit():
                    years.append(year_str)

        return sorted(years, reverse=True)
    except ClientError as e:
        st.error(f"Error listing years from S3: {e.response['Error']['Message']}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching years: {str(e)}")
        return []

@st.cache_data(ttl=300)
def list_files_for_year(username, year):
    """
    Lists all .csv files for a given user and year in the S3 bucket.
    Assumes a structure of: <bucket>/<username>/<year>/<file>.csv
    """
    s3_client = get_s3_client()
    if not s3_client:
        return []

    bucket_name = st.secrets["aws"]["s3_bucket_name"]
    prefix = f"{username}/{year}/"

    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        result = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        csv_files = []
        for page in result:
            for obj in page.get('Contents', []):
                key = obj.get('Key')
                if key and key.lower().endswith('.csv'):
                    # Return just the filename, not the full path
                    file_name = key.split('/')[-1]
                    csv_files.append(file_name)

        return sorted(csv_files)
    except ClientError as e:
        st.error(f"Error listing files from S3: {e.response['Error']['Message']}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching files: {str(e)}")
        return []

def load_csv_from_s3(username, year, filename):
    """
    Loads a specific CSV file from S3 into a pandas DataFrame.
    """
    s3_client = get_s3_client()
    if not s3_client:
        return None

    bucket_name = st.secrets["aws"]["s3_bucket_name"]
    key = f"{username}/{year}/{filename}"

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        csv_data = response['Body'].read()
        df = pd.read_csv(io.BytesIO(csv_data))
        return df
    except ClientError as e:
        st.error(f"Error loading file '{filename}' from S3: {e.response['Error']['Message']}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the file: {str(e)}")
        return None
