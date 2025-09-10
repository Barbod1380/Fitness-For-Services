"""
This module handles user authentication against AWS Cognito.
"""
import streamlit as st
import boto3
from botocore.exceptions import ClientError

def get_cognito_client():
    """
    Initializes and returns a boto3 client for Cognito based on Streamlit secrets.
    """
    try:
        # It's generally better to let boto3 find credentials from the environment,
        # but for this specific setup, we'll use the secrets file.
        # Note: Boto3 automatically searches standard locations, so explicit keys
        # are often not needed if running in a configured environment (like EC2/ECS).
        # However, for local Streamlit development with a secrets file, this is explicit.
        aws_creds = st.secrets["aws"]
        return boto3.client(
            'cognito-idp',
            aws_access_key_id=aws_creds["iam_user_access_key_id"],
            aws_secret_access_key=aws_creds["iam_user_secret_access_key"],
            # It's good practice to specify the region. Let's assume 'us-east-1' for now,
            # as it's a common default. This should be added to secrets.toml ideally.
            region_name="us-east-1"
        )
    except (KeyError, FileNotFoundError):
        st.error("AWS credentials or secrets file not found. Please ensure `.streamlit/secrets.toml` is configured correctly.")
        return None

def login(username, password):
    """
    Authenticates a user against the Cognito User Pool.

    Args:
        username (str): The user's username.
        password (str): The user's password.

    Returns:
        bool: True if login is successful, False otherwise.
    """
    cognito_client = get_cognito_client()
    if not cognito_client:
        return False

    try:
        aws_creds = st.secrets["aws"]
        response = cognito_client.initiate_auth(
            ClientId=aws_creds["cognito_app_client_id"],
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={
                'USERNAME': username,
                'PASSWORD': password,
            }
        )

        # Store user info and tokens in session state
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.cognito_tokens = response['AuthenticationResult']
        st.rerun() # Rerun the app to show the main page
        return True # Though rerun stops execution, return True for clarity

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NotAuthorizedException':
            st.error("Incorrect username or password. Please try again.")
        elif error_code == 'UserNotFoundException':
            st.error("This user does not exist.")
        else:
            st.error(f"An unexpected error occurred: {e.response['Error']['Message']}")
        return False
    except Exception as e:
        st.error(f"A critical error occurred during login: {str(e)}")
        return False

def logout():
    """
    Logs the user out by clearing their session state.
    """
    # List of keys to clear on logout
    keys_to_clear = ['logged_in', 'username', 'cognito_tokens']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()
