from .yaml import register_resolvers
from .style_picker import get_random_style
import os

__all__ = ["register_resolvers", "get_random_style"]


def is_local():
    """Check if Streamlit is running locally"""
    # Check for common cloud environment variables
    cloud_indicators = [
        "STREAMLIT_SHARING",
        "STREAMLIT_CLOUD",
        "HEROKU",
        "AWS_LAMBDA_FUNCTION_NAME",
        "AZURE_FUNCTIONS_ENVIRONMENT",
        "GOOGLE_CLOUD_PROJECT",
    ]

    for indicator in cloud_indicators:
        if os.environ.get(indicator):
            return False

    return True
