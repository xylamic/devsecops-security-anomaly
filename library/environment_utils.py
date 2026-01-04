"""
Utility functions for managing environment variables.
"""

import os
from dotenv import load_dotenv


def get_azure_workspace_id() -> str:
    """
    Load and return the Azure Workspace ID from environment variables.
    
    :return: The Azure Workspace ID.
    :rtype: str
    :raises ValueError: If the AZ_WORKSPACE_ID environment variable is not set.
    """
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
    workspace_id = os.environ.get("AZ_WORKSPACE_ID")
    if workspace_id is None:
        raise ValueError("AZ_WORKSPACE_ID environment variable is not set")
    return workspace_id
