"""
This module provides utility functions for interacting with Azure Log Analytics.
"""

from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient
from datetime import timedelta
import pandas as pd

def la_query_data(workspace_id: str, query: str , timespan=None) -> pd.DataFrame:
    """
    Query Azure Log Analytics and return results as a Pandas DataFrame. This 
    utilizes default Azure credentials for authentication.
    
    Args:
        workspace_id (str): Log Analytics workspace ID
        query (str): Kusto query to execute
        timespan (timedelta, optional): Time range for the query
        
    Returns:
        pandas.DataFrame: Query results as a DataFrame
    """
    # Set up authentication
    credential = DefaultAzureCredential()
    
    # Create the Log Analytics query client
    client = LogsQueryClient(credential)
    
    # Execute the query
    response = client.query_workspace(
        workspace_id=workspace_id,
        query=query,
        timespan=timespan
    )
    
    # Check if we have results
    if not response.tables:
        return pd.DataFrame()
    
    # Get the first table
    table = response.tables[0]
    
    # Convert table to a list of dictionaries
    data = []
    for row in table.rows:
        item = {}
        for i, column in enumerate(table.columns):
            item[column] = row[i]
        data.append(item)
    
    # Create and return DataFrame
    return pd.DataFrame(data)