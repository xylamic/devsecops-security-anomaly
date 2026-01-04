"""
Data ingestion for GitHub Audit Anomaly Detection - V3 algorithm. Pulling data
requires Azure Default Credentials to be setup & configured.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import time
import math

# Add the parent directory to the path so we can import from library
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from library import log_analytics_utils
from library import analysis_utils
from library.anomaly_v3_attrs import AnomalyV3Attrs
from library.environment_utils import get_azure_workspace_id


def pull_features_from_log_analytics(lookback_days:int, workspace_id:str, csv_export: str = "") -> pd.DataFrame:
    """
    This pulls the data from LogAnalytics for the ML features based on the query/function 
    available in that system.
    
    :param lookback_days: The number of days to look back for data.
    :type lookback_days: int
    :param workspace_id: The Log Analytics workspace ID.
    :type workspace_id: str
    :param csv_export: The file path & name to export the data to. Blank will not export.
    :type csv_export: str
    :return: The dataframe that holds all the feature data.
    :rtype: DataFrame
    """

    # create increments for pulling data
    range_increment = 30
    range_pairs = []
    for start_day in range(1, lookback_days, range_increment):
        if start_day + range_increment < lookback_days:
            range_pairs.append((start_day + range_increment, start_day))
        else:
            range_pairs.append((start_day, lookback_days + 1))
    
    # create the target dataframe
    df = pd.DataFrame()
    for range_pair in range_pairs:
        print(f"Querying ML Features from {range_pair[0]} to {range_pair[1]} days...")
        start_time = time.time()

        # Set the query
        query = f"""
        fGHAuditMLFeatures(ago({range_pair[0]}d), ago({range_pair[1]}d))
        """

        df_temp = log_analytics_utils.la_query_data(
            workspace_id=workspace_id,
            query=query
        )
        df = pd.concat([df, df_temp])
        minutes, seconds = divmod(time.time() - start_time, 60)
        print(f"Query execution time: {int(minutes)} minutes, {seconds:.2f} seconds; {len(df_temp)} rows")
    
    print(f"Retrieved {len(df)} total rows.")

    analysis_utils.export_dataset_csv(df, csv_export)
    
    return df


def validate_and_clean_features(df: pd.DataFrame, csv_export: str = "") -> pd.DataFrame:

    # filter to only the column I care about (in case there are extra)
    df_features = df[AnomalyV3Attrs.BASE_FEATURES_TO_USE]

    # verify all days have data
    df_features['day'].min()
    df_features['day'].max()

    # iterate from the earliest date to the latest date to see if any values missing
    num_days = 0
    for day in pd.date_range(start=df['day'].min(), end=df['day'].max()):
        num_days += 1
        if len(df_features[df_features['day'] == day]) == 0:
            raise ValueError(f"Missing data for day: {day}")
    print(f"There are {num_days} days of data.")

    # remove rows where the index has '[bot]' in the name or other known bots
    for bot in AnomalyV3Attrs.BOTS_TO_REMOVE:
        df_features = df_features[~df_features['actor'].str.contains(bot)]
    print(f"Removed {len(df) - len(df_features)} bot rows.")

    # export to CSV
    analysis_utils.export_dataset_csv(df_features, csv_export)

    return df_features


def run_ingestion(csv_export: str = "") -> pd.DataFrame:
    # Load environment variables
    workspace_id: str = get_azure_workspace_id()
    lookback_days = 181

    # Pull the data from Log Analytics
    df = pull_features_from_log_analytics(
        lookback_days,
        workspace_id,
        csv_export)

    # Clean the data
    df = validate_and_clean_features(
        df,
        csv_export)
    
    return df


if __name__ == "__main__":

    # get a string for today's date in the format YYYYMMDD
    today_str: str = datetime.today().strftime('%Y%m%d')
    print(f"Today's date str: {today_str}")

    # Construct the data directory path and create it if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    features_file: str = os.path.join(data_dir, f"v3-ml-features-{today_str}.csv")

    print("Ingesting data...")
    run_ingestion(features_file)
    print("Ingestion complete.")