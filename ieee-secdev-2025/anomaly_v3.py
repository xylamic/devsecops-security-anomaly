"""
This module is for executing the overall Anomaly v3 algorithm, meant to be able to transition
to a production environment.
"""

import anomaly_v3_ingestion
from library import analysis_utils
import pandas as pd
from datetime import datetime
import sys
import os
from dotenv import load_dotenv


def run_ingestion(csv_export: str = "") -> pd.DataFrame:
    # Load environment variables
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
    workspace_id = os.environ.get("AZ_WORKSPACE_ID")
    lookback_days = 188

    # Pull the data from Log Analytics
    df = anomaly_v3_ingestion.pull_features_from_log_analytics(
        lookback_days,
        workspace_id,
        csv_export)

    # Clean the data
    df = anomaly_v3_ingestion.validate_and_clean_features(
        df,
        csv_export)
    
    return df


if __name__ == "__main__":
    commands: list[str] = ["ingest", "features", "evaluate"]

    # check for up to three command line arguments
    if len(sys.argv) == 1:
        print(f"No command line arguments provided. No action. Specify {commands}")
        exit(0)

    # get list of command line arguments
    args = sys.argv[1:]

    # verify all args are valid
    for arg in args:
        if arg not in commands:
            print(f"Invalid command line argument: {arg}. Valid arguments are {commands}")
            exit(1)

    # get a string for today's date in the format YYYYMMDD
    today_str: str = datetime.today().strftime('%Y%m%d')
    print(f"Today's date str: {today_str}")

    features_file: str = f"../data/v3-ml-features-{today_str}.csv"

    # check for the command line argument
    if "ingest" in args:
        print("Ingesting data...")
        run_ingestion(features_file)
        print("Ingestion complete.")
    elif "features" in args:
        print("Running features...")
        print("Feature extraction complete.")
    elif "evaluate" in args:
        print("Running evaluation...")
        print("Evaluation complete.")
