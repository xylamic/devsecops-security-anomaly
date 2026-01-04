"""
This module implements User and Entity Behavior Analytics (UEBA) for anomaly detection 
in enterprise DevSecOps systems, enterprise GitHub in particular.
"""

from abc import abstractmethod
import pandas as pd
import numpy as np
import sys, os
from sklearn.ensemble import IsolationForest
from datetime import datetime, timezone
import logging
from sklearn.ensemble import IsolationForest
from typing import Optional
import joblib
import argparse
from scipy.stats import median_abs_deviation
from multiprocessing import Pool, cpu_count
from functools import partial

# import module from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from library.anomaly_v3_attrs import AnomalyV3Attrs

# Date format constants
DATE_FORMAT_SHORT = '%Y%m%d'
DATE_FORMAT_LONG = '%Y-%m-%d'


def format_date(date) -> str:
    """Format date as YYYY-MM-DD"""
    return date.strftime(DATE_FORMAT_LONG)

class UEBAAnomalyDetector:
    """
    An abstract class to perform UEBA anomaly detection.
    """

    include_zscores: bool = True
    exclude_negative_zscores: bool = True
    include_mod_zscores: bool = False
    include_zscore_composite_max: bool = False


    def __init__(self) -> None:
        """
        Initializes the base UEBAAnomalyDetector with detection parameters.
        """

        self._logger = logging.getLogger("ueba_anomaly_detection")
        self._logger.info(f"Initializing Base UEBAAnomalyDetector")


    def read_base_features_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Reads the base features from a CSV file. This should be aligned with the features available
        in AnomalyV3Attrs.BASE_FEATURES_TO_USE.

        Args:
            file_path (str): The path to the CSV file.
        Returns:
            pd.DataFrame: The DataFrame containing the base features.
        """

        self._logger.info(f"Reading base features from CSV file: {file_path}")
        df = pd.read_csv(file_path)

        # convert 'day' column to a datetime
        df['day'] = pd.to_datetime(df['day'])

        self._logger.debug(f"Data preview:\n{df.head(5)}")

        # validate all attributes are present
        missing_attrs = []
        for attr in AnomalyV3Attrs.BASE_FEATURES_TO_USE:
            if attr not in df.columns:
                missing_attrs.append(attr)
        if missing_attrs:
            self._logger.error(f"Missing expected attributes in the data: {missing_attrs}")
            raise ValueError(f"Missing expected attributes in the data: {missing_attrs}")
        self._logger.info("All expected attributes are present in the data.")

        return df


    def generate_training_set(self, df_base: pd.DataFrame, history_perc: float = 0.5,
                              history_file_path: str = "", training_file_path: str = "") -> pd.DataFrame:
        """
        Generates a training set from the base features DataFrame. For each day of features for
        the training set, it requires some amount of history to identify deviations in the users'
        pasts. This history of the data is represented as a percentage.
        
        For instance, if there is 6 months of training data and the history percentage is 0.5, 
        then the first three months will be used for building the training set based upon the latter
        3 months.

        Args:
            df_base (pd.DataFrame): The base DataFrame containing feature data.
            history_perc (float): The percentage of historical data to use for training. If this value 
                is less than 0, then all but the most recent day is used as history, and only the most recent.
                This is used when you want to prep for predicting a single day.
            history_file_path (str): Optional file path to save the history data.
            training_file_path (str): Optional file path to save the training data.
        Returns:
            pd.DataFrame: The generated training set DataFrame.
        """

        self._logger.info(f"Generating training set with history percentage: {history_perc}")

        # get the earliest date and latest in the dataset
        history_start_date = df_base['day'].min()
        training_end_date = df_base['day'].max()
        self._logger.info(f"Data date range from {format_date(history_start_date)} to {format_date(training_end_date)}")
        
        # get the dates for the history and for training
        if history_perc < 0:
            # use all but the most current day as history, and only the last day for training
            training_start_date = training_end_date
            history_days = (training_end_date - history_start_date).days
            self._logger.info(f"Using all {history_days} days as history. Training on single day: {format_date(training_start_date)}")
        else:
            total_days = (training_end_date - history_start_date).days + 1
            history_days = int(total_days * history_perc)
            training_start_date = history_start_date + pd.Timedelta(days=history_days)
            self._logger.info(f"Using {history_days} days of history. Training data starts from {format_date(training_start_date)}")
        
        # create history dataframe
        df_history = df_base[df_base['day'] < training_start_date]
            
        # dump the history data to a file if specified
        if history_file_path.strip() != "":
            self._logger.info(f"Exporting history data to CSV file: {history_file_path}")
            df_history.to_csv(history_file_path, index=False)

        # iterate from the earliest date to the latest date
        self._logger.debug(f"History starts on {format_date(history_start_date)}")
        for day in pd.date_range(start=history_start_date, end=training_end_date):
            if day == training_start_date:
                self._logger.debug(f"Training starts on {format_date(day)}")
            self._logger.debug(f"{format_date(day)}: {len(df_base[df_base['day'] == day])} values")
        self._logger.info(f"Training ends on {format_date(training_end_date)}")

        # create a copy of the training dataframe to populate
        df_training = df_base[df_base['day'] >= training_start_date].copy()

        # call the abstract method for each implementation
        df_training = self._create_training_set(df_base, df_training, history_start_date, training_start_date, training_end_date)

        # dump the training data to a file if specified
        if training_file_path.strip() != "":
            self._logger.info(f"Exporting training data to CSV file: {training_file_path}")
            df_training.to_csv(training_file_path, index=False)

        self._logger.debug(f"Training data preview:\n{df_training.head(5)}")
        self._logger.info("Training set generation complete.")
        
        return df_training
    

    def _create_training_set(self, df_base: pd.DataFrame, df_training: pd.DataFrame, 
                             history_start_date: datetime, training_start_date: datetime, training_end_date: datetime) -> pd.DataFrame:
        """
        Creates a training set from the history and training dataframes.

        Args:
            df_base (pd.DataFrame): The base data DataFrame.
            df_training (pd.DataFrame): The training data DataFrame.
            history_start_date (datetime): The start date of the history period.
            training_start_date (datetime): The start date of the training period.
            training_end_date (datetime): The end date of the training period.
        Returns:
            pd.DataFrame: The training set DataFrame.
        """

        self._logger.info(f"Generating training set")

        zscore_prefix = 'zscore_'
        modzscore_prefix = 'modzscore_'

        # create z-score columns
        for column_name in df_training.columns:
            if column_name in AnomalyV3Attrs.FEATURES_TO_ZSCORE_EXCLUDE:
                self._logger.debug(f"Skipping z-score calculation for excluded feature: {column_name}")
                continue
            if self.include_zscores:
                df_training[f"{zscore_prefix}{column_name}"] = 0.0
            if self.include_mod_zscores: df_training[f"{modzscore_prefix}{column_name}"] = 0.0
        if self.include_zscores and self.include_zscore_composite_max:
            df_training[f'{zscore_prefix}composite_max'] = 0.0

        # process each day in the training set, if z-score or mod z-score is included
        if self.include_zscores or self.include_mod_zscores:
            for day in pd.date_range(start=training_start_date, end=training_end_date):
                self._logger.info(f"Processing training data for day: {format_date(day)}")

                # Get the historical data for this day
                history_end_day = day - pd.Timedelta(days=1)
                df_history = df_base[(df_base['day'] >= history_start_date) & (df_base['day'] <= history_end_day)]

                # Create the deviation table
                df_deviations = self._create_deviation_table(df_history)

                # Iterate over each row in the current day to calculate deviations
                for index, row in df_training[df_training['day'] == day].iterrows():
                    actor = row['actor']
                    if actor in df_deviations.index:
                        for feature in df_deviations.columns.levels[0]:
                            if feature in AnomalyV3Attrs.FEATURES_TO_ZSCORE_EXCLUDE:
                                continue

                            # Get deviation info
                            mean = df_deviations.loc[actor, (feature, 'mean')]
                            std = df_deviations.loc[actor, (feature, 'std')]
                            med = df_deviations.loc[actor, (feature, 'median')]

                            # Calculate z-score
                            if self.include_zscores:
                                if std > 0:
                                    z_score = (row[feature] - mean) / std

                                    # If the zscore went below 0, then activity went down- don't track that as an anomaly
                                    if self.exclude_negative_zscores and z_score < 0:
                                        z_score = 0.0
                                else:
                                    z_score = 0.0
                                df_training.at[index, f"{zscore_prefix}{feature}"] = z_score

                            # Calculate modified z-score
                            if self.include_mod_zscores:
                                mad = df_deviations.loc[actor, (feature, 'median_abs_deviation')]
                                if mad == 0:
                                    mod_z = 0.0
                                else:
                                    mod_z = 0.6745 * (row[feature] - med) / mad
                                df_training.at[index, f"{modzscore_prefix}{feature}"] = mod_z

                        # create composite z-score if needed
                        if self.include_zscores and self.include_zscore_composite_max:
                            zscore_cols = [col for col in df_training.columns if col.startswith(f"{zscore_prefix}")]
                            if zscore_cols:
                                composite_zscore = df_training.loc[index, zscore_cols].max()
                                if np.isnan(composite_zscore) or composite_zscore < 0:
                                    composite_zscore = 0.0
                                else:
                                    df_training.at[index, f'{zscore_prefix}composite_max'] = composite_zscore
                
                self._logger.info(f"Completed processing for day: {format_date(day)}")

        # verify no columns for z-scores if turned off
        if not self.include_zscores: assert (not any(df_training.columns.str.startswith(f"{zscore_prefix}")))
        if not self.include_mod_zscores: assert (not any(df_training.columns.str.startswith(f"{modzscore_prefix}")))
        
        return df_training
    

    @abstractmethod
    def fit(self, df_features: pd.DataFrame, model_file_path: str = "") -> None:
        """
        Fits the model to the provided training data.

        Args:
            df_features (pd.DataFrame): The feature data for training.
        Returns:
            None
        """
        pass


    @abstractmethod
    def load_model_from_file(self, model_file_path: str) -> None:
        """
        Loads a pre-trained model from a file.

        Args:
            model_file_path (str): The path to the model file.
        """
        pass


    @abstractmethod
    def predict(self, df_features: pd.DataFrame, day: pd.Timestamp, results_file_path: str = "") -> pd.DataFrame:
        """
        Predicts anomalies in the data.

        Args:
            df_features (pd.DataFrame): The feature data for prediction.
            day (pd.Timestamp): The day for which to predict anomalies.
        Returns:
            pd.DataFrame: A dataframe including the original features and an anomaly score.
        """
        pass
    

    def append_user_email_mappings(self, df_results: pd.DataFrame, user_mappings: dict[str, str]) -> pd.DataFrame:
        """
        Appends user email mappings to the results DataFrame.

        Args:
            df_results (pd.DataFrame): The results DataFrame.
            user_mappings (dict[str, str]): A dictionary mapping actor names to user emails.

        Returns:
            pd.DataFrame: The results DataFrame with an additional 'user_email' column.
        """

        self._logger.info("Appending user email mappings to results.")

        # create a new column for user_email
        df_results['user_email'] = df_results['actor'].map(user_mappings).fillna('')

        self._logger.debug(f"Results with user emails preview:\n{df_results.head(5)}")

        return df_results
    

    def _create_deviation_table(self, df_base: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a deviation table for a specific day based on historical data.

        Args:
            df_base (pd.DataFrame): The base DataFrame containing feature data.
        Returns:
            pd.DataFrame: The deviation table DataFrame.
        """

        self._logger.debug(f"Creating deviation table for day range: {format_date(df_base['day'].min())} to {format_date(df_base['day'].max())}")

        # get a copy of the dataframe without the excluded columns
        self._logger.debug(f"Excluding columns for z-score calculation: {AnomalyV3Attrs.FEATURES_TO_ZSCORE_EXCLUDE}")
        df_deviations = df_base.drop(columns=AnomalyV3Attrs.FEATURES_TO_ZSCORE_EXCLUDE)

        # remove the 'day' column as it's not relevant for this calculation
        df_deviations = df_deviations.drop(columns=['day'])

        # get the average and standard deviation for each column for each actor
        aggs: list = ['mean', 'std', 'median', 'max', 'min']
        if self.include_mod_zscores:
            aggs.append(median_abs_deviation)
        df_deviations = df_deviations.groupby('actor').agg(aggs)

        # for each column, replace any NaN in std with 0.0
        df_deviations = df_deviations.fillna(0.0)

        self._logger.debug(f"Deviation table created with shape: {df_deviations.shape}\n{df_deviations.head(5)}")

        return df_deviations


if __name__ == "__main__":
    raise NotImplementedError("This module is not intended to be run as a standalone script. Call one of its classes or methods from another module.")