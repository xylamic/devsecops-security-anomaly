"""
This module implements User and Entity Behavior Analytics (UEBA) for anomaly detection 
in enterprise DevSecOps systems, enterprise GitHub in particular.
"""

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
from ueba_anomaly_detection import UEBAAnomalyDetector

# import module from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from library.anomaly_v3_attrs import AnomalyV3Attrs

# Date format constant
DATE_FORMAT_LONG = '%Y-%m-%d'


def format_date(date) -> str:
    """Format date as YYYY-MM-DD"""
    return date.strftime(DATE_FORMAT_LONG)


class UEBAAnomalyDetectorIsolationForest(UEBAAnomalyDetector):
    """
    A class to perform UEBA anomaly detection using Isolation Forest for daily analysis, as defined in 
    https://ieeexplore.ieee.org/document/11226068.
    """

    _model: Optional[IsolationForest] = None


    def __init__(self, contamination: float = 0.1, anomaly_cutoff: float = -0.32) -> None:
        """
        Initializes the UEBAAnomalyDetector with detection parameters.

        Args:
            contamination (float): The proportion of outliers in the data set.
            anomaly_cutoff (float): The cutoff threshold for anomaly scores.
        """

        super().__init__()

        self._logger = logging.getLogger("ueba_anomaly_detection_isolation_forest")
        self._logger.info(f"Initializing UEBAAnomalyDetectorIsolationForest with contamination={contamination} and anomaly_cutoff={anomaly_cutoff}")
        
        self._contamination: float = contamination
        self._anomaly_cutoff: float = anomaly_cutoff
        self._model = None
    

    def fit(self, df_features: pd.DataFrame, model_file_path: str = "") -> None:
        """
        Fits the model to the provided training data.

        Args:
            df_features (pd.DataFrame): The feature data for training.
        Returns:
            None
        """

        self._logger.info(f"Fitting model to training data with {len(df_features)} rows.")

        # copy the dataframe to ensure it is not modified
        df_features = df_features.copy()

        # convert 'day' column to a datetime
        df_features['day'] = pd.to_datetime(df_features['day'])

        # remove any duplicate rows
        self._logger.debug(f"Number of rows before removing duplicates: {len(df_features)}")
        df_features = df_features.drop_duplicates()
        self._logger.debug(f"Number of rows after removing duplicates: {len(df_features)}")

        # drop the actor and day for training
        df_input = df_features.drop(columns=['actor', 'day'])

        # create the isolation forest model
        self._model = IsolationForest(random_state=0, contamination=self._contamination)

        # fit the model to the dataset
        self._model.fit(df_input)

        # save the model to a file if specified
        if model_file_path.strip() != "":
            self._logger.info(f"Saving trained model to file: {model_file_path}")
            joblib.dump(self._model, model_file_path)
            self._logger.info("Model saved successfully.")

        self._logger.info("Model fitting complete.")


    def _validate_model_input(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Validates and prepares input for prediction.
        """

        if self._model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        df_input = df_features.drop(columns=['actor', 'day'], errors='ignore')
        
        # check feature count
        if df_input.shape[1] != self._model.n_features_in_:
            raise ValueError(
                f"Expected {self._model.n_features_in_} features, got {df_input.shape[1]}"
            )
        
        # check feature names match
        if hasattr(self._model, 'feature_names_in_'):
            expected_features = self._model.feature_names_in_
            actual_features = df_input.columns.tolist()
            
            if list(expected_features) != actual_features:
                raise ValueError(
                    f"Feature names mismatch.\n"
                    f"Expected: {list(expected_features)}\n"
                    f"Got: {actual_features}"
                )
        
        # check for NaN
        if df_input.isnull().any().any():
            raise ValueError("Input contains NaN values")
        
        return df_input


    def load_model_from_file(self, model_file_path: str) -> None:
        """
        Loads a pre-trained model from a file.

        Args:
            model_file_path (str): The path to the model file.
        """

        self._logger.info(f"Loading model from file: {model_file_path}")
        self._model = joblib.load(model_file_path)
        self._logger.info("Model loaded successfully.")


    def predict(self, df_features: pd.DataFrame, day: pd.Timestamp, results_file_path: str = "") -> pd.DataFrame:
        """
        Predicts anomalies in the data.

        Args:
            df_features (pd.DataFrame): The feature data for prediction.
            day (pd.Timestamp): The day for which to predict anomalies.
        Returns:
            pd.DataFrame: A dataframe including the original features and an anomaly score.
        """

        self._logger.info(f"Predicting anomalies for {format_date(day)} with {len(df_features)} rows.")

        # trim to only the data up to the specified day
        df_features_day = df_features[df_features['day'] <= day].copy()
        assert(isinstance(df_features_day, pd.DataFrame))

        # prep the input for the model
        df_input = self.generate_training_set(df_features_day, history_perc=-1)
        
        self._logger.info(f"Validating model input...")
        df_input_clean: pd.DataFrame = self._validate_model_input(df_input)
        
        self._logger.info(f"Predicting anomaly scores...")
        assert(self._model is not None)
        df_anomaly_values = self._model.decision_function(df_input_clean)

        # copy the frame and set the anomaly score in the dataframe
        self._logger.info(f"Anomaly values detected: {len(df_anomaly_values)}")
        df_results = df_input.copy()
        df_results['anomaly_score'] = df_anomaly_values

        df_results: pd.DataFrame = df_results.sort_values(by='anomaly_score', ascending=True).reset_index(drop=True)

        self._logger.info(f"The anomaly min={df_results['anomaly_score'].min()}, max={df_results['anomaly_score'].max()}")

        # add a column for is_anomaly based on the cutoff
        df_results['is_anomaly'] = df_results['anomaly_score'] < self._anomaly_cutoff
        self._logger.debug(f"Results preview:\n{df_results[['actor', 'day', 'anomaly_score', 'is_anomaly']].head(5)}")

        # save results to a file if specified
        if results_file_path.strip() != "":
            self._logger.info(f"Saving prediction results to file: {results_file_path}")
            df_results.to_csv(results_file_path, index=False)
            self._logger.info("Results saved successfully.")

        return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UEBA Anomaly Detection for DevSecOps systems')
    parser.add_argument('-p', '--predict-only', 
                        action='store_true',
                        help='Run predictions only using existing model (skip training)')
    args = parser.parse_args()

    logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '../data/anomaly_detection_debug.log')),
        logging.StreamHandler()  # also prints to console
    ])

    file_paths = {
        "base_file": os.path.join(os.path.dirname(__file__), '../data/v3-ml-features-20260104.csv'),
        "history_file": os.path.join(os.path.dirname(__file__), '../data/20260104-v3-ml-history.csv'),
        "features_file": os.path.join(os.path.dirname(__file__), '../data/20260104-v3-ml-training.csv'),
        "model_file": os.path.join(os.path.dirname(__file__), '../models/ueba-anomaly-model-20260104.pkl'),
        "results_file": os.path.join(os.path.dirname(__file__), '../data/20260104-v3-ml-results.csv')
    }

    ueba = UEBAAnomalyDetectorIsolationForest()
    df_base = ueba.read_base_features_from_csv(file_paths["base_file"])

    # split the last 30 days for testing
    test_start_date = df_base['day'].max() - pd.Timedelta(days=30)
    df_test = df_base[df_base['day'] >= test_start_date]
    df_train = df_base[df_base['day'] < test_start_date]
    predict_start_date = df_base['day'].max() - pd.Timedelta(days=20)

    if args.predict_only:
        # load existing model for predictions only
        ueba.load_model_from_file(file_paths["model_file"])
    else:
        # make sure the model directory exists
        os.makedirs(os.path.dirname(file_paths["model_file"]), exist_ok=True)

        # train the model and then run predictions
        df_features = ueba.generate_training_set(df_train, history_perc=0.5,
                                                 history_file_path=file_paths["history_file"],
                                                 training_file_path=file_paths["features_file"])
        ueba.fit(df_features, model_file_path=file_paths["model_file"])

    # for each day in the predict set, run predictions
    df_complete_results = pd.DataFrame()
    for day in pd.date_range(start=predict_start_date, end=df_base['day'].max()):
        df_results: pd.DataFrame = ueba.predict(df_base, day)
        df_complete_results = pd.concat([df_complete_results, df_results], ignore_index=True)
    
    # sort the complete results by anomaly score, then day
    df_complete_results = df_complete_results.sort_values(by=['anomaly_score', 'day'], ascending=[True, False]).reset_index(drop=True)
    logging.info(f"Complete results preview:\n{df_complete_results.head(10)}")

    # save the complete results to a file
    df_complete_results.to_csv(file_paths["results_file"], index=False)
    logging.info(f"Complete results saved to file: {file_paths['results_file']}")