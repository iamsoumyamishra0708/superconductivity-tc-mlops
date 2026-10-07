"""
Data Ingestion and Processing Module.

This module handles the automated downloading, extraction, and loading of the 
UCI Superconductivity dataset. It provides functionalities to create train-test 
splits, including a custom grouped split strategy to prevent data leakage 
across chemical families, ensuring robust real-world model evaluation.

Usage:
    From the project root, run:
    $ python src/data.py
"""

import os
import zipfile
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

# Define paths relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ZIP_URL = "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"


def download_data():
    """
    Downloads and extracts the UCI dataset if it does not already exist locally.

    This function checks the designated data directory for the required CSV files. 
    If they are missing, it fetches the ZIP archive from the UCI repository, 
    extracts the contents, and removes the ZIP file to maintain a clean workspace.

    Returns:
        None
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "data.zip")
    train_path = os.path.join(DATA_DIR, "train.csv")
    unique_m_path = os.path.join(DATA_DIR, "unique_m.csv")
    
    # Check if files are missing before downloading
    if not os.path.exists(train_path) or not os.path.exists(unique_m_path):
        print("Downloading dataset from UCI...")
        urllib.request.urlretrieve(ZIP_URL, zip_path)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
            
        # Clean up the zip file after extraction
        os.remove(zip_path)
        print("Download and extraction complete.")
    else:
        print("Dataset already exists in data/ directory.")


def load_data():
    """
    Loads the main dataset and unique material identifiers into pandas DataFrames.

    This function calls download_data() to ensure the files are present before 
    attempting to read them.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - train_df (pd.DataFrame): Contains the features and target variable.
            - unique_m_df (pd.DataFrame): Contains the chemical compositions used for grouping.
    """
    download_data()
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    unique_m_df = pd.read_csv(os.path.join(DATA_DIR, "unique_m.csv"))
    return train_df, unique_m_df


def get_train_test_splits(test_size=0.2, use_grouped_split=True):
    """
    Splits the dataset into training and testing sets.

    Provides an option to use a standard random split or a grouped split based 
    on chemical formulas. The grouped split ensures that identical chemical 
    materials do not leak across both the training and testing sets.

    Args:
        test_size (float, optional): The proportion of the dataset to include in the 
            test split. Defaults to 0.2.
        use_grouped_split (bool, optional): If True, uses GroupShuffleSplit based 
            on the 'material' column. Defaults to True.

    Returns:
        tuple: A tuple containing four pandas objects:
            - X_train (pd.DataFrame): Training features.
            - X_test (pd.DataFrame): Testing features.
            - y_train (pd.Series): Training target variable.
            - y_test (pd.Series): Testing target variable.
    """
    train_df, unique_m_df = load_data()
    
    # Separate features (X) and target variable (y)
    X = train_df.drop(columns=['critical_temp'])
    y = train_df['critical_temp']
    
    if use_grouped_split:
        # 'material' column contains the chemical formula string used for grouping
        groups = unique_m_df['material']
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        print("Using Chemical-Family Grouped Split.")
    else:
        # Standard random split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        print("Using Standard Random Split.")
        
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test the module by running it directly
    print("Testing data ingestion module...")
    X_tr, X_te, y_tr, y_te = get_train_test_splits(use_grouped_split=True)
    print(f"Train size: {X_tr.shape}, Test size: {X_te.shape}")