import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List

def read_e4_csv(filepath, column_names: List[str]) -> pd.DataFrame:
    """
    Reads a CSV file from an Empatica E4 device and returns a DataFrame with the specified column names.

    Parameters:
    filepath (str): The path to the CSV file.
    column_names (List[str]): A list of column names for the DataFrame.

    Returns:
    pd.DataFrame: A DataFrame containing the data from the CSV file, indexed by datetime in UTC.
    """
    if not os.path.isfile(filepath):
        return np.array([])
    
    with open(filepath, 'r', encoding='utf-8') as file_to_read:
        initial_pos = file_to_read.tell()
        timestamp = pd.to_datetime(float(str(file_to_read.readline().strip().split(',')[0])), unit='s', utc=True)
        frequency = float(str(file_to_read.readline().strip().split(',')[0]))
        file_to_read.seek(initial_pos)

        data = pd.read_csv(file_to_read, skiprows=2, names=column_names)
        data.index = pd.date_range(start=timestamp, periods=len(data), freq=str(1 / frequency * 1000) + 'ms',
                                   name='datetime', tz='UTC')
        data.sort_index(inplace=True)
    return data

def read_ibi_file(filepath) -> pd.DataFrame:
    """
    Reads an IBI file from an Empatica E4 device and returns a DataFrame.

    Parameters:
    filepath (str): The path to the IBI file.

    Returns:
    pd.DataFrame: A DataFrame containing the IBI data, indexed by datetime in UTC.
    """
    if not os.path.isfile(filepath):
        return np.array([])

    with open(filepath, 'r', encoding='utf-8') as file_to_read:
        initial_pos = file_to_read.tell()
        timestamp = float(str(file_to_read.readline().strip().split(',')[0]))
        file_to_read.seek(initial_pos)

        ibi = pd.read_csv(file_to_read, skiprows=1, names=['ibi'], index_col=0)
        ibi['ibi'] *= 1000 

        ibi.index = pd.to_datetime((ibi.index * 1000 + timestamp * 1000).map(int), unit='ms', utc=True)
        ibi.index.name = 'datetime'
    return ibi


if __name__ == "__main__":
    print(
        "This script is designed to be called from a Jupyter Notebook, not to be run directly."
    )
