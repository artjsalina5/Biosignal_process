import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List
from zoneinfo import ZoneInfo

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

def plot_signals(df, period_title, time_periods, title="Signal Plot", exclude_columns=None):
    """
    Plots the signals in the DataFrame within the specified time range.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the signals.
    period_title (str): The desired segment of the test routine.
    time_periods (List[dict]): A list of dictionaries with 'name', 'start', and 'end' keys.
    title (str): The title of the plot.
    """
    if exclude_columns is None:
        exclude_columns = []

    period = next((p for p in time_periods if p['name'] == period_title), None)
    if period is None:
        raise ValueError(f"Period '{period_title}' not found in time_periods")
    
    start_time = pd.to_datetime(period['start'])
    end_time = pd.to_datetime(period['end'])

    if start_time.tzinfo is None:
        start_time = start_time.tz_localize('UTC')
    else:
        start_time = start_time.tz_convert('UTC')

    if end_time.tzinfo is None:
        end_time = end_time.tz_localize('UTC')
    else:
        end_time = end_time.tz_convert('UTC')

    
    filtered_df = df[start_time:end_time]

    filtered_df = filtered_df.drop(columns=exclude_columns, errors='ignore')

    num_signals = len(filtered_df.columns)
    fig, axes = plt.subplots(num_signals, 1, figsize=(15, 1.2 * num_signals), sharex=True)
    fig.subplots_adjust(top=0.96)
    fig.suptitle(title, fontsize=14)
    
    cmap = plt.colormaps['tab10']
    colors = cmap(range(num_signals))

    for i, column in enumerate(filtered_df.columns):
        axes[i].plot(filtered_df.index, filtered_df[column], label=column, color=colors[i])
        axes[i].set_ylabel(column)
        axes[i].legend(loc='upper right')
        axes[i].grid(True)

    locator = mdates.MinuteLocator(interval=1)
    minor_locator = mdates.SecondLocator(interval=15)
    formatter = mdates.DateFormatter('%H:%M:%S')
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_minor_locator(minor_locator)
    axes[-1].xaxis.set_major_formatter(formatter)
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)

    plt.xlabel('Time')
    plt.show()

def load_event_data(file_path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df = clean_column_names(df)
        df = df.loc[:, ~df.columns.str.startswith("unnamed")]
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except pd.errors.ParserError:
        print(f"Error: Could not parse the file {file_path}. Please check the format.")
    return None


def clean_column_names(df):
    """Cleans the DataFrame's column names."""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def select_participant(data, participant):
    """Selects rows of data based on the participant number."""
    data = data.dropna(subset=["participant_id"])
    data = data[~np.isinf(data["participant_id"])]
    try:
        data["participant_id"] = data["participant_id"].astype(int)
    except ValueError:
        return pd.DataFrame()
    df = data[data["participant_id"] == participant]
    if df.empty:
        print(f"No data for Participant ID: {participant}")
    #else:
        #print(f"Data for Participant ID {participant}:")
        #display(df.head())
        #print("Participant Selected and loaded, converting to UTC for data synchronization...")
    return df


def time_extract(data):
    """
    Identifies and prints the start and end times of the test in UTC, along with other column values.
    Converts all time columns to UTC where applicable.
    """
    #print("Columns in data:", data.columns.tolist())

    if "date" not in data.columns:
        print("Error: 'date' column not found.")
        return None

    date = data["date"].iloc[0]

    times = {}
    utc_times = {}

    for column in data.columns:
        if (
            column.startswith("unnamed")
            or column == "participant_id"
            or column == "date"
        ):
            times[column] = data[column].values.tolist()
            utc_times[column] = data[column].values.tolist()
            continue

        try:
            times[column] = data[column].values.tolist()
            if data[column].dtype == object:
                utc_times[column] = [
                    (
                        convert_to_utc(date, t).strftime("%Y-%m-%d %H:%M:%S")
                        if convert_to_utc(date, t)
                        else t
                    )
                    for t in data[column]
                ]
            else:
                utc_times[column] = data[column].values.tolist()
        except Exception as e:
            print(f"Error processing column '{column}': {e}")
            utc_times[column] = data[column].values.tolist()

    #for column in utc_times:
     #   print(f"{column}: {utc_times[column]}")

    return times, utc_times


def convert_to_utc(
    date_str, time_str, date_format="%m/%d/%Y", time_format="%I:%M:%S %p"
):
    """Converts a date and time from Eastern Time to UTC."""
    try:
        combined_str = f"{date_str} {time_str}"
        combined_format = f"{date_format} {time_format}"
        local_datetime = datetime.strptime(combined_str, combined_format)
        eastern = ZoneInfo("America/New_York")
        localized_datetime = local_datetime.replace(tzinfo=eastern)
        utc_datetime = localized_datetime.astimezone(ZoneInfo("UTC"))
        return utc_datetime
    except ValueError:
        return None
    
if __name__ == "__main__":
    print(
        "This script is designed to be called from a Jupyter Notebook, not to be run directly."
    )
