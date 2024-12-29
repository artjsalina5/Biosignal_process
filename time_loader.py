import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import neurokit2 as nk
from matplotlib import pyplot as plt


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
    else:
        #print(f"Data for Participant ID {participant}:")
        #display(df.head())
        print("Participant Selected and loaded, converting to UTC for data synchronization...")
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


def plot_times(times_dict):
    """Plots the events as vertical lines on a time axis with labels."""
    time_start = datetime.strptime(
        str(times_dict["start_time"][0]), "%Y-%m-%d %H:%M:%S UTC"
    ).replace(tzinfo=ZoneInfo("UTC"))
    time_end = datetime.strptime(
        str(times_dict["end_time"][0]), "%Y-%m-%d %H:%M:%S UTC"
    ).replace(tzinfo=ZoneInfo("UTC"))

    plt.figure(figsize=(12, 6))

    for event, values in times_dict.items():
        if event in [
            "date",
            "start_time",
            "end_time",
            "participant_id",
            "total_time",
            "experimental_time",
            "arrival_time",
        ]:
            continue
        event_time = datetime.strptime(str(values[0]), "%Y-%m-%d %H:%M:%S UTC").replace(
            tzinfo=ZoneInfo("UTC")
        )
        plt.axvline(event_time, color="blue", linestyle="--", alpha=0.7)
        plt.text(
            event_time,
            0.5,
            event.replace("_", " ").capitalize(),
            rotation=90,
            verticalalignment="center",
            horizontalalignment="right",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5),
        )

    plt.axvline(time_start, color="green", linestyle="-", alpha=0.8, label="Start")
    plt.axvline(time_end, color="red", linestyle="-", alpha=0.8, label="End")

    plt.xlim([time_start - timedelta(minutes=5), time_end + timedelta(minutes=5)])

    plt.title("Event Timeline")
    plt.xlabel("Time (UTC)")
    plt.yticks([])
    plt.legend(loc="upper left")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python time_loader.py <file_path> <participant_id>")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        participant_id = int(sys.argv[2])
    except ValueError:
        print("Error: Participant ID must be an integer.")
        sys.exit(1)

    data = load_event_data(file_path)
    if data is not None:
        participant_data = select_participant(data, participant_id)
        if not participant_data.empty:
            times, utc_times = time_extract(participant_data)
        else:
            print(f"No data for participant {participant_id}, cannot proceed.")

        #plot_times(utc_times)