import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from scipy.signal import resample
import matplotlib.pyplot as plt
from scipy.io import savemat
from pyedflib.highlevel import read_edf


# -------------------------------------------------------------------------
# Helper Function for reading E4 Files
# -------------------------------------------------------------------------
def read_e4_csv(path):
    if not os.path.isfile(path):
        return np.array([])
    arr = pd.read_csv(path, header=None, skiprows=2).values
    return arr


def process_data(
    dataDirectory, pID="TestSubject", trim_time=None, plotting=True, resampling=False
):
    """
    Processes Hexoskin and Empatica E4 data, synchronizes them, and optionally trims the data.

    Args:
        dataDirectory (str): Path to the directory containing the data.
        pID (str, optional): Participant ID. Defaults to "TestSubject".
        trim_time (str, optional): Trim time in "MM/DD/YY HH:mm:ss" format. Defaults to None.
        plotting (bool, optional): Flag to enable or disable plotting. Defaults to True.
        resampling (bool, optional): Flag to enable resampling of E4 data. Defaults to False.

    Returns:
        pandas.DataFrame: Synced data.
    """
    fs = 256  # final sampling frequency
    Synched_data = pd.DataFrame()
    # -------------------------------------------------------------------------
    # HEXOSKIN PART
    # -------------------------------------------------------------------------
    hexcsv = os.path.join(dataDirectory, "CSV.csv")
    hexedf = os.path.join(dataDirectory, "EDF.edf")

    if not os.path.isfile(hexcsv):
        print(f"Hexoskin CSV not found at {hexcsv}")
        return
    if not os.path.isfile(hexedf):
        print(f"Hexoskin EDF not found at {hexedf}")
        return

    hextable = pd.read_csv(hexcsv)
    hextable = hextable.iloc[:, [0, 1, 2, 4, 5, 6]]
    hextable = hextable.iloc[1:-1, :].reset_index(drop=True)
    time_start_hexo = hextable.iloc[0, 0] / fs
    startTime_hexo = datetime.fromtimestamp(time_start_hexo, tz=timezone.utc)

    signals, signal_headers, header = read_edf(hexedf)

    # Assumes the standard hexoskin order
    ECG = signals[0]  # 256 Hz
    RespT = signals[3]  # 128 Hz
    RespA = signals[4]  # 128 Hz
    AccX = signals[11]  # 64 Hz
    AccY = signals[12]
    AccZ = signals[13]

    hexo_np = np.column_stack([ECG, RespT, RespA, AccX, AccY, AccZ])

    time_index_hexo = pd.date_range(
        start=startTime_hexo, periods=len(hexo_np), freq=pd.Timedelta(1 / fs, unit="s")
    )
    time_hexo = pd.DataFrame(
        hexo_np,
        index=time_index_hexo,
        columns=["ECG", "RespT", "RespA", "AccX_Hex", "AccY_Hex", "AccZ_Hex"],
    )

    # -------------------------------------------------------------------------
    # E4 PART
    # -------------------------------------------------------------------------
    BVPe4 = os.path.join(dataDirectory, "BVP.csv")
    EDAe4 = os.path.join(dataDirectory, "EDA.csv")
    ACCe4 = os.path.join(dataDirectory, "ACC.csv")
    Tempe4 = os.path.join(dataDirectory, "TEMP.csv")
    TagsCSV = os.path.join(dataDirectory, "tags.csv")

    BVP = read_e4_csv(BVPe4)
    EDA = read_e4_csv(EDAe4)
    ACC = read_e4_csv(ACCe4)
    Temp = read_e4_csv(Tempe4)

    # Tags
    timing_Exp = []
    if os.path.isfile(TagsCSV):
        tags_arr = pd.read_csv(TagsCSV).values.flatten()
        timing_Exp = [datetime.fromtimestamp(x, tz=timezone.utc) for x in tags_arr]

    # First line of BVP file has actual start time
    time_start_E4 = None
    if os.path.isfile(BVPe4):
        with open(BVPe4, "r") as f:
            first_line = f.readline().strip()
        try:
            time_start_E4 = float(first_line.split(",")[0])
        except:
            time_start_E4 = 0.0

    if time_start_E4 is not None:
        startTime_E4 = datetime.fromtimestamp(time_start_E4, tz=timezone.utc)
    else:
        startTime_E4 = datetime.fromtimestamp(0, tz=timezone.utc)

    # Resample E4 signals to 256 (if they are there)
    if resampling:
        if BVP.size > 0:
            BVP_up = resample(BVP, len(BVP) * 4)
        else:
            BVP_up = np.empty((0,))
        if EDA.size > 0:
            EDA_up = resample(EDA, len(EDA) * 64)
        else:
            EDA_up = np.empty((0,))
        if ACC.size > 0:
            ACC_up = resample(ACC, len(ACC) * 8)
        else:
            ACC_up = np.empty((0, 3))
        if Temp.size > 0:
            Temp_up = resample(Temp, len(Temp) * 64)
        else:
            Temp_up = np.empty((0,))

        # Truncate
        lengths = [BVP_up.shape[0], EDA_up.shape[0], ACC_up.shape[0], Temp_up.shape[0]]
        lengths = [l for l in lengths if l > 0]
        min_len = min(lengths) if lengths else 0
        if min_len > 0:
            BVP_up = BVP_up[:min_len]
            EDA_up = EDA_up[:min_len] if EDA_up.size > 0 else np.empty((min_len, 1))
            ACC_up = ACC_up[:min_len] if ACC_up.size > 0 else np.empty((min_len, 3))
            Temp_up = Temp_up[:min_len] if Temp_up.size > 0 else np.empty((min_len, 1))
            E4_np = np.column_stack([BVP_up, EDA_up, ACC_up, Temp_up])

            time_index_E4 = pd.date_range(
                start=startTime_E4, periods=min_len, freq=pd.Timedelta(1 / fs, unit="s")
            )
            col_names = ["BVP", "EDA", "ACCx_E4", "ACCy_E4", "ACCz_E4", "Temp"]
            e4_cols = col_names[: E4_np.shape[1]]
            time_E4 = pd.DataFrame(E4_np, index=time_index_E4, columns=e4_cols)
        else:
            print("No valid E4 data found after resampling.")
            time_E4 = pd.DataFrame()
    else:
        # Resample E4 signals to 256 (if they are there)
        if BVP.size > 0:
            BVP_up = BVP
        else:
            BVP_up = np.empty((0,))
        if EDA.size > 0:
            EDA_up = EDA
        else:
            EDA_up = np.empty((0,))
        if ACC.size > 0:
            ACC_up = ACC
        else:
            ACC_up = np.empty((0, 3))
        if Temp.size > 0:
            Temp_up = Temp
        else:
            Temp_up = np.empty((0,))

        # Truncate
        lengths = [BVP_up.shape[0], EDA_up.shape[0], ACC_up.shape[0], Temp_up.shape[0]]
        lengths = [l for l in lengths if l > 0]
        min_len = min(lengths) if lengths else 0
        if min_len > 0:
            BVP_up = BVP_up[:min_len]
            EDA_up = EDA_up[:min_len] if EDA_up.size > 0 else np.empty((min_len, 1))
            ACC_up = ACC_up[:min_len] if ACC_up.size > 0 else np.empty((min_len, 3))
            Temp_up = Temp_up[:min_len] if Temp_up.size > 0 else np.empty((min_len, 1))
            E4_np = np.column_stack([BVP_up, EDA_up, ACC_up, Temp_up])

            time_index_E4 = pd.date_range(
                start=startTime_E4, periods=min_len, freq=pd.Timedelta(1 / fs, unit="s")
            )
            col_names = ["BVP", "EDA", "ACCx_E4", "ACCy_E4", "ACCz_E4", "Temp"]
            e4_cols = col_names[: E4_np.shape[1]]
            time_E4 = pd.DataFrame(E4_np, index=time_index_E4, columns=e4_cols)
        else:
            print("No valid E4 data found after resampling.")
            time_E4 = pd.DataFrame()

    # -------------------------------------------------------------------------
    # Align & Merge
    # -------------------------------------------------------------------------
    if not time_hexo.empty and not time_E4.empty:
        if startTime_hexo < startTime_E4:
            delay_s = (startTime_E4 - startTime_hexo).total_seconds()
            delay_samples = int(round(delay_s * fs))
            time_hexo = time_hexo.iloc[delay_samples:]
        else:
            delay_s = (startTime_hexo - startTime_E4).total_seconds()
            delay_samples = int(round(delay_s * fs))
            time_E4 = time_E4.iloc[delay_samples:]

        Synched_data = pd.merge_asof(
            time_hexo.sort_index(),
            time_E4.sort_index(),
            left_index=True,
            right_index=True,
            direction="nearest",
        )
        if not Synched_data.empty:
            Synched_data = Synched_data.iloc[1:]
    else:
        # if either empty, use whichever isn't
        Synched_data = time_hexo if not time_hexo.empty else time_E4

    # Rename columns if we have exactly 12
    if Synched_data.shape[1] == 12:
        Synched_data.columns = [
            "ECG",
            "RespT",
            "RespA",
            "AccX_Hex",
            "AccY_Hex",
            "AccZ_Hex",
            "BVP",
            "EDA",
            "ACCx_E4",
            "ACCy_E4",
            "ACCz_E4",
            "Temp",
        ]

    # -------------------------------------------------------------------------
    # Optional Trim
    # -------------------------------------------------------------------------
    if trim_time:
        try:
            fmt_str = "%m/%d/%y %H:%M:%S"
            trim_dt = datetime.strptime(trim_time, fmt_str)
            Synched_data = Synched_data.loc[Synched_data.index <= trim_dt]
        except:
            print("Trim format error. Use MM/DD/YY HH:MM:SS or leave blank.")

    if plotting and not Synched_data.empty:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        axes = axes.ravel()
        fig.suptitle(f"Hexo E4 Data: {pID}", color="blue")

        # 1) ECG
        axes[0].plot(Synched_data.index, Synched_data["ECG"], label="ECG")
        axes[0].legend(loc="upper right")
        axes[0].set_title("ECG")

        # 2) BVP
        axes[1].plot(Synched_data.index, Synched_data["BVP"], label="BVP")
        axes[1].legend(loc="upper right")
        axes[1].set_title("BVP")

        # 3) EDA
        axes[2].plot(Synched_data.index, Synched_data["EDA"], label="EDA")
        axes[2].legend(loc="upper right")
        axes[2].set_title("EDA")

        # 4) Resp
        axes[3].plot(Synched_data.index, Synched_data["RespT"], label="RespT")
        axes[3].plot(Synched_data.index, Synched_data["RespA"], label="RespA")
        axes[3].legend(loc="upper right")
        axes[3].set_title("Resp")

        # 5) Acc Hex
        axes[4].plot(Synched_data.index, Synched_data["AccX_Hex"], label="AccX_Hex")
        axes[4].plot(Synched_data.index, Synched_data["AccY_Hex"], label="AccY_Hex")
        axes[4].plot(Synched_data.index, Synched_data["AccZ_Hex"], label="AccZ_Hex")
        axes[4].legend(loc="upper right")
        axes[4].set_title("Accel Hexoskin")

        # 6) Acc E4
        axes[5].plot(Synched_data.index, Synched_data["ACCx_E4"], label="ACCx_E4")
        axes[5].plot(Synched_data.index, Synched_data["ACCy_E4"], label="ACCy_E4")
        axes[5].plot(Synched_data.index, Synched_data["ACCz_E4"], label="ACCz_E4")
        axes[5].legend(loc="upper right")
        axes[5].set_title("Accel E4")

        # Add vertical lines for Tags
        for ax in axes:
            for tstamp in timing_Exp:
                ax.axvline(tstamp, color="r", linestyle="--")
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    return Synched_data


if __name__ == "__main__":
    print(
        "This script is designed to be called from a Jupyter Notebook, not to be run directly."
    )
