from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from neurokit2.complexity import (complexity_lempelziv, entropy_approximate,
                          entropy_fuzzy, entropy_multiscale, entropy_sample,
                          entropy_shannon, fractal_correlation, fractal_dfa,
                          fractal_higuchi, fractal_katz)
from neurokit2.misc import NeuroKitWarning, find_consecutive
from neurokit2.signal import signal_zerocrossings
from neurokit2.hrv.hrv_utils import _hrv_get_rri, _hrv_sanitize_peaks
sampling_rate = 256
def hrv_nonlinear_edited(peaks, sampling_rate=256, show=False, **kwargs):
    
     # Sanitize input
    peaks = _hrv_sanitize_peaks(peaks)
    if isinstance(peaks, tuple):  # Detect actual sampling rate
        peaks, sampling_rate = peaks[0], peaks[1]

    # Compute R-R intervals (also referred to as NN) in milliseconds
    rri, _ = _hrv_get_rri(peaks, sampling_rate=sampling_rate, interpolate=False)

    # Initialize empty container for results
    out = {}

    # Poincaré features (SD1, SD2, etc.)
    out = _hrv_nonlinear_poincare(rri, out)

    # Heart Rate Fragmentation
    out = _hrv_nonlinear_fragmentation(rri, out)

    # Heart Rate Asymmetry
    out = _hrv_nonlinear_poincare_hra(rri, out)

    # DFA
    out = _hrv_dfa(peaks, rri, out, **kwargs)


    # Complexity
    tolerance = 0.2 * np.std(rri, ddof=1) #could be replaced with the individually computer tolerance value
    
    out["ApEn"], _ = entropy_approximate(rri, delay=1, dimension=2, tolerance=tolerance)
    out["SampEn"], _ = entropy_sample(rri, delay=1, dimension=2, tolerance=tolerance)
    out["ShanEn"], _ = entropy_shannon(rri)
    #out["FuzzyEn"], _ = entropy_fuzzy(rri, delay=1, dimension=2, tolerance=tolerance)
    out["MSEn"], _ = entropy_multiscale(rri, delay=1, dimension=2, method="MSEn")
    out["CMSEn"], _ = entropy_multiscale(rri, dimension=2, tolerance=tolerance, method="CMSEn")
    out["RCMSEn"], _ = entropy_multiscale(rri, dimension=2, tolerance=tolerance, method="RCMSEn")

    out["CD"], _ = fractal_correlation(rri, delay=1, dimension=2, **kwargs)
    out["HFD"], _ = fractal_higuchi(rri, k_max=10, **kwargs)
    out["KFD"], _ = fractal_katz(rri)
    out["LZC"], _ = complexity_lempelziv(rri, **kwargs)

    if show:
        _hrv_nonlinear_show(rri, out)

    out = pd.DataFrame.from_dict(out, orient="index").T.add_prefix("HRV_")
    return out



# =============================================================================
# Get SD1 and SD2
# =============================================================================
def _hrv_nonlinear_poincare(rri, out):
    """Compute SD1 and SD2.

    - Do existing measures of Poincare plot geometry reflect nonlinear features of heart rate
    variability? - Brennan (2001)

    """

    # HRV and hrvanalysis
    rri_n = rri[:-1]
    rri_plus = rri[1:]
    x1 = (rri_n - rri_plus) / np.sqrt(2)  # Eq.7
    x2 = (rri_n + rri_plus) / np.sqrt(2)
    sd1 = np.std(x1, ddof=1)
    sd2 = np.std(x2, ddof=1)

    out["SD1"] = sd1
    out["SD2"] = sd2

    # SD1 / SD2
    out["SD1SD2"] = sd1 / sd2

    # Area of ellipse described by SD1 and SD2
    out["S"] = np.pi * out["SD1"] * out["SD2"]

    # CSI / CVI
    T = 4 * out["SD1"]
    L = 4 * out["SD2"]
    out["CSI"] = L / T
    out["CVI"] = np.log10(L * T)
    out["CSI_Modified"] = L ** 2 / T

    return out


def _hrv_nonlinear_poincare_hra(rri, out):
    """Heart Rate Asymmetry Indices.

    - Asymmetry of Poincaré plot (or termed as heart rate asymmetry, HRA) - Yan (2017)
    - Asymmetric properties of long-term and total heart rate variability - Piskorski (2011)

    """

    N = len(rri) - 1
    x = rri[:-1]  # rri_n, x-axis
    y = rri[1:]  # rri_plus, y-axis

    diff = y - x
    decelerate_indices = np.where(diff > 0)[0]  # set of points above IL where y > x
    accelerate_indices = np.where(diff < 0)[0]  # set of points below IL where y < x
    nochange_indices = np.where(diff == 0)[0]

    # Distances to centroid line l2
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)
    dist_l2_all = abs((x - centroid_x) + (y - centroid_y)) / np.sqrt(2)

    # Distances to LI
    dist_all = abs(y - x) / np.sqrt(2)

    # Calculate the angles
    theta_all = abs(np.arctan(1) - np.arctan(y / x))  # phase angle LI - phase angle of i-th point
    # Calculate the radius
    r = np.sqrt(x ** 2 + y ** 2)
    # Sector areas
    S_all = 1 / 2 * theta_all * r ** 2

    # Guzik's Index (GI)
    den_GI = np.sum(dist_all)
    num_GI = np.sum(dist_all[decelerate_indices])
    out["GI"] = (num_GI / den_GI) * 100

    # Slope Index (SI)
    den_SI = np.sum(theta_all)
    num_SI = np.sum(theta_all[decelerate_indices])
    out["SI"] = (num_SI / den_SI) * 100

    # Area Index (AI)
    den_AI = np.sum(S_all)
    num_AI = np.sum(S_all[decelerate_indices])
    out["AI"] = (num_AI / den_AI) * 100

    # Porta's Index (PI)
    m = N - len(nochange_indices)  # all points except those on LI
    b = len(accelerate_indices)  # number of points below LI
    out["PI"] = (b / m) * 100

    # Short-term asymmetry (SD1)
    sd1d = np.sqrt(np.sum(dist_all[decelerate_indices] ** 2) / (N - 1))
    sd1a = np.sqrt(np.sum(dist_all[accelerate_indices] ** 2) / (N - 1))

    sd1I = np.sqrt(sd1d ** 2 + sd1a ** 2)
    out["C1d"] = (sd1d / sd1I) ** 2
    out["C1a"] = (sd1a / sd1I) ** 2
    out["SD1d"] = sd1d  # SD1 deceleration
    out["SD1a"] = sd1a  # SD1 acceleration
    # out["SD1I"] = sd1I  # SD1 based on LI, whereas SD1 is based on centroid line l1

    # Long-term asymmetry (SD2)
    longterm_dec = np.sum(dist_l2_all[decelerate_indices] ** 2) / (N - 1)
    longterm_acc = np.sum(dist_l2_all[accelerate_indices] ** 2) / (N - 1)
    longterm_nodiff = np.sum(dist_l2_all[nochange_indices] ** 2) / (N - 1)

    sd2d = np.sqrt(longterm_dec + 0.5 * longterm_nodiff)
    sd2a = np.sqrt(longterm_acc + 0.5 * longterm_nodiff)

    sd2I = np.sqrt(sd2d ** 2 + sd2a ** 2)
    out["C2d"] = (sd2d / sd2I) ** 2
    out["C2a"] = (sd2a / sd2I) ** 2
    out["SD2d"] = sd2d  # SD2 deceleration
    out["SD2a"] = sd2a  # SD2 acceleration
    # out["SD2I"] = sd2I  # identical with SD2

    # Total asymmerty (SDNN)
    sdnnd = np.sqrt(0.5 * (sd1d ** 2 + sd2d ** 2))  # SDNN deceleration
    sdnna = np.sqrt(0.5 * (sd1a ** 2 + sd2a ** 2))  # SDNN acceleration
    sdnn = np.sqrt(sdnnd ** 2 + sdnna ** 2)  # should be similar to sdnn in hrv_time
    out["Cd"] = (sdnnd / sdnn) ** 2
    out["Ca"] = (sdnna / sdnn) ** 2
    out["SDNNd"] = sdnnd
    out["SDNNa"] = sdnna

    return out


def _hrv_nonlinear_fragmentation(rri, out):
    """Heart Rate Fragmentation Indices - Costa (2017)

    The more fragmented a time series is, the higher the PIP, IALS, PSS, and PAS indices will be.
    """

    diff_rri = np.diff(rri)
    zerocrossings = signal_zerocrossings(diff_rri)

    # Percentage of inflection points (PIP)
    out["PIP"] = len(zerocrossings) / len(rri)

    # Inverse of the average length of the acceleration/deceleration segments (IALS)
    accelerations = np.where(diff_rri > 0)[0]
    decelerations = np.where(diff_rri < 0)[0]
    consecutive = find_consecutive(accelerations) + find_consecutive(decelerations)
    lengths = [len(i) for i in consecutive]
    out["IALS"] = 1 / np.average(lengths)

    # Percentage of short segments (PSS) - The complement of the percentage of NN intervals in
    # acceleration and deceleration segments with three or more NN intervals
    out["PSS"] = np.sum(np.asarray(lengths) < 3) / len(lengths)

    # Percentage of NN intervals in alternation segments (PAS). An alternation segment is a sequence
    # of at least four NN intervals, for which heart rate acceleration changes sign every beat. We note
    # that PAS quantifies the amount of a particular sub-type of fragmentation (alternation). A time
    # series may be highly fragmented and have a small amount of alternation. However, all time series
    # with large amount of alternation are highly fragmented.
    alternations = find_consecutive(zerocrossings)
    lengths = [len(i) for i in alternations]
    out["PAS"] = np.sum(np.asarray(lengths) >= 4) / len(lengths)

    return out


# =============================================================================
# DFA
# =============================================================================
def _hrv_dfa(peaks, rri, out, n_windows="default", **kwargs):

    # if "dfa_windows" in kwargs:
    #    dfa_windows = kwargs["dfa_windows"]
    # else:
    # dfa_windows = [(4, 11), (12, None)]
    # consider using dict.get() mthd directly
    dfa_windows = kwargs.get("dfa_windows", [(4, 11), (12, None)])

    # Determine max beats
    if dfa_windows[1][1] is None:
        max_beats = len(peaks) / 10
    else:
        max_beats = dfa_windows[1][1]

    # No. of windows to compute for short and long term
    if n_windows == "default":
        n_windows_short = int(dfa_windows[0][1] - dfa_windows[0][0] + 1)
        n_windows_long = int(max_beats - dfa_windows[1][0] + 1)
    elif isinstance(n_windows, list):
        n_windows_short = n_windows[0]
        n_windows_long = n_windows[1]

    # Compute DFA alpha1
    short_window = np.linspace(dfa_windows[0][0], dfa_windows[0][1], n_windows_short).astype(int)
    # For monofractal
    out["DFA_alpha1"], _ = fractal_dfa(rri, multifractal=False, scale=short_window, **kwargs)
    # For multifractal
    mdfa_alpha1, _ = fractal_dfa(
        rri, multifractal=True, q=np.arange(-5, 6), scale=short_window, **kwargs
    )
    for k in mdfa_alpha1.columns:
        out["MFDFA_alpha1_" + k] = mdfa_alpha1[k].values[0]

    # Compute DFA alpha2
    # sanatize max_beats
    if max_beats < dfa_windows[1][0] + 1:
        warn(
            "DFA_alpha2 related indices will not be calculated. "
            "The maximum duration of the windows provided for the long-term correlation is smaller "
            "than the minimum duration of windows. Refer to the `scale` argument in `nk.fractal_dfa()` "
            "for more information.",
            category=NeuroKitWarning,
        )
        return out
    else:
        long_window = np.linspace(dfa_windows[1][0], int(max_beats), n_windows_long).astype(int)
        # For monofractal
        out["DFA_alpha2"], _ = fractal_dfa(rri, multifractal=False, scale=long_window, **kwargs)
        # For multifractal
        mdfa_alpha2, _ = fractal_dfa(
            rri, multifractal=True, q=np.arange(-5, 6), scale=long_window, **kwargs
        )
        for k in mdfa_alpha2.columns:
            out["MFDFA_alpha2_" + k] = mdfa_alpha2[k].values[0]

    return out


# =============================================================================
# Plot
# =============================================================================
def _hrv_nonlinear_show(rri, out, ax=None, ax_marg_x=None, ax_marg_y=None):

    mean_heart_period = np.mean(rri)
    sd1 = out["SD1"]
    sd2 = out["SD2"]
    if isinstance(sd1, pd.Series):
        sd1 = float(sd1)
    if isinstance(sd2, pd.Series):
        sd2 = float(sd2)

    # Poincare values
    ax1 = rri[:-1]
    ax2 = rri[1:]

    # Set grid boundaries
    ax1_lim = (max(ax1) - min(ax1)) / 10
    ax2_lim = (max(ax2) - min(ax2)) / 10
    ax1_min = min(ax1) - ax1_lim
    ax1_max = max(ax1) + ax1_lim
    ax2_min = min(ax2) - ax2_lim
    ax2_max = max(ax2) + ax2_lim

    # Prepare figure
    if ax is None and ax_marg_x is None and ax_marg_y is None:
        gs = matplotlib.gridspec.GridSpec(4, 4)
        fig = plt.figure(figsize=(8, 8))
        ax_marg_x = plt.subplot(gs[0, 0:3])
        ax_marg_y = plt.subplot(gs[1:4, 3])
        ax = plt.subplot(gs[1:4, 0:3])
        gs.update(wspace=0.025, hspace=0.05)  # Reduce spaces
        plt.suptitle("Poincaré Plot")
    else:
        fig = None

    # Create meshgrid
    xx, yy = np.mgrid[ax1_min:ax1_max:100j, ax2_min:ax2_max:100j]

    # Fit Gaussian Kernel
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([ax1, ax2])
    kernel = scipy.stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    cmap = matplotlib.cm.get_cmap("Blues", 10)
    ax.contourf(xx, yy, f, cmap=cmap)
    ax.imshow(np.rot90(f), extent=[ax1_min, ax1_max, ax2_min, ax2_max], aspect="auto")

    # Marginal densities
    ax_marg_x.hist(
        ax1, bins=int(len(ax1) / 10), density=True, alpha=1, color="#ccdff0", edgecolor="none"
    )
    ax_marg_y.hist(
        ax2,
        bins=int(len(ax2) / 10),
        density=True,
        alpha=1,
        color="#ccdff0",
        edgecolor="none",
        orientation="horizontal",
        zorder=1,
    )
    kde1 = scipy.stats.gaussian_kde(ax1)
    x1_plot = np.linspace(ax1_min, ax1_max, len(ax1))
    x1_dens = kde1.evaluate(x1_plot)

    ax_marg_x.fill(x1_plot, x1_dens, facecolor="none", edgecolor="#1b6aaf", alpha=0.8, linewidth=2)
    kde2 = scipy.stats.gaussian_kde(ax2)
    x2_plot = np.linspace(ax2_min, ax2_max, len(ax2))
    x2_dens = kde2.evaluate(x2_plot)
    ax_marg_y.fill_betweenx(
        x2_plot, x2_dens, facecolor="none", edgecolor="#1b6aaf", linewidth=2, alpha=0.8, zorder=2
    )

    # Turn off marginal axes labels
    ax_marg_x.axis("off")
    ax_marg_y.axis("off")

    # Plot ellipse
    angle = 45
    width = 2 * sd2 + 1
    height = 2 * sd1 + 1
    xy = (mean_heart_period, mean_heart_period)
    ellipse = matplotlib.patches.Ellipse(
        xy=xy, width=width, height=height, angle=angle, linewidth=2, fill=False
    )
    ellipse.set_alpha(0.5)
    ellipse.set_facecolor("#2196F3")
    ax.add_patch(ellipse)

    # Plot points only outside ellipse
    cos_angle = np.cos(np.radians(180.0 - angle))
    sin_angle = np.sin(np.radians(180.0 - angle))
    xc = ax1 - xy[0]
    yc = ax2 - xy[1]
    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle
    rad_cc = (xct ** 2 / (width / 2.0) ** 2) + (yct ** 2 / (height / 2.0) ** 2)

    points = np.where(rad_cc > 1)[0]
    ax.plot(ax1[points], ax2[points], "o", color="k", alpha=0.5, markersize=4)

    # SD1 and SD2 arrow
    sd1_arrow = ax.arrow(
        mean_heart_period,
        mean_heart_period,
        float(-sd1 * np.sqrt(2) / 2),
        float(sd1 * np.sqrt(2) / 2),
        linewidth=3,
        ec="#E91E63",
        fc="#E91E63",
        label="SD1",
    )
    sd2_arrow = ax.arrow(
        mean_heart_period,
        mean_heart_period,
        float(sd2 * np.sqrt(2) / 2),
        float(sd2 * np.sqrt(2) / 2),
        linewidth=3,
        ec="#FF9800",
        fc="#FF9800",
        label="SD2",
    )

    ax.set_xlabel(r"$RR_{n} (ms)$")
    ax.set_ylabel(r"$RR_{n+1} (ms)$")
    ax.legend(handles=[sd1_arrow, sd2_arrow], fontsize=12, loc="best")

    return fig


def compute_higher_HRV(hrv_final_df): 
    
    # assuming "hrv_final_df" is the dataframe where my HRV metric values are listed segment-wise per subject
    
    # compute basic stats
    
    higher_order_HRV_basic_stats=[]
    
    #print(type(hrv_final_df.iloc[:,[0]].to_numpy()))
    #print(hrv_final_df.iloc[:,[i]])
    #print(hrv_final_df.iloc[:,[0]].to_numpy().dtype)
    
    
    for i in range(hrv_final_df.shape[1]): #last column is the SubjectID string, so skipping it below
        #print(type(hrv_final_df.iloc[:,[i]]))
        
        hrv_metric=hrv_final_df.iloc[:,[i]].values
        #print(type(hrv_metric))
        
        #String skip logic to skip over SubjectID column
        if skip_last_column(hrv_final_df.iloc[:,[i]].values) == False:
            HRV_results_temp1=compute_basic_stats(hrv_final_df.iloc[:,[i]].astype(np.float64),SubjectID)
            higher_order_HRV_basic_stats.append(HRV_results_temp1)
    
        else:
            i+=1

    HRV_basic_stats=pd.DataFrame(higher_order_HRV_basic_stats, columns=['SubjectID','HRV_mean', 'HRV_coeff_variation', 'quartile_coefficient_dispersion','HRV metrics entropy'])
    hrv_columns=hrv_final_df.columns[0:63] #make sure I don't select the last column which has SubjectID
    #print(hrv_columns)
    HRV_basic_stats.index=[hrv_columns]
    HRV_basic_stats_final=HRV_basic_stats.T                                 #transpose
    
    # Estimate HMM probabilities output for a given segmented HRV metric
    # Then compute basic_stats on this estimate;
    # Hypothesis: stable tracings will have tight distributions of HMM values and resemble entropy estimates;
    # This will apply statistically significantly for stressed (tighter distributions) versus control subjects

    higher_order_HRV_basic_stats_on_HMM=[]
    
    
    for i in range(hrv_final_df.shape[1]): #last column is the SubjectID string, so removing it
        
        hrv_metric=hrv_final_df.iloc[:,[i]].values
        print("HRV_metric has the type", type(hrv_metric))
        # some HRV metrics have NaNs and the "do_hmm" script crashes on those; 
        # Adding logic to skip if NaN is present
        a=any(pd.isna(hrv_metric)) #checking if _any_ values in HRV metrics list are NaN
        b=skip_last_column(hrv_metric)
        skip_reasons={a:'True', b:'True'}
        #NaN or string skip logic
        if any(skip_reasons):
            i+=1
        else:
            HRV_results_hmm_temp2=do_hmm(hrv_metric)
            print(HRV_results_hmm_temp2)
            print(type(HRV_results_hmm_temp2))
            HRV_results_stats_hmm_temp=compute_basic_stats(HRV_results_hmm_temp2.tolist(),SubjectID) #j being the file number; != SubjectID
            higher_order_HRV_basic_stats_on_HMM.append(HRV_results_stats_hmm_temp)
    
    HRV_basic_stats_on_HMM=pd.DataFrame(higher_order_HRV_basic_stats_on_HMM, columns=['HRV_HMM_mean', 'HRV_HMM_coeff_variation', 'HMM_quartile_coefficient_dispersion','HMM_HRV metrics entropy'])
    HRV_basic_stats_on_HMM.index=[hrv_columns]
    HRV_basic_stats_on_HMM_final=HRV_basic_stats_on_HMM.T                  #transpose
        
    higher_hrv_final_df=pd.concat([HRV_basic_stats_final, HRV_basic_stats_on_HMM_final],axis=1)    
    
    #higher_hrv_final_df=HRV_basic_stats_final #leaving the syntax above for when the data allow HMM analysis    
        
    return higher_hrv_final_df #this includes SubjectID

#one could change the df saved by splitting into four separate DFs: just for mean values, just for CV, etc. That will make later corr easier.