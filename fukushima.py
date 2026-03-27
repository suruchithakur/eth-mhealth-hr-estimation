"""
Heart Rate Estimation from PPG signals
ETH Zurich - Mobile Health and Activity Monitoring, Spring 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DEBUG = False

FS = 128
WIN_SEC = 30
WIN_N = FS * WIN_SEC

HR_MIN_BPM = 40
HR_MAX_BPM = 180
HR_MIN_HZ = HR_MIN_BPM / 60
HR_MAX_HZ = HR_MAX_BPM / 60

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def bandpass_filter(sig):
    nyq = FS / 2
    b, a = butter(4, [HR_MIN_HZ / nyq, HR_MAX_HZ / nyq], btype='band')
    return filtfilt(b, a, sig)

def acc_magnitude(imu):
    x, y, z = imu[0].astype(float), imu[1].astype(float), imu[2].astype(float)
    return np.sqrt(x**2 + y**2 + z**2)

def power_spectrum(window, pad_factor=4):
    n = len(window)
    n_fft = n * pad_factor

    win = window * np.hanning(n)
    fft = np.fft.rfft(win, n=n_fft)

    freqs = np.fft.rfftfreq(n_fft, d=1.0/FS)
    power = np.abs(fft) ** 2

    mask = (freqs >= HR_MIN_HZ) & (freqs <= HR_MAX_HZ)
    return freqs[mask], power[mask]

# ─────────────────────────────────────────────────────────────────────────────
# HR ESTIMATION CORE
# ─────────────────────────────────────────────────────────────────────────────

def refine_peak(freqs, power):
    i = np.argmax(power)
    if i == 0 or i == len(power)-1:
        return freqs[i]

    y0, y1, y2 = power[i-1], power[i], power[i+1]
    denom = (y0 - 2*y1 + y2)

    if denom == 0:
        return freqs[i]

    delta = 0.5 * (y0 - y2) / denom
    return freqs[i] + delta * (freqs[1] - freqs[0])

def correct_harmonic(freqs, power, hr_hz):
    half = hr_hz / 2

    idx_half = np.argmin(np.abs(freqs - half))
    idx_main = np.argmin(np.abs(freqs - hr_hz))

    if power[idx_half] > 0.6 * power[idx_main]:
        return half

    return hr_hz

def spectral_confidence(power):
    return power.max() / (power.mean() + 1e-10)

def motion_level(acc_win):
    return np.std(acc_win)

def hr_from_spectrum(ppg_win, acc_win=None):
    freqs, ppg_pwr = power_spectrum(ppg_win)

    if acc_win is not None:
        _, acc_pwr = power_spectrum(acc_win)
        ppg_norm = ppg_pwr / (ppg_pwr.max() + 1e-10)
        acc_norm = acc_pwr / (acc_pwr.max() + 1e-10)
        power = np.clip(ppg_norm - 0.7 * acc_norm, 0, None)
    else:
        power = ppg_pwr

    hr_hz = refine_peak(freqs, power)
    hr_hz = correct_harmonic(freqs, power, hr_hz)

    return hr_hz * 60

def hr_from_peaks(ppg_win):
    min_dist = int(FS / HR_MAX_HZ)

    peaks, _ = find_peaks(
        ppg_win,
        distance=min_dist,
        prominence=np.std(ppg_win) * 0.5
    )

    if len(peaks) < 2:
        return None

    ibi = np.diff(peaks) / FS
    valid = ibi[(ibi >= 1.0 / HR_MAX_HZ) & (ibi <= 1.0 / HR_MIN_HZ)]

    if len(valid) == 0:
        return None

    return 60.0 / np.median(valid)

# ─────────────────────────────────────────────────────────────────────────────
# SMOOTHING
# ─────────────────────────────────────────────────────────────────────────────

def smooth_hr(hr_list):
    hr = np.array(hr_list)

    med = pd.Series(hr).rolling(5, center=True, min_periods=1).median()

    alpha = 0.3
    ema = np.zeros_like(med)
    ema[0] = med.iloc[0]

    for i in range(1, len(med)):
        ema[i] = alpha * med.iloc[i] + (1 - alpha) * ema[i-1]

    return ema

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def estimate_hr(ppg, imu):
    n_win = len(ppg) // WIN_N
    acc_mag = acc_magnitude(imu)

    hr_vals = []

    for w in range(n_win):
        s = w * WIN_N
        e = s + WIN_N

        ppg_win = ppg[s:e].astype(float)
        acc_win = acc_mag[s:e]

        ppg_filt = bandpass_filter(ppg_win)

        freqs, pwr = power_spectrum(ppg_filt)
        conf = spectral_confidence(pwr)

        hr_spec = hr_from_spectrum(ppg_filt, acc_win)
        hr_peak = hr_from_peaks(ppg_filt)

        motion = motion_level(acc_win)

        if motion > 150:
            hr = hr_spec
        else:
            if conf > 5:
                hr = hr_spec
            elif hr_peak is not None:
                hr = hr_peak
            else:
                hr = hr_spec

        hr_vals.append(float(np.clip(hr, HR_MIN_BPM, HR_MAX_BPM)))

    return list(smooth_hr(hr_vals))

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    DATA_PATH = '/Users/suruchithakur/Desktop/UZH/Sem-1/Mobile Health and Activity Monitoring/Exercise-2/ppg-hr-exercise-2/mhealth26_ex2.npy'

    data = np.load(DATA_PATH, allow_pickle=True).item()

    ppg_all = data['ppg']
    imu_all = data['imu']

    rows = []

    for phase_idx in range(len(ppg_all)):
        pred = estimate_hr(ppg_all[phase_idx], imu_all[phase_idx])

        for win_idx, hr_val in enumerate(pred):
            rows.append({
                "Id": f"window{phase_idx}_{win_idx}",   # ✅ CORRECT FORMAT
                "Predicted": float(hr_val)             # ✅ CORRECT COLUMN NAME
            })

    submission = pd.DataFrame(rows)

    # enforce column order EXACTLY
    submission = submission[['Id', 'Predicted']]

    submission.to_csv("submission.csv", index=False)

    print("Submission saved!")

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()