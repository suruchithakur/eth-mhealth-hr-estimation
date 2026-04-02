"""
Heart Rate Estimation - Full Evaluation Script v3
ETH Zurich - Mobile Health and Activity Monitoring, Spring 2026

Changes from v2:
  - best_fundamental() no longer takes a continuity bias — it was locking
    onto wrong harmonics for entire phases
  - continuity_correction() is now much more conservative: only corrects
    when the jump is large AND the alternative (half/double) is clearly
    closer AND the neighboring windows agree
  - smoothing window reduced — less bleeding between windows
  - peak detection gets more trust on clean low-motion signal
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = '/Users/suruchithakur/Desktop/UZH/Sem-1/Mobile Health and Activity Monitoring/Exercise-2/ppg-hr-exercise-2/mhealth26_ex2.npy'

FS      = 128
WIN_SEC = 30
WIN_N   = FS * WIN_SEC

HR_MIN_BPM = 40
HR_MAX_BPM = 180
HR_MIN_HZ  = HR_MIN_BPM / 60
HR_MAX_HZ  = HR_MAX_BPM / 60

FILTER_LOW_HZ  = 35  / 60
FILTER_HIGH_HZ = 200 / 60

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def bandpass_filter(sig):
    nyq  = FS / 2
    b, a = butter(4, [FILTER_LOW_HZ / nyq, FILTER_HIGH_HZ / nyq], btype='band')
    return filtfilt(b, a, sig)


def acc_magnitude(imu):
    x, y, z = imu[0].astype(float), imu[1].astype(float), imu[2].astype(float)
    return np.sqrt(x**2 + y**2 + z**2)


def power_spectrum(window, pad_factor=4):
    n     = len(window)
    n_fft = n * pad_factor
    win   = window * np.hanning(n)
    fft   = np.fft.rfft(win, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / FS)
    power = np.abs(fft) ** 2
    mask  = (freqs >= HR_MIN_HZ) & (freqs <= HR_MAX_HZ)
    return freqs[mask], power[mask]


def signal_snr(ppg_filt):
    noise      = ppg_filt - np.convolve(ppg_filt, np.ones(5)/5, mode='same')
    signal_var = np.var(ppg_filt)
    noise_var  = np.var(noise)
    if noise_var < 1e-10:
        return 100.0
    return signal_var / noise_var

# ─────────────────────────────────────────────────────────────────────────────
# HARMONIC-AWARE SPECTRAL PEAK SELECTION  (no continuity bias)
#
# Score each candidate f0 by how well it explains the full spectrum.
# A true fundamental scores its own power PLUS fractional harmonic power.
# A harmonic gets penalised because its sub-frequency has more power.
# ─────────────────────────────────────────────────────────────────────────────

def best_fundamental(freqs, power):
    """
    Score each candidate f0 by how well it explains the full spectrum.

    Critical fix: extend harmonic lookup ABOVE HR_MAX_HZ so that high-HR
    fundamentals (~100 bpm) get credit for their 2nd harmonic (~200 bpm)
    even though it's outside the output HR range.
    Without this, a 50 bpm subharmonic scores better than the true 100 bpm
    fundamental because the 100 bpm candidate can't see its own harmonics.
    This was the root cause of Phase 9 being wrong on every window.
    """
    f_fine = np.linspace(HR_MIN_HZ, HR_MAX_HZ, 2000)
    p_fine = np.interp(f_fine, freqs, power)

    # Wide grid for harmonic lookups only (up to 3× max HR)
    f_wide = np.linspace(HR_MIN_HZ, HR_MAX_HZ * 3, 6000)
    p_wide = np.interp(f_wide, freqs, power, right=0.0)

    scores = np.zeros_like(f_fine)
    for i, f0 in enumerate(f_fine):
        s = p_fine[i]

        # 2nd harmonic — use wide grid so high-HR candidates aren't blind
        s += 0.5  * float(np.interp(f0 * 2, f_wide, p_wide))

        # 3rd harmonic
        s += 0.25 * float(np.interp(f0 * 3, f_wide, p_wide))

        # Penalise if sub-frequency has strong power → we're on a harmonic
        f_half = f0 / 2.0
        if f_half >= HR_MIN_HZ:
            p_half = float(np.interp(f_half, f_fine, p_fine))
            if p_half > p_fine[i] * 0.6:
                s *= 0.3

        scores[i] = s

    best_idx = np.argmax(scores)
    return f_fine[best_idx], scores[best_idx]


def spectral_confidence(freqs, power):
    return power.max() / (power.mean() + 1e-10)


def motion_level(acc_win):
    return np.std(acc_win)


def subtract_motion(ppg_pwr, acc_pwr):
    ppg_norm = ppg_pwr / (ppg_pwr.max() + 1e-10)
    acc_norm = acc_pwr / (acc_pwr.max() + 1e-10)
    overlap  = np.dot(ppg_norm, acc_norm) / (np.linalg.norm(acc_norm) + 1e-10)
    weight   = np.clip(overlap * 1.5, 0.0, 0.9)
    cleaned  = np.clip(ppg_norm - weight * acc_norm, 0, None)
    if cleaned.max() < 1e-6:
        return ppg_norm
    return cleaned

# ─────────────────────────────────────────────────────────────────────────────
# PEAK-BASED HR
# ─────────────────────────────────────────────────────────────────────────────

def hr_from_peaks(ppg_win):
    min_dist = int(FS / HR_MAX_HZ)
    peaks, _ = find_peaks(
        ppg_win,
        distance=min_dist,
        prominence=np.std(ppg_win) * 0.4
    )
    if len(peaks) < 2:
        return None, 0.0

    ibi   = np.diff(peaks) / FS
    valid = ibi[(ibi >= 1.0 / HR_MAX_HZ) & (ibi <= 1.0 / HR_MIN_HZ)]
    if len(valid) < 2:
        return None, 0.0

    hr     = 60.0 / np.median(valid)
    ibi_cv = np.std(valid) / (np.mean(valid) + 1e-10)
    conf   = 1.0 / (1.0 + ibi_cv)
    return float(hr), float(conf)

# ─────────────────────────────────────────────────────────────────────────────
# PER-WINDOW ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

def estimate_window(ppg_win, acc_win):
    ppg_filt = bandpass_filter(ppg_win)

    freqs, ppg_pwr = power_spectrum(ppg_filt)
    _, acc_pwr     = power_spectrum(acc_win)
    motion         = motion_level(acc_win)

    if motion > 80:
        pwr_for_spec = subtract_motion(ppg_pwr, acc_pwr)
    else:
        pwr_for_spec = ppg_pwr / (ppg_pwr.max() + 1e-10)

    hr_spec_hz, _ = best_fundamental(freqs, pwr_for_spec)
    hr_spec        = hr_spec_hz * 60.0

    hr_peak, peak_conf = hr_from_peaks(ppg_filt)
    snr       = signal_snr(ppg_filt)
    spec_conf = spectral_confidence(freqs, pwr_for_spec)

    # Detect spectrum failure: raw peak is stuck at the filter floor (40 bpm).
    # This means the spectrum is dominated by low-freq noise and is useless.
    # Phase 1 pattern: raw_peak == 40.0 almost every window.
    raw_peak_hz  = freqs[np.argmax(pwr_for_spec)]
    spectrum_failed = (raw_peak_hz * 60.0) < (HR_MIN_BPM + 2.0)

    # Fusion
    if spectrum_failed:
        # Spectrum is useless — trust peak detector if we have it
        if hr_peak is not None:
            hr = hr_peak
        else:
            hr = hr_spec   # nothing better available
    elif motion < 100 and snr > 5 and hr_peak is not None:
        diff = abs(hr_spec - hr_peak)
        if diff < 10:
            # Both agree — blend, weighting spectrum more (peak detector is
            # noisy on low-HR signals like Phase 13)
            hr = 0.6 * hr_spec + 0.4 * hr_peak
        elif peak_conf > 0.75:
            # Peaks very regular — trust them
            hr = hr_peak
        else:
            # Spectrum wins when disagreement is large and peaks aren't clean
            hr = hr_spec
    else:
        hr = hr_spec

    hr = float(np.clip(hr, HR_MIN_BPM, HR_MAX_BPM))
    return hr, spec_conf

# ─────────────────────────────────────────────────────────────────────────────
# CONTINUITY CORRECTION  (conservative — requires neighbor agreement)
#
# Only correct window i if:
#   1. It jumps by >25 bpm from the previous window, AND
#   2. The half or double alternative is much closer to the previous value, AND
#   3. The NEXT window also disagrees with window i (so we're not correcting
#      a real sudden HR change)
# ─────────────────────────────────────────────────────────────────────────────

def continuity_correction(hr_vals):
    hr  = np.array(hr_vals, dtype=float)
    out = hr.copy()
    n   = len(hr)

    for i in range(1, n):
        prev = out[i - 1]
        curr = hr[i]
        jump = abs(curr - prev)

        if jump < 25:
            continue   # not a suspicious jump

        half   = curr / 2.0
        double = curr * 2.0

        d_curr   = jump
        d_half   = abs(half   - prev) if HR_MIN_BPM <= half   <= HR_MAX_BPM else 1e9
        d_double = abs(double - prev) if HR_MIN_BPM <= double <= HR_MAX_BPM else 1e9

        # Only correct if alternative is clearly better (not just marginally)
        if d_half < d_curr * 0.4 and d_half < d_double:
            candidate = half
        elif d_double < d_curr * 0.4 and d_double < d_half:
            candidate = double
        else:
            continue   # no clear harmonic alternative → keep original

        # Extra gate: does the next window also sit closer to prev than to curr?
        # If yes, curr is likely a spurious spike. If no, HR may have genuinely jumped.
        if i < n - 1:
            next_hr   = hr[i + 1]
            next_jump = abs(next_hr - prev)
            curr_jump = abs(next_hr - curr)
            # If next window is closer to curr than to prev, HR genuinely changed
            if curr_jump < next_jump:
                continue

        out[i] = np.clip(candidate, HR_MIN_BPM, HR_MAX_BPM)

    return out


def smooth_hr_weighted(hr_vals, conf_vals, half_window=1):
    """
    Confidence-weighted smoothing with a small window (±1 window = 3 points).
    Smaller than v2 to avoid blending across genuine HR transitions.
    """
    hr   = np.array(hr_vals,   dtype=float)
    conf = np.array(conf_vals, dtype=float)
    norm = conf / (conf.max() + 1e-10)

    smoothed = hr.copy()
    for i in range(len(hr)):
        lo = max(0, i - half_window)
        hi = min(len(hr), i + half_window + 1)
        w  = norm[lo:hi]
        smoothed[i] = np.average(hr[lo:hi], weights=w + 1e-6)

    return smoothed

# ─────────────────────────────────────────────────────────────────────────────
# PHASE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def estimate_hr(ppg, imu):
    n_win   = len(ppg) // WIN_N
    acc_mag = acc_magnitude(imu)

    hr_vals   = []
    conf_vals = []

    for w in range(n_win):
        s, e    = w * WIN_N, (w + 1) * WIN_N
        ppg_win = ppg[s:e].astype(float)
        acc_win = acc_mag[s:e]
        hr, conf = estimate_window(ppg_win, acc_win)
        hr_vals.append(hr)
        conf_vals.append(conf)

    hr_corrected = continuity_correction(hr_vals)
    hr_smoothed  = smooth_hr_weighted(hr_corrected.tolist(), conf_vals, half_window=1)

    return list(hr_smoothed), conf_vals

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def competition_score(pred, gt):
    errs  = np.abs(np.array(pred) - np.array(gt))
    mae   = np.mean(errs)
    medae = np.median(errs)
    return (mae + medae) / 2.0, mae, medae


def evaluate(data):
    ppg_all = data['ppg']
    imu_all = data['imu']
    gt_all  = data.get('hr', None)

    if gt_all is None:
        print("No ground truth key 'hr' found. Keys:", list(data.keys()))
        return None

    print("\n" + "="*76)
    print(f"{'Phase':>6}  {'#Win':>5}  {'MAE':>8}  {'MedAE':>8}  {'Score':>8}  {'WorstWin':>12}")
    print("="*76)

    all_pred      = []
    all_gt        = []
    phase_results = []

    for phase_idx in range(len(ppg_all)):
        gt_phase = gt_all[phase_idx]
        if gt_phase is None or len(gt_phase) == 0:
            continue

        pred_list, conf_list = estimate_hr(ppg_all[phase_idx], imu_all[phase_idx])
        pred = np.array(pred_list)
        gt   = np.array(gt_phase)
        n    = min(len(pred), len(gt))
        pred, gt = pred[:n], gt[:n]

        errs          = np.abs(pred - gt)
        s, mae, medae = competition_score(pred, gt)
        worst_win     = int(np.argmax(errs))

        print(f"{phase_idx:>6}  {n:>5}  {mae:>8.2f}  {medae:>8.2f}  {s:>8.2f}"
              f"  win {worst_win:>3} ({errs[worst_win]:.1f} bpm)")

        all_pred.extend(pred.tolist())
        all_gt.extend(gt.tolist())
        phase_results.append({
            'phase': phase_idx, 'n_windows': n,
            'mae': mae, 'medae': medae, 'score': s,
            'worst_window': worst_win, 'worst_error': errs[worst_win],
            'pred': pred, 'gt': gt, 'conf': conf_list[:n], 'errors': errs,
        })

    all_pred = np.array(all_pred)
    all_gt   = np.array(all_gt)
    final_s, final_mae, final_medae = competition_score(all_pred, all_gt)

    print("="*76)
    print(f"\n{'OVERALL':>6}  {len(all_pred):>5}  {final_mae:>8.2f}  "
          f"{final_medae:>8.2f}  {final_s:>8.2f}")
    print(f"\n  ✦  FINAL SCORE  →  {final_s:.4f}   (MAE {final_mae:.2f} | MedAE {final_medae:.2f})")
    print("="*76 + "\n")

    return phase_results, all_pred, all_gt

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_phase_summary(phase_results):
    phases = [r['phase'] for r in phase_results]
    scores = [r['score'] for r in phase_results]
    maes   = [r['mae']   for r in phase_results]
    medaes = [r['medae'] for r in phase_results]
    colors = ['#d62728' if s > 10 else '#ff7f0e' if s > 5 else '#2ca02c' for s in scores]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle('Per-Phase Evaluation', fontsize=14, fontweight='bold')

    axes[0].bar(phases, scores, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].axhline(5,  color='green', linestyle='--', linewidth=1, label='Full points (≤5)')
    axes[0].axhline(15, color='red',   linestyle='--', linewidth=1, label='Zero points (≥15)')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Competition Score per Phase')
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, max(max(scores) * 1.15, 17))

    x = np.arange(len(phases))
    w = 0.4
    axes[1].bar(x - w/2, maes,   width=w, label='MAE',   color='#1f77b4', alpha=0.85)
    axes[1].bar(x + w/2, medaes, width=w, label='MedAE', color='#aec7e8', alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(phases)
    axes[1].set_xlabel('Phase Index')
    axes[1].set_ylabel('BPM Error')
    axes[1].set_title('MAE and MedAE per Phase')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('phase_summary.png', dpi=150, bbox_inches='tight')
    print("Saved: phase_summary.png")


def plot_worst_phases(phase_results, top_n=6):
    sorted_phases = sorted(phase_results, key=lambda r: r['score'], reverse=True)[:top_n]
    fig, axes = plt.subplots(top_n, 1, figsize=(14, 4 * top_n))
    if top_n == 1:
        axes = [axes]
    fig.suptitle(f'Top {top_n} Hardest Phases — Pred vs GT', fontsize=13, fontweight='bold')

    for ax, r in zip(axes, sorted_phases):
        t = np.arange(len(r['gt'])) * 30
        ax.plot(t, r['gt'],   label='Ground Truth', color='#2ca02c', linewidth=2)
        ax.plot(t, r['pred'], label='Predicted',    color='#d62728', linewidth=1.5, alpha=0.85)
        for i, (ti, err) in enumerate(zip(t, r['errors'])):
            if err > 10:
                ax.axvspan(ti, ti + 30, alpha=0.12, color='red')
        ax.set_title(f"Phase {r['phase']}  |  Score {r['score']:.2f}  |  "
                     f"MAE {r['mae']:.2f}  |  MedAE {r['medae']:.2f}")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('HR (bpm)')
        ax.legend(fontsize=9)
        ax.set_ylim(HR_MIN_BPM - 5, HR_MAX_BPM + 5)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('worst_phases.png', dpi=150, bbox_inches='tight')
    print("Saved: worst_phases.png")


def plot_error_distribution(all_pred, all_gt):
    errs = np.abs(all_pred - all_gt)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(errs, bins=50, color='#1f77b4', edgecolor='white', linewidth=0.4)
    ax.axvline(np.mean(errs),   color='red',    linestyle='--', linewidth=1.5,
               label=f'MAE   = {np.mean(errs):.2f}')
    ax.axvline(np.median(errs), color='orange', linestyle='--', linewidth=1.5,
               label=f'MedAE = {np.median(errs):.2f}')
    ax.set_xlabel('Absolute Error (bpm)')
    ax.set_ylabel('Number of Windows')
    ax.set_title('Distribution of Per-Window Absolute Errors')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=150, bbox_inches='tight')
    print("Saved: error_distribution.png")


def plot_per_window_errors(phase_results):
    all_errs = []
    all_gt   = []
    all_conf = []
    for r in phase_results:
        all_errs.extend(r['errors'].tolist())
        all_gt.extend(r['gt'].tolist())
        all_conf.extend(r['conf'])

    all_errs = np.array(all_errs)
    all_gt   = np.array(all_gt)
    all_conf = np.array(all_conf)
    all_conf_norm = all_conf / (all_conf.max() + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(all_gt, all_errs, c=all_conf_norm, cmap='RdYlGn',
                    alpha=0.55, s=18, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Spectral Confidence (normalised)')
    ax.axhline(10, color='red',     linestyle='--', linewidth=1, label='10 bpm error')
    ax.axhline(20, color='darkred', linestyle='--', linewidth=1, label='20 bpm error')
    ax.set_xlabel('Ground Truth HR (bpm)')
    ax.set_ylabel('Absolute Error (bpm)')
    ax.set_title('Per-Window Error vs True HR  (colour = spectral confidence)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('error_vs_hr.png', dpi=150, bbox_inches='tight')
    print("Saved: error_vs_hr.png")

# ─────────────────────────────────────────────────────────────────────────────
# SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────

def generate_submission(data, output_path='submission.csv'):
    ppg_all = data['ppg']
    imu_all = data['imu']
    rows    = []
    for phase_idx in range(len(ppg_all)):
        pred_list, _ = estimate_hr(ppg_all[phase_idx], imu_all[phase_idx])
        for win_idx, hr_val in enumerate(pred_list):
            rows.append({'Id': f'window{phase_idx}_{win_idx}', 'Predicted': float(hr_val)})
    df = pd.DataFrame(rows)[['Id', 'Predicted']]
    df.to_csv(output_path, index=False)
    print(f"\nSubmission saved → {output_path}  ({len(df)} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def debug_plot_spectra(data, phase_idx, n_windows=4):
    """
    Plot the filtered PPG power spectrum for the first n_windows of a phase.
    Also marks GT frequency (green), raw loudest peak (red), and
    best_fundamental result (blue). Saves to spectrum_phaseN.png.

    This tells us whether the TRUE HR frequency has any power at all,
    or whether the spectrum is just noise with no peak near GT.
    """
    ppg     = data['ppg'][phase_idx]
    imu     = data['imu'][phase_idx]
    gt      = data['hr'][phase_idx]
    acc_mag = acc_magnitude(imu)

    fig, axes = plt.subplots(n_windows, 1, figsize=(12, 3 * n_windows))
    if n_windows == 1:
        axes = [axes]
    fig.suptitle(f'Power Spectra — Phase {phase_idx}', fontsize=13, fontweight='bold')

    for w in range(min(n_windows, len(gt))):
        s, e     = w * WIN_N, (w + 1) * WIN_N
        ppg_win  = ppg[s:e].astype(float)
        acc_win  = acc_mag[s:e]
        ppg_filt = bandpass_filter(ppg_win)

        freqs, pwr  = power_spectrum(ppg_filt)
        pwr_norm    = pwr / (pwr.max() + 1e-10)

        raw_peak_hz = freqs[np.argmax(pwr)]
        best_hz, _  = best_fundamental(freqs, pwr_norm)
        gt_hz       = gt[w] / 60.0

        ax = axes[w]
        ax.plot(freqs * 60, pwr_norm, color='#1f77b4', linewidth=1.2)
        ax.axvline(gt_hz    * 60, color='green',  linewidth=2,   label=f'GT = {gt[w]:.1f} bpm')
        ax.axvline(raw_peak_hz*60,color='red',    linewidth=1.5, linestyle='--',
                   label=f'raw peak = {raw_peak_hz*60:.1f} bpm')
        ax.axvline(best_hz  * 60, color='orange', linewidth=1.5, linestyle=':',
                   label=f'corrected = {best_hz*60:.1f} bpm')
        ax.set_xlim(HR_MIN_BPM, HR_MAX_BPM)
        ax.set_ylabel('Power (norm)')
        ax.set_title(f'Window {w}')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Frequency (bpm)')
    plt.tight_layout()
    fname = f'spectrum_phase{phase_idx}.png'
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def debug_bad_phase(data, phase_idx, n_windows=8):
    """
    For each window, prints:
      GT         — ground truth HR
      raw_peak   — loudest spectral bin before any harmonic correction
      corrected  — what best_fundamental() picks after harmonic scoring
      peak_hr    — peak-detection estimate (or None)
      motion     — accelerometer std (proxy for movement intensity)
      peak_conf  — IBI regularity confidence from peak detector
    """
    ppg     = data['ppg'][phase_idx]
    imu     = data['imu'][phase_idx]
    gt      = data['hr'][phase_idx]
    acc_mag = acc_magnitude(imu)

    print(f"\n{'='*80}")
    print(f"DEBUG  Phase {phase_idx}  —  first {n_windows} windows")
    print(f"{'='*80}")
    print(f"  {'win':>3}  {'GT':>6}  {'raw_peak':>9}  {'corrected':>10}  "
          f"{'peak_hr':>8}  {'motion':>7}  {'pk_conf':>7}")
    print(f"  {'-'*3}  {'-'*6}  {'-'*9}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*7}")

    for w in range(min(n_windows, len(gt))):
        s, e     = w * WIN_N, (w + 1) * WIN_N
        ppg_win  = ppg[s:e].astype(float)
        acc_win  = acc_mag[s:e]
        ppg_filt = bandpass_filter(ppg_win)

        freqs, pwr       = power_spectrum(ppg_filt)
        raw_peak_hz      = freqs[np.argmax(pwr)]
        best_hz, _       = best_fundamental(freqs, pwr / (pwr.max() + 1e-10))
        hr_peak, pk_conf = hr_from_peaks(ppg_filt)
        motion           = motion_level(acc_win)

        peak_str = f"{hr_peak:>6.1f}" if hr_peak is not None else "  None"

        print(f"  {w:>3}  {gt[w]:>6.1f}  {raw_peak_hz*60:>9.1f}  "
              f"{best_hz*60:>10.1f}  {peak_str:>8}  "
              f"{motion:>7.1f}  {pk_conf:>7.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def explore_data_structure(data):
    """Print everything about the data structure so we know all available signals."""
    print("\n" + "="*60)
    print("DATA STRUCTURE")
    print("="*60)
    print(f"Top-level keys: {list(data.keys())}")
    for k, v in data.items():
        if hasattr(v, '__len__'):
            print(f"\n  '{k}': len={len(v)}")
            # Show shape/type of first element
            if len(v) > 0 and v[0] is not None:
                el = v[0]
                if hasattr(el, 'shape'):
                    print(f"    element[0].shape = {el.shape}, dtype = {el.dtype}")
                elif hasattr(el, '__len__'):
                    print(f"    element[0]: len={len(el)}, type={type(el[0]) if len(el)>0 else 'empty'}")
                    # If it's a list/array of arrays (multi-channel)
                    if len(el) > 0 and hasattr(el[0], 'shape'):
                        for i, ch in enumerate(el):
                            print(f"      channel[{i}]: shape={ch.shape}")
                else:
                    print(f"    element[0]: {el}")
    print("="*60 + "\n")


def debug_plot_dual_channel(data, phase_idx, n_windows=4):
    """
    If there are multiple PPG channels, plot spectra of EACH channel
    side by side for the worst windows. This shows whether the other
    channel has a cleaner signal near the true HR.
    """
    ppg_data = data['ppg']
    gt       = data['hr'][phase_idx]

    # Detect if ppg is multi-channel (list of arrays per phase)
    phase_ppg = ppg_data[phase_idx]

    # Check shape — could be (n_samples,) or (n_channels, n_samples) or list
    if hasattr(phase_ppg, 'shape') and phase_ppg.ndim == 2:
        n_channels = phase_ppg.shape[0]
        channels   = [phase_ppg[i] for i in range(n_channels)]
        print(f"Phase {phase_idx}: found {n_channels} PPG channels, shape {phase_ppg.shape}")
    elif isinstance(phase_ppg, (list, np.ndarray)) and hasattr(phase_ppg[0], '__len__'):
        channels = [np.array(ch) for ch in phase_ppg]
        n_channels = len(channels)
        print(f"Phase {phase_idx}: found {n_channels} PPG channels (list format)")
    else:
        print(f"Phase {phase_idx}: single PPG channel — no multi-channel data available")
        return

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    fig, axes = plt.subplots(n_windows, 1, figsize=(13, 3.5 * n_windows))
    if n_windows == 1:
        axes = [axes]
    fig.suptitle(f'Multi-Channel Spectra — Phase {phase_idx}', fontsize=13, fontweight='bold')

    for w in range(min(n_windows, len(gt))):
        s, e = w * WIN_N, (w + 1) * WIN_N
        ax   = axes[w]
        ax.axvline(gt[w], color='green', linewidth=2, label=f'GT={gt[w]:.1f}')

        for c_idx, ch in enumerate(channels):
            ch_win  = ch[s:e].astype(float)
            ch_filt = bandpass_filter(ch_win)
            freqs, pwr = power_spectrum(ch_filt)
            pwr_norm   = pwr / (pwr.max() + 1e-10)
            raw_peak   = freqs[np.argmax(pwr)] * 60
            ax.plot(freqs * 60, pwr_norm, color=colors[c_idx % len(colors)],
                    linewidth=1.2, alpha=0.8, label=f'ch{c_idx} peak={raw_peak:.1f}')

        ax.set_xlim(HR_MIN_BPM, HR_MAX_BPM)
        ax.set_ylabel('Power (norm)')
        ax.set_title(f'Window {w}  |  GT={gt[w]:.1f} bpm')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Frequency (bpm)')
    plt.tight_layout()
    fname = f'multichannel_phase{phase_idx}.png'
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


if __name__ == '__main__':
    print(f"Loading data...\n  {DATA_PATH}\n")
    data = np.load(DATA_PATH, allow_pickle=True).item()

    # ── Step 1: explore what signals are available ────────────────────────────
    explore_data_structure(data)

    # ── Step 2: check if multi-channel PPG helps on bad phases ───────────────
    for bad_phase in [1, 9]:
        debug_plot_dual_channel(data, phase_idx=bad_phase, n_windows=4)

    # ── Step 3: full evaluation ───────────────────────────────────────────────
    results = evaluate(data)

    if results is not None:
        phase_results, all_pred, all_gt = results
        plot_phase_summary(phase_results)
        plot_worst_phases(phase_results, top_n=6)
        plot_error_distribution(all_pred, all_gt)
        plot_per_window_errors(phase_results)

    generate_submission(data, output_path='submission.csv')