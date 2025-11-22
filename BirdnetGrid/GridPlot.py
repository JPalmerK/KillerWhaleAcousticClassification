# -*- coding: utf-8 -*-
"""
TKW Call-Type Spectrogram Panel
No colorbar, no title, with T07/T08 brightened.
kHz on y-axis.
"""

import os
from pathlib import Path
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# ---------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------
data_dir = Path(r"C:\TempData\TKWCalls\birdnet02\TKW")

n_calltypes_to_plot = 8       # 8 panels
segment_duration_s = 3.0      # each clip is 3 s

n_fft = 1024
hop_length = n_fft //10       # % overlap

fmax_hz = 6000                # spectrogram cropped to 6 kHz
dynamic_range_db = 65         # display top 60 dB

# ---------------------------------------------------------------------
# 2. Collect one file per TKW call type (using first 3 chars)
# ---------------------------------------------------------------------
wav_files = sorted(data_dir.glob("*.wav"))

if len(wav_files) == 0:
    raise FileNotFoundError(f"No .wav files found in {data_dir}")

selected_files = OrderedDict()
for f in wav_files:
    call_type = f.stem[:3]
    if call_type not in selected_files:
        selected_files[call_type] = f
    if len(selected_files) >= n_calltypes_to_plot:
        break

# ---------------------------------------------------------------------
# 3. Load, normalize, STFT, track global max magnitude
# ---------------------------------------------------------------------
spec_magnitudes = []
sr_global = None
global_max_mag = 0.0

for call_type, path in selected_files.items():
    y, sr = librosa.load(path, sr=None, mono=True)

    if sr_global is None:
        sr_global = sr
    elif sr != sr_global:
        y = librosa.resample(y, orig_sr=sr, target_sr=sr_global)
        sr = sr_global

    target_len = int(segment_duration_s * sr)
    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))

    max_abs = np.max(np.abs(y))
    if max_abs > 0:
        y = y / max_abs

    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     window="hann", center=True)
    S_abs = np.abs(S)

    if S_abs.max() > global_max_mag:
        global_max_mag = S_abs.max()

    spec_magnitudes.append((call_type, S_abs))

# ---------------------------------------------------------------------
# 4. Convert to dB with shared reference + crop to 0–6 kHz
# ---------------------------------------------------------------------
if global_max_mag <= 0:
    raise RuntimeError("Global max magnitude <= 0, check audio files.")

freqs = librosa.fft_frequencies(sr=sr_global, n_fft=n_fft)
max_bin = np.where(freqs <= fmax_hz)[0].max()

spec_db_list = []
for call_type, S_abs in spec_magnitudes:
    S_db = librosa.amplitude_to_db(S_abs, ref=global_max_mag)
    S_db_crop = S_db[:max_bin + 1, :]
    spec_db_list.append((call_type, S_db_crop))

# ---------------------------------------------------------------------
# 5. Plot: 4×2 grid, no colorbar/title, brighten T07/T08, y-axis in kHz, grayscale
# ---------------------------------------------------------------------
n_rows, n_cols = 4, 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 10),
                         sharex=True, sharey=True)
axes = axes.ravel()

for i, (call_type, S_db_crop) in enumerate(spec_db_list):
    ax = axes[i]

    # brighten T07 & T08
    if call_type in ("T07", "T08"):
        S_db_plot = S_db_crop + 6
    else:
        S_db_plot = S_db_crop

    # ---- Grayscale spectrogram ----
    librosa.display.specshow(
        S_db_plot,
        sr=sr_global,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        vmin=-dynamic_range_db,
        vmax=0,
        cmap="gray",    # <-- HERE: grayscale
        ax=ax,
    )

    ax.set_ylim([0, fmax_hz])

    # Convert Hz ticks to kHz
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{yt/1000:.1f}" for yt in yticks])

    # Left column labels
    if i % 2 == 0:
        ax.set_ylabel("kHz")
    else:
        ax.set_ylabel("")

    # X labels everywhere, bottom row gets the axis label
    if i // 2 == (n_rows - 1):
        ax.set_xlabel("Time (s)")
    else:
        ax.set_xlabel("")

    ax.set_title(call_type, fontsize=10)

fig.tight_layout(rect=[0, 0, 1, 1])

plt.savefig("TKW_calltype_spectrogram_panel_bw.pdf",
            dpi=300, bbox_inches="tight")
plt.show()

