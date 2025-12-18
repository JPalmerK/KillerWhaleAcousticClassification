# -*- coding: utf-8 -*-
"""
TKW Call-Type Spectrogram Panel
No colorbar, no title, with T07/T08 brightened.
kHz on y-axis, 3x3 layout.

Assumptions:
- Exactly one example per call type, with the call type given by the first
  three characters of the filename (e.g., 'T01_...', 'T07-...').
- Audio files may be .wav, .aiff, or .aif.
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
data_dir = Path(r"C:\TempData\TKWCalls\birdnet02\TKW\GoodExamples")

n_calltypes_to_plot = 10       # 9 panels for 3x3
segment_duration_s = 3.0      # each clip is ~3 s

n_fft = 1024
hop_length = n_fft // 10      # ~90% overlap

fmax_hz = 8000                # spectrogram cropped to 6 kHz
dynamic_range_db = 65         # display top 65 dB

# ---------------------------------------------------------------------
# 2. Collect one file per call type (first 3 chars of stem)
# ---------------------------------------------------------------------
audio_files = []
audio_files.extend(data_dir.glob("*.wav"))
audio_files.extend(data_dir.glob("*.aiff"))
audio_files.extend(data_dir.glob("*.aif"))

audio_files = sorted(audio_files)

if len(audio_files) == 0:
    raise FileNotFoundError(f"No .wav/.aiff/.aif files found in {data_dir}")

# call_type (first 3 chars) -> first file seen
calltype_to_file = {}

for f in audio_files:
    stem = f.stem
    if len(stem) < 3:
        # Skip weird filenames that don't have 3 chars
        continue
    call_type = stem[:3]

    if call_type not in calltype_to_file:
        calltype_to_file[call_type] = f

# Sort call types so T01, T02, ... etc. are in order
sorted_items = sorted(calltype_to_file.items(), key=lambda kv: kv[0])

if len(sorted_items) < n_calltypes_to_plot:
    print(
        f"Warning: only found {len(sorted_items)} unique call types "
        f"from the first 3 characters. Plotting {len(sorted_items)} panels."
    )

sorted_items = sorted_items[:n_calltypes_to_plot]

# Optional: print what will be plotted
print("Call types and files used:")
for ct, fp in sorted_items:
    print(f"  {ct}: {fp.name}")

# ---------------------------------------------------------------------
# 3. Load, normalize, STFT, track global max magnitude
# ---------------------------------------------------------------------
spec_magnitudes = []  # list of (call_type, S_abs)
sr_global = None
global_max_mag = 0.0

for call_type, path in sorted_items:
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
# 5. Plot: 3×3 grid, no colorbar/title, brighten T07/T08, y-axis in kHz
# ---------------------------------------------------------------------
n_rows, n_cols = 4, 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 11),
                         sharex=True, sharey=True)
axes = axes.ravel()

for i, (call_type, S_db_crop) in enumerate(spec_db_list):
    ax = axes[i]

    # brighten T07 & T08
    if call_type in ("T07", "T08"):
        S_db_plot = S_db_crop + 6
    else:
        S_db_plot = S_db_crop

    librosa.display.specshow(
        S_db_plot + 11,   # your existing offset
        sr=sr_global,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        vmin=-dynamic_range_db,
        vmax=0,
        cmap="Greys",
        ax=ax,
    )

    ax.set_ylim([0, fmax_hz])

    # Convert Hz ticks to kHz
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{yt/1000:.1f}" for yt in yticks])

    # Left column gets y-label
    if i % n_cols == 0:
        ax.set_ylabel("kHz")
    else:
        ax.set_ylabel("")

    # Bottom row gets x-label
    if i // n_cols == (n_rows - 1):
        ax.set_xlabel("Time (s)")
    else:
        ax.set_xlabel("")

    ax.set_title(call_type, fontsize=10)

# Remove any unused axes if there are < 9 call types
for j in range(len(spec_db_list), len(axes)):
    fig.delaxes(axes[j])

fig.tight_layout(rect=[0, 0, 1, 1])

plt.savefig("TKW_calltype_spectrogram_panel_3x3_bw.pdf",
            dpi=300, bbox_inches="tight")
plt.show()
