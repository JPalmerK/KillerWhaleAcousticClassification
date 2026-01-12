# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 12:34:25 2026

@author: kaity
"""

import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Literal, Iterable, Tuple

import numpy as np
import pandas as pd
import librosa
import scipy.special
import soundfile as sf
import tensorflow.lite as tflite

ChannelSpec = Union[int, List[int], Literal["mix", "all"]]
ThresholdSpec = Optional[Dict[Union[int, str], float]]


class BirdNetPredictorNew:
    """
    TFLite (BirdNET-style) predictor that can:
      - run sliding-window inference on a single audio file (safe streaming: no full-file load)
      - run inference across a folder (recursively) producing ONE global Raven timeline
      - export results to a Raven selection table

    Key Raven requirement supported:
      - Begin/End times are referenced to t=0 at the start of the FIRST file
        and increase continuously across subsequent files (global timeline).

    Performance / stability:
      - Uses soundfile streaming (SoundFile.read) to avoid loading entire files
      - Batches windows before calling the TFLite interpreter
      - Can append to Raven file continuously during processing

    Channel selection:
      channels =
        - "mix" (default): average all channels -> single timeline, Raven Channel=1
        - int (e.g., 0): pick one channel -> Raven Channel=1
        - list (e.g., [0,1]): average those channels -> Raven Channel=1
        - "all": run each channel separately -> Raven Channel = actual channel index + 1

    Per-class thresholds:
      - confidence_thresh is the default threshold for all classes
      - class_thresholds can override per class by:
          * class index (int), e.g. {5: 0.9}
          * class label (str), matching exactly labels read from label_path, e.g. {"SRKW": 0.9}
        Resolution order: index override > label override > default confidence_thresh
    """

    def __init__(
        self,
        model_path: str,
        label_path: str,
        audio_folder: Optional[Union[str, Path]] = None,
        sample_rate: int = 48000,
        audio_duration: float = 3.0,
        confidence_thresh: float = 0.5,
        class_thresholds: ThresholdSpec = None,
        recursive: bool = True,
        validate_threshold_labels: bool = True,
    ):
        self.model_path = str(model_path)
        self.label_path = str(label_path)
        self.audio_folder = str(audio_folder) if audio_folder is not None else None
        self.sample_rate = int(sample_rate)
        self.audio_duration = float(audio_duration)
        self.confidence_thresh = float(confidence_thresh)
        self.recursive = bool(recursive)

        # Load labels from label_path (required for label-based thresholds + outputs)
        self.labels = self.load_labels(self.label_path)

        # Store threshold overrides (by index or label string)
        self.class_thresholds: Dict[Union[int, str], float] = dict(class_thresholds or {})

        # Optional: validate any string keys in class_thresholds against loaded labels
        if validate_threshold_labels and self.class_thresholds:
            for k in self.class_thresholds:
                if isinstance(k, str) and k not in self.labels:
                    raise ValueError(
                        f"class_thresholds key '{k}' not found in labels from: {self.label_path}\n"
                        f"Available labels: {self.labels}"
                    )

        # Load TFLite model once
        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Infer duration from model input when possible:
        input_shape = self.input_details[0]["shape"]
        if len(input_shape) >= 2 and int(input_shape[1]) > 0:
            # samples / sample_rate
            self.audio_duration = float(input_shape[1]) / float(self.sample_rate)

        print(f"Model expects ~{self.audio_duration:.6f} seconds of audio at {self.sample_rate} Hz")

    # ----------------------------
    # IO helpers
    # ----------------------------
    @staticmethod
    def load_labels(label_path: str) -> List[str]:
        with open(label_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    @staticmethod
    def _is_audio_file(p: Path) -> bool:
        return p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}

    def _list_audio_files(
        self,
        audio_folder: Optional[Union[str, Path]] = None,
        recursive: Optional[bool] = None,
    ) -> List[str]:
        folder = Path(audio_folder) if audio_folder is not None else Path(self.audio_folder)  # type: ignore[arg-type]
        if recursive is None:
            recursive = self.recursive

        if not folder.exists():
            raise FileNotFoundError(str(folder))
        if not folder.is_dir():
            raise NotADirectoryError(str(folder))

        it = folder.rglob("*") if recursive else folder.glob("*")
        files = sorted([str(p) for p in it if p.is_file() and self._is_audio_file(p)])
        return files

    @staticmethod
    def _file_duration_seconds(audio_path: str) -> float:
        """Prefer soundfile for exact frame count + samplerate without decoding all samples."""
        with sf.SoundFile(audio_path) as f:
            return float(len(f)) / float(f.samplerate)

    # ----------------------------
    # Threshold resolution
    # ----------------------------
    def _threshold_for_class(self, class_idx: int) -> float:
        """
        Resolve threshold for a class index using:
          1) int index override in self.class_thresholds
          2) label string override in self.class_thresholds (labels read from label_path)
          3) default self.confidence_thresh
        """
        if class_idx in self.class_thresholds:
            return float(self.class_thresholds[class_idx])

        if 0 <= class_idx < len(self.labels):
            lab = self.labels[class_idx]
            if lab in self.class_thresholds:
                return float(self.class_thresholds[lab])

        return float(self.confidence_thresh)

    # ----------------------------
    # Audio preprocessing + model
    # ----------------------------
    def preprocess_audio(
        self,
        audio: np.ndarray,
        sr: int,
        target_sr: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> np.ndarray:
        """
        Resample if needed, then pad/trim to exactly `duration` seconds at `target_sr`.
        Returns array shaped (1, samples) float32.
        """
        if target_sr is None:
            target_sr = self.sample_rate
        if duration is None:
            duration = self.audio_duration

        # Ensure mono 1D if user passed 2D
        if audio.ndim > 1:
            audio = audio[:, 0]

        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        required_length = int(round(target_sr * duration))

        if len(audio) < required_length:
            audio = np.pad(audio, (0, required_length - len(audio)), mode="constant")
        else:
            audio = audio[:required_length]

        return np.expand_dims(audio.astype(np.float32), axis=0)

    def predict_batch(self, audio_batch: np.ndarray) -> np.ndarray:
        """
        Run inference on a batch of preprocessed segments.
        Expects shape (batch_size, samples) float32.
        Returns model outputs shaped (batch_size, n_classes) or similar.
        """
        in_idx = self.input_details[0]["index"]

        # Resize input tensor to batch shape (convenient, sometimes slower)
        self.interpreter.resize_tensor_input(in_idx, audio_batch.shape)
        self.interpreter.allocate_tensors()

        self.interpreter.set_tensor(in_idx, audio_batch.astype(np.float32))
        self.interpreter.invoke()

        out_idx = self.output_details[0]["index"]
        return self.interpreter.get_tensor(out_idx)

    # ----------------------------
    # Channel mixing / selection
    # ----------------------------
    @staticmethod
    def _mix_channels(x2d: np.ndarray, channels: ChannelSpec) -> Union[np.ndarray, List[np.ndarray]]:
        """
        x2d: shape (n_samples, n_channels) float32 (or 1D)
        Returns:
          - 1D mono array if channels != "all"
          - list of 1D arrays if channels == "all"
        """
        if x2d.ndim == 1:
            x2d = x2d[:, None]

        n_ch = x2d.shape[1]

        if channels == "all":
            return [x2d[:, c].copy() for c in range(n_ch)]

        if channels == "mix":
            return np.mean(x2d, axis=1)

        if isinstance(channels, int):
            if channels < 0 or channels >= n_ch:
                raise ValueError(f"channel index {channels} out of range [0, {n_ch - 1}]")
            return x2d[:, channels]

        # list of channels
        chs = list(channels)
        for c in chs:
            if c < 0 or c >= n_ch:
                raise ValueError(f"channel index {c} out of range [0, {n_ch - 1}]")
        return np.mean(x2d[:, chs], axis=1)

    # ----------------------------
    # Streaming window iterator (NO full-file load)
    # ----------------------------
    def _iter_windows_streaming(
        self,
        audio_path: Union[str, Path],
        hop_s: float,
        channels: ChannelSpec = "mix",
        dtype: str = "float32",
    ) -> Iterable[Tuple[np.ndarray, float, float, Optional[int], int]]:
        """
        Yields tuples:
          (window_audio_1d, window_start_sec, window_end_sec, channel_index_or_None, file_sr)
        without ever loading the full file.
        """
        audio_path = str(audio_path)

        with sf.SoundFile(audio_path) as f:
            sr = int(f.samplerate)
            seg_len = int(round(sr * self.audio_duration))
            hop_len = int(round(sr * hop_s))
            if seg_len <= 0 or hop_len <= 0:
                return

            # rolling buffer of shape (n_samples_in_buf, n_channels)
            buf = np.empty((0, int(f.channels)), dtype=np.float32)
            buf_start_sample = 0
            next_start_sample = 0

            while True:
                need_until = next_start_sample + seg_len
                have_until = buf_start_sample + buf.shape[0]
                to_read = max(0, need_until - have_until)

                if to_read > 0:
                    chunk = f.read(frames=to_read, dtype=dtype, always_2d=True)
                    if chunk.size == 0:
                        break
                    buf = np.vstack([buf, chunk])

                have_until = buf_start_sample + buf.shape[0]
                if need_until > have_until:
                    break  # no more full windows

                start_in_buf = next_start_sample - buf_start_sample
                end_in_buf = start_in_buf + seg_len
                win2d = buf[start_in_buf:end_in_buf, :]  # (seg_len, channels)

                mixed = self._mix_channels(win2d, channels)

                start_t = next_start_sample / sr
                end_t = (next_start_sample + seg_len) / sr

                if channels == "all":
                    for ch_idx, w in enumerate(mixed):  # list of 1D
                        yield w, start_t, end_t, ch_idx, sr
                else:
                    yield mixed, start_t, end_t, None, sr

                next_start_sample += hop_len

                # drop old buffer samples to keep memory bounded
                drop_until = max(0, next_start_sample - seg_len)
                if drop_until > buf_start_sample:
                    drop_n = drop_until - buf_start_sample
                    buf = buf[drop_n:, :]
                    buf_start_sample = drop_until

    # ----------------------------
    # Sliding-window inference (streaming) for one file
    # ----------------------------
    def predict_long_audio_windows(
        self,
        audio_path: Union[str, Path],
        hop_s: Optional[float] = None,
        channels: ChannelSpec = "mix",
        include_all_class_scores: bool = True,
        use_sigmoid: bool = True,
        batch_size: int = 16,
    ) -> pd.DataFrame:
        """
        Sliding-window inference over ONE audio file (streaming).
        Returns DataFrame with times RELATIVE to file start.

        Note: if channels="all", results include a 'Raven Channel' column (1-based).
              otherwise Raven Channel is always 1.
        """
        audio_path = str(audio_path)
        hop_s = float(hop_s) if hop_s is not None else float(self.audio_duration)

        rows: List[Dict] = []
        pending_waves: List[np.ndarray] = []
        pending_meta: List[Tuple[float, float, Optional[int], int]] = []  # (begin_s, end_s, ch_idx, sr)

        def flush():
            nonlocal pending_waves, pending_meta, rows

            if not pending_waves:
                return

            X = np.stack(pending_waves, axis=0).astype(np.float32, copy=False)  # (B, samples)
            preds = self.predict_batch(X)

            for i, pred in enumerate(preds):
                scores = scipy.special.expit(pred) if use_sigmoid else pred
                top_idx = int(np.argmax(scores))
                top_score = float(scores[top_idx])

                thr = self._threshold_for_class(top_idx)
                if top_score < thr:
                    continue

                begin_s, end_s, ch_idx, _sr0 = pending_meta[i]
                rav_ch = (int(ch_idx) + 1) if ch_idx is not None else 1

                row = {
                    "Begin Time (S)": round(float(begin_s), 6),
                    "End Time (S)": round(float(end_s), 6),
                    "File": os.path.basename(audio_path),
                    "FilePath": audio_path,
                    "Predicted Class": self.labels[top_idx],
                    "Top Score": round(top_score, 6),
                    "Raven Channel": rav_ch,
                }

                if include_all_class_scores:
                    for class_idx, label in enumerate(self.labels):
                        row[label] = round(float(scores[class_idx]), 6)

                rows.append(row)

            pending_waves.clear()
            pending_meta.clear()
            del X, preds

        # stream windows
        for w, start_t, end_t, ch_idx, sr in self._iter_windows_streaming(
            audio_path, hop_s=hop_s, channels=channels
        ):
            x = self.preprocess_audio(w, sr=sr, target_sr=self.sample_rate, duration=self.audio_duration)[0]
            pending_waves.append(x)
            pending_meta.append((start_t, end_t, ch_idx, sr))

            if len(pending_waves) >= int(batch_size):
                flush()

        flush()
        return pd.DataFrame(rows)

    # ----------------------------
    # Folder inference with global Raven timeline (returns df)
    # ----------------------------
    def predict_folder_global_raven(
        self,
        audio_folder: Optional[Union[str, Path]] = None,
        hop_s: Optional[float] = None,
        channels: ChannelSpec = "mix",
        recursive: Optional[bool] = None,
        include_all_class_scores: bool = True,
        use_sigmoid: bool = True,
        batch_size: int = 16,
        sort_by_begin_time: bool = True,
    ) -> pd.DataFrame:
        """
        Process all audio files in a folder and return ONE DataFrame whose
        Begin/End times are GLOBAL:
          - t=0 at the beginning of the first file
          - times increase continuously across subsequent files
            by adding each file's duration as an offset.
        """
        files = self._list_audio_files(audio_folder=audio_folder, recursive=recursive)
        if not files:
            return pd.DataFrame()

        hop_s = float(hop_s) if hop_s is not None else float(self.audio_duration)

        global_offset = 0.0
        all_dfs: List[pd.DataFrame] = []

        for k, ap in enumerate(files, start=1):
            print(f"[{k}/{len(files)}] {ap}")

            df = self.predict_long_audio_windows(
                ap,
                hop_s=hop_s,
                channels=channels,
                include_all_class_scores=include_all_class_scores,
                use_sigmoid=use_sigmoid,
                batch_size=batch_size,
            )

            if not df.empty:
                df["Begin Time (S)"] = df["Begin Time (S)"] + global_offset
                df["End Time (S)"] = df["End Time (S)"] + global_offset
                all_dfs.append(df)

            global_offset += self._file_duration_seconds(ap)

        if not all_dfs:
            return pd.DataFrame()

        out = pd.concat(all_dfs, ignore_index=True)

        if sort_by_begin_time and "Begin Time (S)" in out.columns:
            out = out.sort_values("Begin Time (S)").reset_index(drop=True)

        return out

    # ----------------------------
    # Raven export (df -> selection table)
    # ----------------------------
    def export_to_raven(
        self,
        df: pd.DataFrame,
        raven_file: str = "raven_output.txt",
        low_hz: int = 0,
        high_hz: int = 24000,
        view: str = "Spectrogram 1",
        default_channel: int = 1,
        class_col: str = "Predicted Class",
        score_col: str = "Top Score",
        include_file_cols: bool = True,
    ) -> None:
        """
        Export a DataFrame to a Raven selection table (tab-delimited).

        Required Raven fields:
          Selection, View, Channel, Begin Time (S), End Time (S)

        Also writes Low/High freq + class + score.

        If df has column 'Raven Channel', that will be used for Channel,
        otherwise default_channel is used.
        """
        cols = [
            "Selection", "View", "Channel",
            "Begin Time (S)", "End Time (S)",
            "Low Freq (Hz)", "High Freq (Hz)",
            "Class", "Score",
        ]
        if include_file_cols:
            cols += ["File", "FilePath"]

        if df is None or df.empty:
            with open(raven_file, "w", encoding="utf-8") as f:
                f.write("\t".join(cols) + "\n")
            print(f"Raven selection table exported (empty) to {raven_file}")
            return

        out = df.copy()

        if "Begin Time (S)" not in out.columns or "End Time (S)" not in out.columns:
            raise ValueError("DataFrame must include 'Begin Time (S)' and 'End Time (S)'.")

        if class_col not in out.columns:
            raise ValueError(f"Expected class column '{class_col}' not found in DataFrame.")
        if score_col not in out.columns:
            raise ValueError(f"Expected score column '{score_col}' not found in DataFrame.")

        out["Selection"] = range(1, len(out) + 1)
        out["View"] = view
        out["Low Freq (Hz)"] = int(low_hz)
        out["High Freq (Hz)"] = int(high_hz)
        out["Class"] = out[class_col]
        out["Score"] = out[score_col]

        if "Raven Channel" in out.columns:
            out["Channel"] = out["Raven Channel"].astype(int)
        else:
            out["Channel"] = int(default_channel)

        with open(raven_file, "w", encoding="utf-8") as f:
            f.write("\t".join(cols) + "\n")
            for _, row in out.iterrows():
                base = (
                    f"{int(row['Selection'])}\t{row['View']}\t{int(row['Channel'])}\t"
                    f"{float(row['Begin Time (S)']):.6f}\t{float(row['End Time (S)']):.6f}\t"
                    f"{int(row['Low Freq (Hz)'])}\t{int(row['High Freq (Hz)'])}\t"
                    f"{row['Class']}\t{float(row['Score']):.6f}"
                )
                if include_file_cols:
                    filev = row["File"] if "File" in out.columns else ""
                    pathv = row["FilePath"] if "FilePath" in out.columns else ""
                    f.write(base + f"\t{filev}\t{pathv}\n")
                else:
                    f.write(base + "\n")

        print(f"Raven selection table exported to {raven_file}")

    # ----------------------------
    # Streaming Raven output while processing folder (append as you go)
    # ----------------------------
    def _init_raven_file(self, raven_file: str, include_file_cols: bool = True) -> None:
        cols = [
            "Selection", "View", "Channel",
            "Begin Time (S)", "End Time (S)",
            "Low Freq (Hz)", "High Freq (Hz)",
            "Class", "Score",
        ]
        if include_file_cols:
            cols += ["File", "FilePath"]
        with open(raven_file, "w", encoding="utf-8") as f:
            f.write("\t".join(cols) + "\n")

    def predict_folder_global_raven_streaming_to_file(
        self,
        raven_file: str,
        audio_folder: Optional[Union[str, Path]] = None,
        hop_s: float = 1.5,
        channels: ChannelSpec = "mix",
        recursive: Optional[bool] = None,
        batch_size: int = 16,
        view: str = "Spectrogram 1",
        low_hz: int = 0,
        high_hz: int = 24000,
        use_sigmoid: bool = True,
        include_file_cols: bool = True,
    ) -> None:
        """
        Stream windows from each file, batch inference, and APPEND Raven rows as we go.
        Global timeline maintained (t=0 at start of first file).

        Per-class thresholds are applied based on the TOP predicted class for each window.
        """
        files = self._list_audio_files(audio_folder=audio_folder, recursive=recursive)
        self._init_raven_file(raven_file, include_file_cols=include_file_cols)

        if not files:
            print(f"No audio files found. Wrote empty Raven file: {raven_file}")
            return

        selection = 0
        global_offset = 0.0

        pending_waves: List[np.ndarray] = []
        pending_meta: List[Tuple[float, float, str, str, int]] = []  # begin_g, end_g, file, filepath, rav_ch

        is_all = (channels == "all")

        def flush():
            nonlocal selection, pending_waves, pending_meta

            if not pending_waves:
                return

            X = np.stack(pending_waves, axis=0).astype(np.float32, copy=False)  # (B, samples)
            preds = self.predict_batch(X)

            with open(raven_file, "a", encoding="utf-8") as f:
                for i, pred in enumerate(preds):
                    scores = scipy.special.expit(pred) if use_sigmoid else pred

                    top_idx = int(np.argmax(scores))
                    top_score = float(scores[top_idx])

                    thr = self._threshold_for_class(top_idx)
                    if top_score < thr:
                        continue

                    begin_g, end_g, fname, fpath, rav_ch = pending_meta[i]

                    selection += 1
                    rav_ch= rav_ch+1
                    base = (
                        f"{selection}\t{view}\t{rav_ch}\t"
                        f"{begin_g:.6f}\t{end_g:.6f}\t"
                        f"{low_hz}\t{high_hz}\t"
                        f"{self.labels[top_idx]}\t{top_score:.6f}"
                    )
                    if include_file_cols:
                        f.write(base + f"\t{fname}\t{fpath}\n")
                    else:
                        f.write(base + "\n")

            pending_waves.clear()
            pending_meta.clear()
            del X, preds

        for k, ap in enumerate(files, start=1):
            print(f"[{k}/{len(files)}] {ap}")

            file_duration = self._file_duration_seconds(ap)

            for w, start_t, end_t, ch_idx, sr in self._iter_windows_streaming(ap, hop_s=hop_s, channels=channels):
                x = self.preprocess_audio(w, sr=sr, target_sr=self.sample_rate, duration=self.audio_duration)[0]

                begin_g = global_offset + start_t
                end_g = global_offset + end_t

                # Raven channel mapping rules you specified:
                # - if channels is int or list -> mix/pick -> Raven Channel = 1
                # - if channels == "mix" -> Raven Channel = 1
                # - if channels == "all" -> preserve -> Raven Channel = (ch_idx+1)
                rav_ch = (int(ch_idx) + 1) if (is_all and ch_idx is not None) else 1

                pending_waves.append(x)
                pending_meta.append((begin_g, end_g, os.path.basename(ap), str(ap), rav_ch))

                if len(pending_waves) >= int(batch_size):
                    flush()

            flush()
            global_offset += float(file_duration)

        print(f"Done. Wrote streaming Raven table: {raven_file}")


if __name__ == "__main__":
    pred = BirdNetPredictorNew(
        model_path=r"C:\Users\kaity\Documents\GitHub\EcotypeFinal\BirdNET Models\birdnet07\birdnet07_8khz_cutoff.tflite",
        label_path=r"C:\Users\kaity\Documents\GitHub\EcotypeFinal\BirdNET Models\birdnet07\birdnet07_8khz_cutoff_Labels.txt",
        audio_folder=r"E:\AdriftData\Adrift_040",
        confidence_thresh=0.6,                 # default for all classes
        class_thresholds={"SRKW": 0.95, 'TKW': 0.90, 'HW': 0.9},        # stricter threshold just for SRKW
        recursive=True,
    )

    # Streaming Raven output (writes as it goes)
    pred.predict_folder_global_raven_streaming_to_file(
        raven_file=r"C:\TempData\Adrift_040_streaming.txt",
        hop_s=1.5,
        channels=1,           # int index of channel to pick (0-based). Raven Channel will be 1 (mixed/picked)
        batch_size=16,
        recursive=True,
        low_hz=0,
        high_hz=24000,
        include_file_cols=False,
        view="Spectrogram 1",
    )

    # Or: build a DataFrame first, then export at end
    # df = pred.predict_folder_global_raven(hop_s=1.5, channels="mix", recursive=True, batch_size=16)
    # pred.export_to_raven(df, r"C:\TempData\Adrift_040_end.txt")
