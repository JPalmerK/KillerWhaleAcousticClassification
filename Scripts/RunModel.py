import os
from pathlib import Path
from typing import Iterable, Optional, Union, List, Dict

import numpy as np
import pandas as pd
import librosa
import scipy.special
import soundfile as sf
import tensorflow.lite as tflite


class BirdNetPredictorNew:
    """
    TFLite (BirdNET-style) predictor that can:
      - run sliding-window inference on a single audio file
      - run inference across a folder (recursively) producing ONE global Raven timeline
      - export results to a Raven selection table

    Key Raven requirement supported:
      - Begin/End times are referenced to t=0 at the start of the FIRST file
        and increase continuously across subsequent files (global timeline).
    """

    def __init__(
        self,
        model_path: str,
        label_path: str,
        audio_folder: Optional[Union[str, Path]] = None,
        sample_rate: int = 48000,
        audio_duration: float = 3.0,
        confidence_thresh: float = 0.5,
        recursive: bool = True,
    ):
        self.model_path = str(model_path)
        self.label_path = str(label_path)
        self.audio_folder = str(audio_folder) if audio_folder is not None else None
        self.sample_rate = int(sample_rate)
        self.audio_duration = float(audio_duration)
        self.confidence_thresh = float(confidence_thresh)
        self.recursive = bool(recursive)

        # Load TFLite model once
        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Use input shape to infer duration when possible:
        # expected input is (1, samples) or (batch, samples)
        input_shape = self.input_details[0]["shape"]
        # Some models store dynamic batch dim, but samples should be dim 1.
        # If it's not, we leave audio_duration as provided.
        if len(input_shape) >= 2 and input_shape[1] > 0:
            self.audio_duration = float(input_shape[1]) / float(self.sample_rate)

        print(f"Model expects ~{self.audio_duration:.6f} seconds of audio at {self.sample_rate} Hz")

        self.labels = self.load_labels(self.label_path)

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
        """
        Prefer soundfile for exact frame count + samplerate without decoding all samples.
        """
        with sf.SoundFile(audio_path) as f:
            return float(len(f)) / float(f.samplerate)

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

        # Ensure mono
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
        # Resize input tensor to batch shape (some models require this)
        in_idx = self.input_details[0]["index"]
        self.interpreter.resize_tensor_input(in_idx, audio_batch.shape)
        self.interpreter.allocate_tensors()

        self.interpreter.set_tensor(in_idx, audio_batch.astype(np.float32))
        self.interpreter.invoke()

        out_idx = self.output_details[0]["index"]
        predictions = self.interpreter.get_tensor(out_idx)
        return predictions

    # ----------------------------
    # Sliding-window inference
    # ----------------------------
    def predict_long_audio_windows(
        self,
        audio_path: Union[str, Path],
        hop_s: Optional[float] = None,
        include_all_class_scores: bool = True,
        use_sigmoid: bool = True,
    ) -> pd.DataFrame:
        """
        Sliding-window inference over ONE audio file.

        Returns:
            DataFrame with times RELATIVE to file start (seconds):
              - Begin Time (S), End Time (S)
              - Predicted Class, Top Score
              - optional per-class scores columns (label names)
              - File, FilePath
        """
        audio_path = str(audio_path)
        hop_s = float(hop_s) if hop_s is not None else float(self.audio_duration)

        y, sr = librosa.load(audio_path, sr=None, mono=True)

        seg_len = int(round(sr * self.audio_duration))
        hop_len = int(round(sr * hop_s))

        if seg_len <= 0 or hop_len <= 0 or len(y) == 0:
            return pd.DataFrame()

        processed_segments = []
        start_ends = []

        # Build windows; include last partial window by padding via preprocess_audio()
        # We'll step until start < len(y)
        for start in range(0, len(y), hop_len):
            end = min(start + seg_len, len(y))
            segment = y[start:end]
            processed = self.preprocess_audio(segment, sr=sr, target_sr=self.sample_rate, duration=self.audio_duration)
            processed_segments.append(processed)
            start_ends.append((start, end))

            if end >= len(y):
                break

        audio_batch = np.vstack(processed_segments)  # (n_windows, 1, samples)?? -> preprocess returns (1, samples)
        # Ensure shape (n_windows, samples)
        if audio_batch.ndim == 3 and audio_batch.shape[1] == 1:
            audio_batch = audio_batch[:, 0, :]

        predictions = self.predict_batch(audio_batch)

        rows = []
        for i, pred in enumerate(predictions):
            pred = np.asarray(pred)

            # Convert logits to scores (your current approach)
            scores = scipy.special.expit(pred) if use_sigmoid else pred

            top_idx = int(np.argmax(scores))
            top_score = float(scores[top_idx])

            if top_score < self.confidence_thresh:
                continue

            start_sample, end_sample = start_ends[i]
            row = {
                "Begin Time (S)": round(start_sample / sr, 6),
                "End Time (S)": round(end_sample / sr, 6),
                "File": os.path.basename(audio_path),
                "FilePath": audio_path,
                "Predicted Class": self.labels[top_idx],
                "Top Score": round(top_score, 6),
            }

            if include_all_class_scores:
                for class_idx, label in enumerate(self.labels):
                    row[label] = round(float(scores[class_idx]), 6)

            rows.append(row)

        return pd.DataFrame(rows)

    # ----------------------------
    # Folder inference with global Raven timeline
    # ----------------------------
    def predict_folder_global_raven(
        self,
        audio_folder: Optional[Union[str, Path]] = None,
        hop_s: Optional[float] = None,
        recursive: Optional[bool] = None,
        include_all_class_scores: bool = True,
        use_sigmoid: bool = True,
        sort_by_begin_time: bool = True,
    ) -> pd.DataFrame:
        """
        Process all audio files in a folder and return ONE DataFrame whose
        Begin/End times are GLOBAL:
          - t=0 at the beginning of the first file
          - times increase continuously across subsequent files
            by adding each file's duration as an offset.

        This matches the Raven requirement you stated.
        """
        files = self._list_audio_files(audio_folder=audio_folder, recursive=recursive)
        if not files:
            return pd.DataFrame()

        global_offset = 0.0
        all_dfs = []

        for k, ap in enumerate(files, start=1):
            print(f"[{k}/{len(files)}] {ap}")

            df = self.predict_long_audio_windows(
                ap,
                hop_s=hop_s,
                include_all_class_scores=include_all_class_scores,
                use_sigmoid=use_sigmoid,
            )

            if not df.empty:
                df["Begin Time (S)"] = df["Begin Time (S)"] + global_offset
                df["End Time (S)"] = df["End Time (S)"] + global_offset
                all_dfs.append(df)

            # advance global offset by true file duration
            global_offset += self._file_duration_seconds(ap)

        if not all_dfs:
            return pd.DataFrame()

        out = pd.concat(all_dfs, ignore_index=True)

        if sort_by_begin_time and "Begin Time (S)" in out.columns:
            out = out.sort_values("Begin Time (S)").reset_index(drop=True)

        return out

    # ----------------------------
    # Raven export
    # ----------------------------
    def export_to_raven(
        self,
        df: pd.DataFrame,
        raven_file: str = "raven_output.txt",
        low_hz: int = 0,
        high_hz: int = 24000,
        view: str = "Spectrogram 1",
        channel: int = 1,
        class_col: str = "Predicted Class",
        score_col: str = "Top Score",
    ) -> None:
        """
        Export a DataFrame to a Raven selection table (tab-delimited).

        Required Raven fields:
          Selection, View, Channel, Begin Time (S), End Time (S)

        Also writes Low/High freq + class + score.
        """
        if df is None or df.empty:
            # Still write header for convenience
            cols = [
                "Selection", "View", "Channel",
                "Begin Time (S)", "End Time (S)",
                "Low Freq (Hz)", "High Freq (Hz)",
                "Class", "Score",
            ]
            with open(raven_file, "w", encoding="utf-8") as f:
                f.write("\t".join(cols) + "\n")
            print(f"Raven selection table exported (empty) to {raven_file}")
            return

        out = df.copy()

        # Ensure required cols exist
        if "Begin Time (S)" not in out.columns or "End Time (S)" not in out.columns:
            raise ValueError("DataFrame must include 'Begin Time (S)' and 'End Time (S)'.")

        # Map to Raven names
        out["Selection"] = range(1, len(out) + 1)
        out["View"] = view
        out["Channel"] = int(channel)
        out["Low Freq (Hz)"] = int(low_hz)
        out["High Freq (Hz)"] = int(high_hz)

        if class_col not in out.columns:
            raise ValueError(f"Expected class column '{class_col}' not found in DataFrame.")
        if score_col not in out.columns:
            raise ValueError(f"Expected score column '{score_col}' not found in DataFrame.")

        out["Class"] = out[class_col]
        out["Score"] = out[score_col]

        cols = [
            "Selection", "View", "Channel",
            "Begin Time (S)", "End Time (S)",
            "Low Freq (Hz)", "High Freq (Hz)",
            "Class", "Score",
        ]

        with open(raven_file, "w", encoding="utf-8") as f:
            f.write("\t".join(cols) + "\n")
            for _, row in out.iterrows():
                f.write(
                    f"{int(row['Selection'])}\t{row['View']}\t{int(row['Channel'])}\t"
                    f"{float(row['Begin Time (S)']):.6f}\t{float(row['End Time (S)']):.6f}\t"
                    f"{int(row['Low Freq (Hz)'])}\t{int(row['High Freq (Hz)'])}\t"
                    f"{row['Class']}\t{float(row['Score']):.6f}\n"
                )

        print(f"Raven selection table exported to {raven_file}")

if __name__ == "__main__":
    
    
    pred = BirdNetPredictorNew(
        model_path="C:/Users/kaity/Documents/GitHub/EcotypeFinal/BirdNET Models/birdnet07/birdnet07.tflite",
        label_path="C:/Users/kaity/Documents/GitHub/EcotypeFinal/BirdNET Models/birdnet07/birdnet07_8khz_cutoff_Labels.txt",
        audio_folder="C:\\TempData\\TestDays\\Biggs",
        confidence_thresh=0.9,
    )
    
    df = pred.predict_folder_global_raven(hop_s=1.5, recursive=True)
    pred.export_to_raven(df, "C:/TempData\\malahat_global_raven.txt")


    
    





