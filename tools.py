from pathlib import Path

import librosa
import numpy as np


def analyze_mp3_features(file_path):
    file_path = Path(file_path)

    y, sr = librosa.load(file_path, sr=None)

    tempo_array, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo_array)[0])

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    density = len(onset_frames) / duration if duration else 0.0

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_pitch_proxy = np.mean(centroid)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    brightness = np.mean(rolloff)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    pitch_indices = np.argmax(chroma, axis=0)
    pitch_diffs = np.diff(pitch_indices)
    volatility = np.std(pitch_diffs) if pitch_diffs.size else 0.0
    pitch_range = np.max(pitch_indices) - np.min(pitch_indices)

    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    wetness_proxy = np.mean(zcr)

    return {
        "tempo": tempo,
        "density": float(density),
        "mean_pitch": float(mean_pitch_proxy),
        "brightness": float(brightness),
        "volatility": float(volatility),
        "pitch_range": float(pitch_range),
        "wetness": float(wetness_proxy),
    }
