from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


DATASET_PATH = Path("dataset/audio_va.csv")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "music_va_gmm_v2.pkl"
#INPUT_COLS = ["valence_mean", "valence_std", "arousal_mean", "arousal_std"]
INPUT_COLS = ["valence_mean", "arousal_mean"]
OUTPUT_COLS = [
    "tempo",
    "density",
    "mean_pitch",
    "brightness",
    "volatility",
    "pitch_range",
    "wetness",
]
FEATURE_COLS = INPUT_COLS + OUTPUT_COLS


def process_large_dataset(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in dataset: {missing_cols}")

    raw_data = df[FEATURE_COLS].to_numpy(dtype=float)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(raw_data)

    return scaled_data, scaler, FEATURE_COLS


def train_complex_gmm(data, n_components=10):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=200,
        random_state=42,
    )
    gmm.fit(data)
    return gmm


def _gaussian_log_pdf(x, mean, covariance):
    dim = x.shape[0]
    regularized_cov = covariance + np.eye(dim) * 1e-6
    diff = x - mean
    sign, logdet = np.linalg.slogdet(regularized_cov)
    if sign <= 0:
        raise np.linalg.LinAlgError("Covariance matrix is not positive definite.")

    mahalanobis = diff @ np.linalg.solve(regularized_cov, diff)
    return -0.5 * (dim * np.log(2 * np.pi) + logdet + mahalanobis)


def sample_with_scaling(v, a, model, scaler):
    input_values = np.array([v, a], dtype=float)
    input_dim = len(INPUT_COLS)
    scaled_input = (input_values - scaler.mean_[:input_dim]) / scaler.scale_[:input_dim]

    input_idx = list(range(input_dim))
    output_idx = list(range(input_dim, len(scaler.mean_)))

    log_weights = []
    for comp in range(model.n_components):
        mu = model.means_[comp]
        cov = model.covariances_[comp]
        mu_1 = mu[input_idx]
        sig_11 = cov[np.ix_(input_idx, input_idx)]
        log_prior = np.log(model.weights_[comp] + 1e-12)
        log_likelihood = _gaussian_log_pdf(scaled_input, mu_1, sig_11)
        log_weights.append(log_prior + log_likelihood)

    log_weights = np.array(log_weights)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= weights.sum()

    comp = np.random.choice(model.n_components, p=weights)
    mu = model.means_[comp]
    sig = model.covariances_[comp]

    mu_1, mu_2 = mu[input_idx], mu[output_idx]
    sig_11 = sig[np.ix_(input_idx, input_idx)]
    sig_12 = sig[np.ix_(input_idx, output_idx)]
    sig_21 = sig[np.ix_(output_idx, input_idx)]
    sig_22 = sig[np.ix_(output_idx, output_idx)]

    inv_sig_11 = np.linalg.inv(sig_11 + np.eye(len(input_idx)) * 1e-6)
    cond_mu = mu_2 + sig_21 @ inv_sig_11 @ (scaled_input - mu_1)
    cond_sig = sig_22 - sig_21 @ inv_sig_11 @ sig_12
    cond_sig += np.eye(len(output_idx)) * 1e-6

    scaled_sample = np.random.multivariate_normal(cond_mu, cond_sig)

    full_sample_scaled = np.concatenate([scaled_input, scaled_sample])
    full_sample_original = scaler.inverse_transform(full_sample_scaled.reshape(1, -1))

    return full_sample_original[0, input_dim:]


def main():
    scaled_data, scaler, cols = process_large_dataset(DATASET_PATH)
    model = train_complex_gmm(scaled_data)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_data = {
        "gmm_model": model,
        "data_scaler": scaler,
        "feature_names": cols,
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    v_in, a_in = 0.2, 0.8
    params = sample_with_scaling(v_in, a_in, model, scaler)

    print(f"Generated music parameters for valence={v_in}, arousal={a_in}:")
    for name, val in zip(cols[len(INPUT_COLS):], params):
        print(f"{name:12}: {val:.4f}")


if __name__ == "__main__":
    main()
