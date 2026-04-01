import argparse
from pathlib import Path

import joblib
import numpy as np


MODEL_PATH = Path("model/music_va_gmm_v1.pkl")
INPUT_COLS = ["valence_mean", "valence_std", "arousal_mean", "arousal_std"]
NON_NEGATIVE_FEATURES = {
    "tempo",
    "density",
    "mean_pitch",
    "brightness",
    "volatility",
    "pitch_range",
    "wetness",
}


def _gaussian_log_pdf(x, mean, covariance):
    dim = x.shape[0]
    regularized_cov = covariance + np.eye(dim) * 1e-6
    diff = x - mean
    sign, logdet = np.linalg.slogdet(regularized_cov)
    if sign <= 0:
        raise np.linalg.LinAlgError("Covariance matrix is not positive definite.")

    mahalanobis = diff @ np.linalg.solve(regularized_cov, diff)
    return -0.5 * (dim * np.log(2 * np.pi) + logdet + mahalanobis)


def load_model(model_path):
    model_data = joblib.load(model_path)
    return (
        model_data["gmm_model"],
        model_data["data_scaler"],
        model_data["feature_names"],
    )


def postprocess_sample(sampled):
    processed = {}
    for name, value in sampled.items():
        if name in NON_NEGATIVE_FEATURES:
            value = max(0.0, value)
        if name == "pitch_range":
            value = float(round(value))
        processed[name] = value
    return processed


def sample_music_params(
    valence,
    arousal,
    model,
    scaler,
    feature_names,
    rng,
    valence_std=0.0,
    arousal_std=0.0,
    temperature = 0.1
):
    input_values = np.array([valence, valence_std, arousal, arousal_std], dtype=float)
    input_dim = len(INPUT_COLS)
    scaled_input = (input_values - scaler.mean_[:input_dim]) / scaler.scale_[:input_dim]

    input_idx = list(range(input_dim))
    output_idx = list(range(input_dim, len(feature_names)))

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

    #comp = rng.choice(model.n_components, p=weights)
    comp = np.argmax(weights)
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

    scaled_sample = rng.multivariate_normal(cond_mu, cond_sig * temperature)
    full_sample_scaled = np.concatenate([scaled_input, scaled_sample])
    full_sample_original = scaler.inverse_transform(full_sample_scaled.reshape(1, -1))[0]

    output_feature_names = feature_names[input_dim:]
    output_values = full_sample_original[input_dim:]
    return postprocess_sample(dict(zip(output_feature_names, output_values)))


def parse_args():
    parser = argparse.ArgumentParser(description="Sample music parameters from a trained GMM.")
    parser.add_argument("--valence", type=float, default=0.2, help="Target valence mean.")
    parser.add_argument("--arousal", type=float, default=0.8, help="Target arousal mean.")
    parser.add_argument("--valence-std", type=float, default=0.0, help="Target valence std.")
    parser.add_argument("--arousal-std", type=float, default=0.0, help="Target arousal std.")
    parser.add_argument("--samples", type=int, default=3, help="Number of random samples to generate.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH, help="Path to saved model file.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    model, scaler, feature_names = load_model(args.model_path)
    rng = np.random.default_rng(args.seed)

    print(f"Loaded model from {args.model_path}")
    print(
        "Sampling with "
        f"valence_mean={args.valence}, valence_std={args.valence_std}, "
        f"arousal_mean={args.arousal}, arousal_std={args.arousal_std}"
    )

    for sample_index in range(1, args.samples + 1):
        sampled = sample_music_params(
            valence=args.valence,
            arousal=args.arousal,
            valence_std=args.valence_std,
            arousal_std=args.arousal_std,
            model=model,
            scaler=scaler,
            feature_names=feature_names,
            rng=rng,
        )
        print(f"\nSample {sample_index}:")
        for name, value in sampled.items():
            print(f"{name:12}: {value:.4f}")


if __name__ == "__main__":
    main()
