import os
import json
import joblib
import numpy as np
import pandas as pd

# Optional: only used if you saved the AE
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# ---------- 1) Build features EXACTLY like the notebook ----------
# Edit this to match how you built X in FRAUD1.ipynb.
# Based on your sample head, these columns exist:
# - TOTAL_PAYABLE (float)
# - COVER_LIMIT (float)
# - DAYS_SINCE_LAST_VISIT (int)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Ensure required columns exist (create safe fallbacks if missing)
    for c in ["TOTAL_PAYABLE", "COVER_LIMIT", "DAYS_SINCE_LAST_VISIT"]:
        if c not in d.columns:
            d[c] = np.nan

    # Coerce numerics
    d["TOTAL_PAYABLE"] = pd.to_numeric(d["TOTAL_PAYABLE"], errors="coerce")
    d["COVER_LIMIT"] = pd.to_numeric(d["COVER_LIMIT"], errors="coerce").replace(0, np.nan)
    d["DAYS_SINCE_LAST_VISIT"] = pd.to_numeric(d["DAYS_SINCE_LAST_VISIT"], errors="coerce")

    # Simple, stable features (mirror your notebook’s spirit)
    d["LOG_TOTAL"] = np.log1p(d["TOTAL_PAYABLE"].clip(lower=0))
    d["AMOUNT_RATIO"] = (d["TOTAL_PAYABLE"] / d["COVER_LIMIT"]).clip(upper=50)
    d["GAP"] = d["DAYS_SINCE_LAST_VISIT"].fillna(d["DAYS_SINCE_LAST_VISIT"].median())

    feats = ["LOG_TOTAL", "AMOUNT_RATIO", "GAP"]

    # Fill any remaining NaNs with column medians
    return d[feats].fillna(d[feats].median())


# ---------- 2) Rank-averaging helper ----------
def rank01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if np.isnan(a).any():
        a[np.isnan(a)] = np.nanmedian(a)
    r = a.argsort().argsort().astype(float)
    return r / max(len(a) - 1, 1)


# ---------- 3) Optional: load AE if present ----------
def load_autoencoder_if_available(path_pt: str):
    """
    Expects torch.save({'state_dict': ..., 'meta': {'input_dim': ...}}, path_pt)
    Returns (model, input_dim) or (None, None) if missing/unavailable.
    """
    if not (TORCH_AVAILABLE and os.path.exists(path_pt)):
        return None, None

    ckpt = torch.load(path_pt, map_location="cpu")
    meta = ckpt.get("meta", {})
    input_dim = int(meta.get("input_dim", 0))

    import torch.nn as nn

    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, input_dim),
            )
        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    model = Autoencoder(input_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, input_dim


# ---------- 4) Main scorer ----------
class StableAnomalyScorer:
    def __init__(self,
                 model_path: str = "models/anomaly_v1.joblib",
                 tuning_path: str = "models/tuning_v1.json",
                 ae_path: str = "models/autoencoder_v1.pt"  # optional
                 ):
        # Load sklearn artifacts
        self.art = joblib.load(model_path)
        self.scaler = self.art.get("scaler")
        self.kmeans = self.art.get("kmeans")
        self.iforest = self.art.get("iforest")
        self.feature_names = self.art.get("feature_names")
        self.train_medians = self.art.get("feature_medians")  # dict or None


        # Load tuning (frozen threshold)
        with open(tuning_path, "r") as f:
            self.tuning = json.load(f)
        self.threshold = float(self.tuning["threshold_combo_rank"])
        self.version = self.tuning.get("version", "v?")

        # Optional AE
        self.ae_model, self.ae_in_dim = load_autoencoder_if_available(ae_path)

    def compute_components(self, df_raw: pd.DataFrame):
        # 1) Build features
        X = build_features(df_raw)

        # 2) Align to training order; create any missing training columns
        if self.feature_names is not None:
            # Ensure all training columns exist
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = np.nan
            # Drop any unexpected extras
            X = X.reindex(columns=self.feature_names)

        # 3) Force numeric & clean infinities
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)

        # 4) Impute: median of the current batch (since we didn't save medians yet)
        #    (Proper fix below will use *training* medians instead.)
        if self.train_medians:
        # Use training medians for imputation (stable & reproducible)
            X = X.fillna(pd.Series(self.train_medians))
        else:
         X = X.fillna(X.median(numeric_only=True))


        # 5) Scale
        Xs = self.scaler.transform(X) if self.scaler is not None else X.values

        # 6) Last-resort guard in case something still slipped through
        if np.isnan(Xs).any():
            Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

        parts = {}

        # Isolation Forest component
        if self.iforest is not None:
            if_component = -self.iforest.decision_function(Xs)  # higher = riskier
            parts["if_component"] = if_component

        # KMeans distance to nearest centroid
        if self.kmeans is not None:
            kdist = self.kmeans.transform(Xs).min(axis=1)  # higher = riskier
            parts["kmeans_min_distance"] = kdist

        # Autoencoder (optional)
        if self.ae_model is not None and Xs.shape[1] == self.ae_in_dim:
            import torch
            with torch.no_grad():
                xt = torch.tensor(Xs, dtype=torch.float32)
                recon = self.ae_model(xt)
                mae = torch.mean(torch.abs(recon - xt), dim=1).cpu().numpy()
                parts["autoencoder_anomaly_score"] = mae

        return parts, X.index # return index to align back to df

    def score(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        parts, idx = self.compute_components(df_raw)

        if not parts:
            raise ValueError("No anomaly components computed. Ensure models were saved correctly.")

        ranks = [rank01(v) for v in parts.values()]
        combined = np.mean(ranks, axis=0)  # 0..1, higher = riskier

        out = df_raw.copy()
        out = out.loc[idx]  # align
        out["if_component"] = parts.get("if_component")
        out["kmeans_min_distance"] = parts.get("kmeans_min_distance")
        out["autoencoder_anomaly_score"] = parts.get("autoencoder_anomaly_score")
        out["combined_anomaly_score"] = combined
        out["needs_review"] = out["combined_anomaly_score"] >= self.threshold
        return out

    def info(self):
        return {
            "version": self.version,
            "threshold": self.threshold,
            "review_rate_target": self.tuning.get("review_rate_target"),
            "created_at": self.tuning.get("created_at"),
        }


# ---------- 5) Convenience CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV/XLSX with the same columns expected by build_features()")
    ap.add_argument("--out", default="scored.csv")
    ap.add_argument("--model", default="models/anomaly_v1.joblib")
    ap.add_argument("--tuning", default="models/tuning_v1.json")
    ap.add_argument("--ae", default="models/autoencoder_v1.pt")
    args = ap.parse_args()

    # Read file
    if args.input.lower().endswith(".xlsx"):
        df_in = pd.read_excel(args.input)
    else:
        # Try UTF-8 then latin-1 (as you did)
        try:
            df_in = pd.read_csv(args.input, encoding="utf-8")
        except UnicodeDecodeError:
            df_in = pd.read_csv(args.input, encoding="latin-1")

    scorer = StableAnomalyScorer(args.model, args.tuning, args.ae)
    scored = scorer.score(df_in)
    scored.to_csv(args.out, index=False)
    print(f"[OK] Scored {len(scored):,} rows • saved -> {args.out}")
    print(f"[INFO] Model {scorer.info()['version']} • threshold={scorer.info()['threshold']:.3f}")
