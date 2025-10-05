import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Optional: only used if you saved the AE
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Import Streamlit for debug output
import streamlit as st


# ---------- 1) Build features EXACTLY like the notebook ----------
def build_features(df: pd.DataFrame):
    """
    Build all features exactly as done in the FRAUD1-checkpoint.ipynb notebook
    """
    debug_output = []
    debug_output.append(f"DEBUG: Original input shape: {df.shape}")
    debug_output.append(f"DEBUG: Original columns: {list(df.columns)}")
    
    d = df.copy()
    
    # Standardize column names (as done in notebook Cell 2)
    d.columns = d.columns.str.strip().str.replace(' ', '_')
    debug_output.append(f"DEBUG: After column cleaning: {list(d.columns)}")
    # Standardize column names (as done in notebook Cell 2)
    d.columns = d.columns.str.strip().str.lower().str.replace(' ', '_')
    debug_output.append(f"DEBUG: After column cleaning: {list(d.columns)}")
    
    # === ADD THIS DIAGNOSTIC PATCH RIGHT HERE ===
    debug_output.append("=== COLUMN NAME DIAGNOSTIC ===")
    
    # Check if the hardcoded columns from your feature engineering actually exist
    hardcoded_columns_used = [
        'date_of_birth(claimant)',  # Used in age calculation
        'age(claimant)',            # Used in age calculation  
        'ailment_date',             # Used in time-based features
        'gender(claimant)',         # Used in required_cols
        'relationship',             # Used in age_rel_mismatch_flag
        'member_suddo',             # Used in repeat_claimant_count
        'claimant_suddo',           # Used in many groupby operations
        'provider',                 # Used in provider features
        'ailments',                 # Used in service charge features
        'broad_benefit',            # Used in benefit flags
        'company',                  # Used in company features
        'total_payable',            # Used in many calculations
        'cover_limit'               # Used in claim_ratio
    ]
    
    missing_columns = []
    for col in hardcoded_columns_used:
        if col in d.columns:
            debug_output.append(f"✓ '{col}' found in dataframe")
        else:
            debug_output.append(f"❌ '{col}' MISSING from dataframe")
            missing_columns.append(col)
    
    if missing_columns:
        debug_output.append("=== MISSING COLUMNS FOUND ===")
        debug_output.append("These columns are used in feature engineering but don't exist after standardization:")
        for col in missing_columns:
            debug_output.append(f"  - {col}")
        
        # Show what similar columns we do have
        debug_output.append("Available similar columns:")
        for missing_col in missing_columns:
            debug_output.append(f"  For '{missing_col}':")
            for actual_col in d.columns:
                if missing_col.replace('(', '').replace(')', '').replace('_', '') in actual_col.replace('(', '').replace(')', '').replace('_', ''):
                    debug_output.append(f"    → '{actual_col}'")
    # === ADD THIS PATCH ===
    # Fix column names to match what your feature engineering code expects
    column_fixes = {
        'date_of_birth(claimant)': 'date_of_birth(claimant)',
        'age(claimant)': 'age(claimant)', 
        'gender(claimant)': 'gender(claimant)',
        'age(main_member)': 'age(main_member)',
        'main_member_gender': 'main_member_gender'
    }
    
    # Check what column names we actually have vs what we need
    debug_output.append("DEBUG: Checking column name mapping:")
    for expected_col in ['date_of_birth(claimant)', 'age(claimant)', 'ailment_date']:
        if expected_col in d.columns:
            debug_output.append(f"DEBUG: ✓ Found {expected_col}")
        else:
            # Look for similar columns
            found = False
            for actual_col in d.columns:
                if expected_col.replace('(', '').replace(')', '') in actual_col.replace('(', '').replace(')', ''):
                    debug_output.append(f"DEBUG: ↪ Using {actual_col} for {expected_col}")
                    column_fixes[expected_col] = actual_col
                    found = True
                    break
            if not found:
                debug_output.append(f"DEBUG: ❌ Could not find {expected_col}")
    # Ensure required columns exist with safe fallbacks
    required_cols = [
        'total_payable', 'cover_limit', 'days_since_last_visit', 'claimant_suddo',
        'ailments', 'provider', 'broad_benefit', 'benefit', 'relationship',
        'gender(claimant)', 'age(claimant)', 'company', 'member_suddo'
    ]
    
    for col in required_cols:
        if col not in d.columns:
            d[col] = np.nan
            debug_output.append(f"DEBUG: Added missing column: {col}")
    
    # Coerce numerics (Cell 2)
    d['total_payable'] = pd.to_numeric(d['total_payable'], errors='coerce')
    d['cover_limit'] = pd.to_numeric(d['cover_limit'], errors='coerce').replace(0, np.nan)
    d['days_since_last_visit'] = pd.to_numeric(d['days_since_last_visit'], errors='coerce')
    d['age(claimant)'] = pd.to_numeric(d['age(claimant)'], errors='coerce')
    
    # Create 'totals' and 'limit_amount' (Cell 2)
    d['totals'] = d['total_payable']
    d['limit_amount'] = d['cover_limit']
    
    # Create 'age' column (Cell 2)
    current_year = pd.Timestamp.now().year
    d['age'] = current_year - pd.to_datetime(d.get('date_of_birth(claimant)', pd.NaT), errors='coerce').dt.year
    d['age'] = d['age'].fillna(d['age(claimant)'])  # Fallback to age(claimant)
    
    # Broad Benefit Category Flags (Cell 4)
    d['broad_benefits_cleaned'] = d['broad_benefit'].astype(str).str.lower().str.strip()
    d['is_inpatient_benefit'] = (d['broad_benefits_cleaned'].str.contains('in-patient|inpatients', na=False)).astype(int)
    d['is_outpatient_benefit'] = (d['broad_benefits_cleaned'].str.contains('out-patient|outpatients', na=False)).astype(int)
    d['is_optical_benefit'] = (d['broad_benefits_cleaned'].str.contains('optical', na=False)).astype(int)
    d['is_dental_benefit'] = (d['broad_benefits_cleaned'].str.contains('dental', na=False)).astype(int)
    d['is_maternity_benefit'] = (d['broad_benefits_cleaned'].str.contains('maternity', na=False)).astype(int)
    d['is_last_expense_benefit'] = (d['broad_benefits_cleaned'].str.contains('last expense', na=False)).astype(int)
    
    # Claim Amount Statistics by Claimant (Cell 5)
    d['claim_amount_mean_by_claimant'] = d.groupby('claimant_suddo')['totals'].transform('mean')
    d['claim_amount_dev_by_claimant'] = d.groupby('claimant_suddo')['totals'].transform('std').fillna(0)
    d['claim_amount_var_by_claimant'] = d.groupby('claimant_suddo')['totals'].transform('var').fillna(0)
    
    # Claim Statistics by Broad Benefit Type (Cell 6)
    d['average_claim_per_broad_benefit'] = d.groupby('broad_benefits_cleaned')['totals'].transform('mean')
    d['deviation_claim_per_broad_benefit'] = d.groupby('broad_benefits_cleaned')['totals'].transform('std').fillna(0)
    
    # Age and Relationship Mismatch Flags (Cell 7)
    d['age_rel_mismatch_flag'] = ((d['relationship'].astype(str).str.upper() == 'CHILD') & (d['age'] > 25)).astype(int)
    d['spouse_gender_mismatch_flag'] = 0  # Simplified for runtime
    
    # Service Charge Statistics by Ailment (Cell 8)
    d['service_charge_mean_by_ailment'] = d.groupby('ailments')['total_payable'].transform('mean')
    d['service_charge_dev_by_ailment'] = d.groupby('ailments')['total_payable'].transform('std').fillna(0)
    d['service_charge_pct_dev_by_ailment'] = (
        (d['total_payable'] - d['service_charge_mean_by_ailment']) / d['service_charge_mean_by_ailment']
    ).fillna(0)
    
    # Provider and Service Charge Statistics (Cell 11)
    d['provider_service_mean_charge'] = d.groupby(['provider', 'ailments'])['total_payable'].transform('mean').fillna(0)
    d['provider_service_dev_charge'] = d.groupby(['provider', 'ailments'])['total_payable'].transform('std').fillna(0)
    d['provider_service_pct_dev_charge'] = (
        (d['total_payable'] - d['provider_service_mean_charge']) / d['provider_service_mean_charge']
    ).fillna(0)
    
    # Repeat Claimant Count (Cell 12)
    d['repeat_claimant_count'] = d.groupby('claimant_suddo')['member_suddo'].transform('count')
    
    # Claim Ratio and High Claim Flag (Cell 13)
    d['claim_ratio'] = (d['total_payable'] / d['cover_limit']).fillna(0)
    d['high_claim_flag'] = (d['total_payable'] > (0.8 * d['cover_limit'])).astype(int)
    
    # Average Claim per User (Cell 14)
    d['average_claim_per_user'] = d.groupby('member_suddo')['total_payable'].transform('mean').fillna(0)
    
    # Claim Amount Variance (Cell 15)
    d['claim_amount_variance_claimant'] = d.groupby('claimant_suddo')['total_payable'].transform('var').fillna(0)
    d['claim_amount_variance_company'] = d.groupby('company')['total_payable'].transform('var').fillna(0)
    
    # Company Total Claims (Cell 16)
    d['company_total_claims'] = d.groupby('company')['total_payable'].transform('sum').fillna(0)
    
    # Medical & Benefit Features (Cell 17)
    le_ailment = LabelEncoder()
    d['ailment_type_encoded'] = le_ailment.fit_transform(d['ailments'].astype(str))
    
    le_benefit = LabelEncoder()
    d['benefit_type_encoded'] = le_benefit.fit_transform(d['benefit'].astype(str))
    
    le_broad_benefits = LabelEncoder()
    d['broad_benefits_encoded'] = le_broad_benefits.fit_transform(d['broad_benefit'].astype(str))
    
    # Diagnosis Group
    def map_diagnosis_group(ailment):
        if pd.isna(ailment):
            return 'Unknown_Diagnosis'
        ailment = str(ailment).lower()
        if 'fever' in ailment:
            return 'Fever_Related'
        elif 'pain' in ailment:
            return 'Pain_Related'
        elif 'dental' in ailment:
            return 'Dental_Related'
        elif 'maternity' in ailment or 'cysis' in ailment:
            return 'Maternity_Related'
        else:
            return 'Other_Diagnosis'
    
    d['diagnosis_group'] = d['ailments'].apply(map_diagnosis_group)
    le_diagnosis_group = LabelEncoder()
    d['diagnosis_group_encoded'] = le_diagnosis_group.fit_transform(d['diagnosis_group'])
    
    # Hospital Risk Score
    d['hospital_risk_score'] = 0
    
    # Behavioral & Derived Features (Cell 18)
    d['is_first_claim_claimant'] = d.groupby('claimant_suddo')['ailment_date'].rank(method='first') == 1
    d['is_first_claim_main_member'] = d.groupby('member_suddo')['ailment_date'].rank(method='first') == 1
    
    # Provider claim z-score
    d['provider_claim_zscore'] = d.groupby('provider')['total_payable'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    ).fillna(0)
    
    # Relationship frequency score
    d['rel_frequency_score'] = d.groupby('relationship')['member_suddo'].transform('count')
    
    # High risk location
    d['is_high_risk_location'] = 0
    
    # Company fraud incident flag
    d['company_fraud_incident_flag'] = 0
    
    # Additional time-based features
    d['day_of_week'] = pd.to_datetime(d.get('ailment_date', pd.NaT), errors='coerce').dt.dayofweek.fillna(0)
    d['month_of_year'] = pd.to_datetime(d.get('ailment_date', pd.NaT), errors='coerce').dt.month.fillna(1)
    
    # Provider behavior statistics
    provider_stats = d.groupby('provider')['total_payable'].agg(['mean', 'count']).reset_index()
    provider_stats.rename(columns={'mean': 'provider_avg_claim', 'count': 'provider_claim_count'}, inplace=True)
    d = d.merge(provider_stats, on='provider', how='left')
    
    # Select final features as per Cell 19
    final_features = [
        'days_since_last_visit',
        'claim_amount_mean_by_claimant', 'claim_amount_dev_by_claimant', 'claim_amount_var_by_claimant',
        'average_claim_per_broad_benefit', 'deviation_claim_per_broad_benefit',
        'age', 'age_rel_mismatch_flag', 'spouse_gender_mismatch_flag',
        'service_charge_mean_by_ailment', 'service_charge_dev_by_ailment', 'service_charge_pct_dev_by_ailment',
        'provider_service_mean_charge', 'provider_service_dev_charge', 'provider_service_pct_dev_charge',
        'repeat_claimant_count', 'claim_ratio', 'high_claim_flag',
        'average_claim_per_user', 'claim_amount_variance_claimant', 'claim_amount_variance_company',
        'company_total_claims',
        'ailment_type_encoded', 'benefit_type_encoded', 'broad_benefits_encoded', 'diagnosis_group_encoded',
        'hospital_risk_score',
        'is_first_claim_claimant', 'is_first_claim_main_member',
        'provider_claim_zscore', 'rel_frequency_score', 'is_high_risk_location', 'company_fraud_incident_flag',
        'is_inpatient_benefit', 'is_outpatient_benefit', 'is_optical_benefit',
        'is_dental_benefit', 'is_maternity_benefit', 'is_last_expense_benefit',
        'day_of_week', 'month_of_year', 'provider_avg_claim', 'provider_claim_count'
    ]
    
    # Ensure all final features exist
    for feat in final_features:
        if feat not in d.columns:
            d[feat] = 0
            debug_output.append(f"DEBUG: Added missing feature: {feat}")
    
    debug_output.append(f"DEBUG: Full features dataframe shape: {d.shape}")
    debug_output.append(f"DEBUG: Full features columns: {list(d.columns)}")
    debug_output.append("DEBUG: Engineered features sample:")
    for feat in final_features[:5]:  # Show first 5 features
        if feat in d.columns:
            debug_output.append(f"  {feat}: {d[feat].iloc[0] if len(d) > 0 else 'N/A'}")
    
    # Store debug output in session state so we can display it
    if 'debug_messages' not in st.session_state:
        st.session_state.debug_messages = []
    st.session_state.debug_messages.extend(debug_output)
    
    # === CRITICAL FIX: Ensure we return ALL columns, not just final_features ===
    debug_output.append("=== BUILD_FEATURES FINAL CHECK ===")
    debug_output.append(f"Final d columns: {list(d.columns)}")
    
    # Check if our key features exist
    key_features = ['claim_ratio', 'service_charge_pct_dev_by_ailment', 'repeat_claimant_count', 
                   'age_rel_mismatch_flag', 'provider_claim_zscore', 'is_maternity_benefit']
    for feat in key_features:
        if feat in d.columns:
            debug_output.append(f"✓ {feat} exists in d - sample: {d[feat].iloc[0] if len(d) > 0 else 'N/A'}")
        else:
            debug_output.append(f"❌ {feat} MISSING in d")
    
    # FIX: Return the FULL dataframe with ALL columns for df_with_features
    # This ensures all engineered features are available for explanations
    return d[final_features], d


# ---------- 2) Rank-averaging helper ----------
def rank01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if np.isnan(a).any():
        a[np.isnan(a)] = np.nanmedian(a)
    r = a.argsort().argsort().astype(float)
    return r / max(len(a) - 1, 1)


# ---------- 3) Optional: load AE if present ----------
def load_autoencoder_if_available(path_pt: str):
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


# ---------- 4) Main scorer with enhanced explanation ----------
class StableAnomalyScorer:
    def __init__(self,
                 model_path: str = "models/anomaly_v1.joblib",
                 tuning_path: str = "models/tuning_v1.json",
                 ae_path: str = "models/autoencoder_v1.pt"
                 ):
        debug_output = ["DEBUG: Loading scorer..."]
        self.art = joblib.load(model_path)
        self.scaler = self.art.get("scaler")
        self.kmeans = self.art.get("kmeans")
        self.iforest = self.art.get("iforest")
        self.feature_names = self.art.get("feature_names")
        self.train_medians = self.art.get("feature_medians")

        with open(tuning_path, "r") as f:
            self.tuning = json.load(f)
        self.threshold = float(self.tuning["threshold_combo_rank"])
        self.version = self.tuning.get("version", "v?")

        self.ae_model, self.ae_in_dim = load_autoencoder_if_available(ae_path)
        debug_output.append(f"DEBUG: Scorer loaded. Feature names: {self.feature_names}")
        
        # --- START PATCH: canonicalize model feature names & build mapping ---
        def canon_name(s):
            if s is None:
                return None
            return (str(s).strip().lower()
                    .replace(' ', '_')
                    .replace('-', '_')
                    .replace('__', '_'))

        # Original model feature names (may have different case/format)
        self.raw_feature_names = self.feature_names[:] if self.feature_names is not None else None
        # Canonical version of the names (used to match runtime-built features)
        if self.raw_feature_names is not None:
            self.canonical_feature_names = [canon_name(f) for f in self.raw_feature_names]
            # Build mapping: canonical -> original (useful for reindexing and reporting)
            self.model_feature_map = {canon: orig for canon, orig in zip(self.canonical_feature_names, self.raw_feature_names)}
        else:
            self.canonical_feature_names = None
            self.model_feature_map = {}
        # --- END PATCH ---
        
        # Store debug output
        if 'debug_messages' not in st.session_state:
            st.session_state.debug_messages = []
        st.session_state.debug_messages.extend(debug_output)

    def compute_components(self, df_raw: pd.DataFrame):
        debug_output = ["DEBUG: Starting compute_components..."]
        X, df_with_features = build_features(df_raw)
        
        # ADD DEBUGGING HERE to check what we're getting:
        debug_output.append(f"DEBUG: Model features (X) shape: {X.shape}")
        debug_output.append(f"DEBUG: Model features columns: {list(X.columns)}")
        debug_output.append(f"DEBUG: Full features df shape: {df_with_features.shape}")
        debug_output.append(f"DEBUG: Full features df columns: {list(df_with_features.columns)}")

        # Check if our key explanation features are in df_with_features
        explanation_features = ['claim_ratio', 'service_charge_pct_dev_by_ailment', 'repeat_claimant_count', 
                            'age_rel_mismatch_flag', 'provider_claim_zscore', 'is_maternity_benefit']
        for feat in explanation_features:
            if feat in df_with_features.columns:
                debug_output.append(f"DEBUG: ✓ Found {feat} in full features")
            else:
                debug_output.append(f"DEBUG: ❌ MISSING {feat} in full features")

        # --- START PATCH: align runtime feature names with model feature names ---
        def canon_name(s):
            if s is None:
                return None
            return (str(s).strip().lower().replace(' ', '_').replace('-', '_').replace('__', '_'))

        # Canonicalize columns in df_with_features and X (but keep originals too)
        df_with_features.columns = [canon_name(c) for c in df_with_features.columns]
        X.columns = [canon_name(c) for c in X.columns]

        if self.canonical_feature_names is not None:
            # Ensure all canonical model feature names exist in X (create NaNs if necessary),
            # but *do not* blindly reindex to original raw names — keep canonical names.
            for f in self.canonical_feature_names:
                if f not in X.columns:
                    X[f] = np.nan

            # Reindex X to model canonical order (safe)
            X = X.reindex(columns=self.canonical_feature_names)

            # Save a note if any canonical model features are missing from runtime features
            runtime_missing = [f for f in self.canonical_feature_names if f not in df_with_features.columns]
            if runtime_missing:
                debug_output.append(f"DEBUG: These model features are not present (canonical) in runtime df: {runtime_missing}")
        # --- END PATCH ---

        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)

        if self.train_medians:
            X = X.fillna(pd.Series(self.train_medians))
        else:
            X = X.fillna(X.median(numeric_only=True))

        Xs = self.scaler.transform(X) if self.scaler is not None else X.values

        if np.isnan(Xs).any():
            Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

        parts = {}

        if self.iforest is not None:
            if_component = -self.iforest.decision_function(Xs)
            parts["if_component"] = if_component

        if self.kmeans is not None:
            kdist = self.kmeans.transform(Xs).min(axis=1)
            parts["kmeans_min_distance"] = kdist

        if self.ae_model is not None and Xs.shape[1] == self.ae_in_dim:
            import torch
            with torch.no_grad():
                xt = torch.tensor(Xs, dtype=torch.float32)
                recon = self.ae_model(xt)
                mae = torch.mean(torch.abs(recon - xt), dim=1).cpu().numpy()
                parts["autoencoder_anomaly_score"] = mae

        debug_output.append(f"DEBUG: Components computed. Parts keys: {list(parts.keys())}")
        
        # Store debug output
        st.session_state.debug_messages.extend(debug_output)
        
        return parts, X.index, df_with_features

    def generate_explanation(self, row):
        """Generate plain English explanations for flagged claims using the engineered features"""
        explanations = []
        
        # Check each feature and generate explanation
        if 'claim_ratio' in row and pd.notna(row['claim_ratio']):
            ratio = row['claim_ratio']
            if ratio > 0.8:
                explanations.append(f"Very high claim ratio ({ratio:.1%} of coverage limit)")
            elif ratio > 0.5:
                explanations.append(f"High claim ratio ({ratio:.1%} of coverage limit)")
        
        if 'service_charge_pct_dev_by_ailment' in row and pd.notna(row['service_charge_pct_dev_by_ailment']):
            dev = row['service_charge_pct_dev_by_ailment']
            if abs(dev) > 0.5:
                direction = "above" if dev > 0 else "below"
                explanations.append(f"Service charge {abs(dev):.0%} {direction} average for this ailment")
            elif abs(dev) > 0.3:
                direction = "above" if dev > 0 else "below"
                explanations.append(f"Service charge {abs(dev):.0%} {direction} typical rate for this ailment")
        
        if 'repeat_claimant_count' in row and pd.notna(row['repeat_claimant_count']):
            count = row['repeat_claimant_count']
            if count > 10:
                explanations.append(f"Very high frequency of claims ({count} claims)")
            elif count > 5:
                explanations.append(f"High frequency of claims ({count} claims)")
        
        if 'age_rel_mismatch_flag' in row and row.get('age_rel_mismatch_flag') == 1:
            explanations.append("Potential age-relationship mismatch (child over 25 years)")
        
        if 'provider_claim_zscore' in row and pd.notna(row['provider_claim_zscore']):
            z = row['provider_claim_zscore']
            if abs(z) > 3.0:
                direction = "above" if z > 0 else "below"
                explanations.append(f"Provider charge {abs(z):.1f} standard deviations {direction} average")
            elif abs(z) > 2.0:
                direction = "above" if z > 0 else "below"
                explanations.append(f"Provider charge {abs(z):.1f} standard deviations {direction} typical")
        
        # High-risk service types
        if row.get('is_maternity_benefit') == 1:
            explanations.append("Maternity service (high-risk category)")
        if row.get('is_dental_benefit') == 1:
            explanations.append("Dental service (high-risk category)")
        if row.get('is_inpatient_benefit') == 1:
            explanations.append("Inpatient service (high-risk category)")
        
        # Fallback if no specific reasons but claim is flagged
        if not explanations and row.get('needs_review') == 1:
            explanations.append("Multiple anomaly detection models flagged this claim as unusual")
        
        return "; ".join(explanations) if explanations else "No specific risk factors identified"

    def score(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        debug_output = ["DEBUG: Starting score method..."]

        # Defensive: normalize incoming column names at the very start so all downstream
        # code works with canonical lowercase_snake_case names and debug logs are consistent.
        def _canon_cols(cols):
            return [str(c).strip().lower().replace(' ', '_').replace('-', '_').replace('__', '_') for c in cols]

        # Make a copy and canonicalize
        df_raw = df_raw.copy()
        original_cols = list(df_raw.columns)
        df_raw.columns = _canon_cols(df_raw.columns)

        debug_output.append(f"DEBUG: Normalized incoming columns. Original -> Canonical mapping:")
        for orig, canon in zip(original_cols, df_raw.columns):
            debug_output.append(f"  '{orig}' -> '{canon}'")

        parts, idx, df_with_features = self.compute_components(df_raw)
        
         # ADD THIS DEBUG CHECK:
        debug_output.append("DEBUG: Checking what features we have before final output:")
        key_features = ['claim_ratio', 'service_charge_pct_dev_by_ailment', 'repeat_claimant_count', 
                    'age_rel_mismatch_flag', 'provider_claim_zscore', 'is_maternity_benefit']
        for feat in key_features:
            if feat in df_with_features.columns:
                debug_output.append(f"DEBUG: ✓ {feat} is available")
            else:
                debug_output.append(f"DEBUG: ❌ {feat} is MISSING")

        if not parts:
            raise ValueError("No anomaly components computed.")

        ranks = [rank01(v) for v in parts.values()]
        combined = np.mean(ranks, axis=0)

        # FIX: Use df_with_features as the base instead of df_raw
        # This contains ALL the engineered features + original data
        out = df_with_features.copy().loc[idx].reset_index(drop=True)
        
        debug_output.append(f"DEBUG: Base output shape (with features): {out.shape}")
        debug_output.append(f"DEBUG: Base output columns: {list(out.columns)}")
        
        # Add model components to the feature-rich dataframe
        out["if_component"] = parts.get("if_component", 0)
        out["kmeans_min_distance"] = parts.get("kmeans_min_distance", 0)
        out["autoencoder_anomaly_score"] = parts.get("autoencoder_anomaly_score", 0)
        out["combined_anomaly_score"] = combined
        out["needs_review"] = (out["combined_anomaly_score"] >= self.threshold).astype(int)
        
        debug_output.append(f"DEBUG: Final output shape: {out.shape}")
        debug_output.append(f"DEBUG: Final output columns: {list(out.columns)}")
        
        # Generate explanations using the engineered features
        out["explanation"] = ""
        for i in range(len(out)):
            row_data = out.iloc[i].to_dict()
            out.at[i, "explanation"] = self.generate_explanation(row_data)
        
        debug_output.append(f"DEBUG: Sample explanations: {out['explanation'].head(3).tolist()}")
        
        # Store debug output
        st.session_state.debug_messages.extend(debug_output)
        
        return out

    def info(self):
        return {
            "version": self.version,
            "threshold": self.threshold,
            "review_rate_target": self.tuning.get("review_rate_target"),
            "created_at": self.tuning.get("created_at"),
        }
