import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import io
import base64
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from io import BytesIO
import traceback
import pyarrow as pa

# --- Page Configuration and Custom Styling ---
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a sleek, modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #004d40;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #00695c;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.25rem;
    }
    .info-box {
        background-color: #f0f2f6;
        color: black;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #ffeeba;
    }
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown('<h1 class="main-header">Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <p>Upload your claims data to automatically detect potential fraudulent claims based on our pre-trained machine learning model.</p>
    <p><strong>How to use:</strong></p>
    <ol>
        <li>Upload a CSV or Excel file containing claims data</li>
        <li>The system will automatically process and analyze your data</li>
        <li>Review the fraud detection results</li>
        <li>Download the results if needed</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# --- File Upload Section ---
st.markdown('<h2 class="sub-header">Upload Claims Data</h2>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=['csv', 'xlsx'],
    help="Drag and drop or browse to select a file. The expected format is a CSV/Excel file with claimant, claim, and visit information."
)

# Show data format requirements
with st.expander("Data Format Requirements"):
    st.markdown("""
    ### Required Columns:
    Your data should include these columns (case-sensitive):
    
    **Claimant Information:**
    - `DATE_OF_BIRTH(CLAIMANT)`: Date of birth (format: DD/MM/YYYY)
    - `AGE(CLAIMANT)`: Age of claimant (numeric)
    - `GENDER(CLAIMANT)`: Gender (M/F)
    - `RELATIONSHIP`: Relationship to main member (SELF, SPOUSE, CHILD)
    
    **Claim Information:**
    - `AILMENTS`: Medical condition
    - `BENEFIT`: Type of benefit
    - `PROVIDER`: Healthcare provider name
    - `CLAIMANT_SUDDO`: Unique claimant identifier
    - `BROAD_BENEFIT`: Broad benefit category
    - `COMPANY`: Company name
    - `COVER_LIMIT`: Coverage limit amount
    
    **Visit Information:**
    - `AILMENT_DATE`: Date of service (format: DD/MM/YYYY)
    - `DAYS_SINCE_LAST_VISIT`: Days since last visit (numeric)
    - `VISIT_ID`: Unique visit identifier
    - `TOTAL_PAYABLE`: Claim amount (numeric)
    
    **Additional Fields:**
    - `FREQUENCY_OF_VISIT`: Frequency of visits (numeric)
    - `MAIN_MEMBER_GENDER`: Gender of main member (M/F)
    - `AGE(MAIN_MEMBER)`: Age of main member (numeric)
    """)

# Example data (with only the column names, no actual data)
example_columns = [
    'AILMENTS', 'BENEFIT', 'DATE_OF_BIRTH(CLAIMANT)', 'AGE(CLAIMANT)', 'GENDER(CLAIMANT)', 
    'RELATIONSHIP', 'AILMENT_DATE', 'DAYS_SINCE_LAST_VISIT', 'DAY_OF_MONTH_VISITED', 
    'MONTH_VISITED', 'YEAR_VISITED', 'PROVIDER', 'MEMBER_SUDDO', 'MAIN_MEMBER_GENDER', 
    'AGE(MAIN_MEMBER)', 'CLAIMANT_SUDDO', 'BROAD_BENEFIT', 'COMPANY', 'COVER_LIMIT', 
    'UNIQUE_VISIT', 'FREQUENCY_OF_VISIT', 'VISIT_ID', 'TOTAL_PAYABLE'
]

# Create a DataFrame with column names only
example_data = {col: [''] for col in example_columns}
example_df = pd.DataFrame(example_data)

# Provide option to download example CSV
def download_csv(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encoding CSV file
    href = f'<a href="data:file/csv;base64,{b64}" download="example_data.csv">Download Example Data (CSV)</a>'
    return href

# Display the "See Preview" button and allow download of example CSV
if st.button("See Preview"):
    st.markdown(download_csv(example_df), unsafe_allow_html=True)


# Define the Autoencoder class (same as in notebook)
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Function to preprocess data
def preprocess_data(df):
    """Preprocess the uploaded data"""
    df_processed = df.copy()
    
    # Clean column names
    df_processed.columns = df_processed.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Convert date columns - handle different date formats
    date_columns = ['ailment_date', 'date_of_birth(claimant)']
    for col in date_columns:
        if col in df_processed.columns:
            # Try multiple date formats
            df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce', dayfirst=True)
    
    # Calculate age if not present
    if 'age(claimant)' not in df_processed.columns and 'date_of_birth(claimant)' in df_processed.columns:
        current_year = pd.Timestamp.now().year
        df_processed['age(claimant)'] = current_year - df_processed['date_of_birth(claimant)'].dt.year
    
    # Convert numeric columns
    numeric_columns = ['total_payable', 'cover_limit', 'days_since_last_visit', 'age(claimant)']
    for col in numeric_columns:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Fill missing values
    df_processed.fillna(0, inplace=True)
    
    return df_processed

# Function to create features
def create_features(df):
    """Create features for fraud detection"""
    df_features = df.copy()
    
    # Basic features
    if 'total_payable' in df_features.columns and 'cover_limit' in df_features.columns:
        df_features['claim_ratio'] = df_features['total_payable'] / df_features['cover_limit']
        df_features['high_claim_flag'] = (df_features['total_payable'] > (0.8 * df_features['cover_limit'])).astype(int)
    
    if 'days_since_last_visit' in df_features.columns:
        df_features['frequent_visitor'] = (df_features['days_since_last_visit'] < 7).astype(int)
    
    # Encode categorical variables
    categorical_cols = ['gender(claimant)', 'relationship', 'broad_benefit', 'company']
    for col in categorical_cols:
        if col in df_features.columns:
            le = LabelEncoder()
            # Handle unseen categories by converting to string and filling NaN
            df_features[col] = df_features[col].astype(str).fillna('Unknown')
            df_features[col + '_encoded'] = le.fit_transform(df_features[col])
    
    return df_features

# Function to run fraud detection
def run_fraud_detection(df):
    """Run the fraud detection pipeline with comprehensive features"""
    
    # Select features for anomaly detection - expanded based on your model output
    features_for_anomaly = [
        'age(claimant)', 'days_since_last_visit', 'claim_ratio', 
        'high_claim_flag', 'frequent_visitor',
        # Additional features from your model output
        'claim_pattern_anomaly_score', 'provider_claim_zscore', 
        'rel_frequency_score', 'autoencoder_anomaly_score',
        'combined_anomaly_score', 'hospital_risk_score',
        'mismatch_score', 'provider_service_pct_dev_charge',
        'claim_amount_dev_by_claimant', 'service_charge_pct_dev_by_ailment'
    ]
    
    # Filter to available features
    available_features = [f for f in features_for_anomaly if f in df.columns]
    
    if not available_features or len(df) < 10:
        st.error("Not enough features or data for fraud detection.")
        return None, None
    
    # Prepare data for anomaly detection
    X = df[available_features].fillna(0)
    
    # Remove infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run Isolation Forest
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso_scores = -iso.fit_predict(X_scaled)
    
    # Run KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_clusters = kmeans.fit_predict(X_scaled)
    
    # Calculate reconstruction error (simplified)
    reconstruction_error = np.mean((X_scaled - np.mean(X_scaled, axis=0))**2, axis=1)
    
    # Combine scores
    combined_scores = (iso_scores + reconstruction_error) / 2
    
    # Apply threshold
    threshold = np.quantile(combined_scores, 0.95)
    fraud_predictions = (combined_scores > threshold).astype(int)
    
    # Create risk levels
    risk_levels = []
    for score in combined_scores:
        if score > np.quantile(combined_scores, 0.95):
            risk_levels.append("Very High")
        elif score > np.quantile(combined_scores, 0.85):
            risk_levels.append("High")
        elif score > np.quantile(combined_scores, 0.75):
            risk_levels.append("Medium")
        else:
            risk_levels.append("Low")
    
    # Create results dataframe
    results = pd.DataFrame({
        'fraud_prediction': fraud_predictions,
        'fraud_score': combined_scores,
        'risk_level': risk_levels,
        'iso_score': iso_scores,
        'reconstruction_error': reconstruction_error
    })
    
    # Calculate thresholds for different risk factors
    thresholds = {}
    
    def get_series(name):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return None
    
    # Claim pattern anomaly score
    s = get_series("claim_pattern_anomaly_score")
    thresholds["claim_pattern_high"] = s.quantile(0.9) if s is not None else None
    
    # Provider z-score
    s = get_series("provider_claim_zscore")
    thresholds["provider_z_high"] = 3.0 if s is not None else None
    thresholds["provider_z_very_high"] = 5.0 if s is not None else None
    
    # Visit frequency / recency
    s = get_series("rel_frequency_score")
    thresholds["rel_freq_high"] = s.quantile(0.9) if s is not None else None
    thresholds["rel_freq_vhigh"] = s.quantile(0.95) if s is not None else None
    
    s = get_series("days_since_last_visit")
    thresholds["recent_visit"] = s.quantile(0.1) if s is not None else None
    
    s = get_series("days_since_last_provider_visit")
    thresholds["recent_same_provider"] = s.quantile(0.1) if s is not None else None
    
    s = get_series("days_since_last_ailment")
    thresholds["recent_ailment"] = s.quantile(0.1) if s is not None else None
    
    # Autoencoder / combined anomaly
    s = get_series("autoencoder_anomaly_score")
    thresholds["ae_high"] = s.quantile(0.9) if s is not None else None
    
    s = get_series("combined_anomaly_score")
    thresholds["combined_high"] = s.quantile(0.9) if s is not None else None
    
    # Hospital risk & mismatches
    s = get_series("hospital_risk_score")
    thresholds["hospital_risky"] = 1.0 if s is not None else None
    
    s = get_series("mismatch_score")
    thresholds["mismatch_any"] = 1.0 if s is not None else None
    thresholds["mismatch_many"] = 3.0 if s is not None else None
    
    # Add user-friendly reasons for flagging
    reasons = []
    for idx, row in results.iterrows():
        reason_parts = []
        
        # Get the corresponding row from the original data
        data_row = df.iloc[idx] if idx < len(df) else {}
        
        # Check for unusual patterns
        if 'claim_pattern_anomaly_score' in data_row and thresholds["claim_pattern_high"] is not None:
            score_val = pd.to_numeric(data_row["claim_pattern_anomaly_score"], errors="coerce")
            if pd.notna(score_val) and score_val >= thresholds["claim_pattern_high"]:
                reason_parts.append("Unusual claim pattern compared to peers")
        
        # Check for provider outliers
        if 'provider_claim_zscore' in data_row and thresholds["provider_z_high"] is not None:
            z = pd.to_numeric(data_row["provider_claim_zscore"], errors="coerce")
            if pd.notna(z) and z >= thresholds["provider_z_very_high"]:
                reason_parts.append("Provider appears as a strong outlier (very high claim amount)")
            elif pd.notna(z) and z >= thresholds["provider_z_high"]:
                reason_parts.append("Provider appears as an outlier (high claim amount)")
        
        # Check for frequency anomalies
        if 'rel_frequency_score' in data_row and thresholds["rel_freq_high"] is not None:
            f = pd.to_numeric(data_row["rel_frequency_score"], errors="coerce")
            if pd.notna(f) and f >= thresholds["rel_freq_vhigh"]:
                reason_parts.append("Very frequent visits in a short period")
            elif pd.notna(f) and f >= thresholds["rel_freq_high"]:
                reason_parts.append("Higher-than-normal visit frequency")
        
        # Check for recency issues
        if 'days_since_last_visit' in data_row and thresholds["recent_visit"] is not None:
            v = pd.to_numeric(data_row["days_since_last_visit"], errors="coerce")
            if pd.notna(v) and v <= thresholds["recent_visit"]:
                reason_parts.append("Visited again very soon after a previous visit")
        
        if 'days_since_last_provider_visit' in data_row and thresholds["recent_same_provider"] is not None:
            v = pd.to_numeric(data_row["days_since_last_provider_visit"], errors="coerce")
            if pd.notna(v) and v <= thresholds["recent_same_provider"]:
                reason_parts.append("Repeat visit to the same provider within a short time")
        
        if 'days_since_last_ailment' in data_row and thresholds["recent_ailment"] is not None:
            v = pd.to_numeric(data_row["days_since_last_ailment"], errors="coerce")
            if pd.notna(v) and v <= thresholds["recent_ailment"]:
                reason_parts.append("New ailment recorded unusually soon after a previous one")
        
        # Check for data mismatches
        if 'mismatch_score' in data_row and thresholds["mismatch_any"] is not None:
            m = pd.to_numeric(data_row["mismatch_score"], errors="coerce")
            if pd.notna(m) and m >= thresholds["mismatch_many"]:
                reason_parts.append("Multiple data mismatches in claim details")
            elif pd.notna(m) and m >= thresholds["mismatch_any"]:
                reason_parts.append("Some data mismatches in claim details")
        
        # Check for hospital risk
        if 'hospital_risk_score' in data_row and thresholds["hospital_risky"] is not None:
            h = pd.to_numeric(data_row["hospital_risk_score"], errors="coerce")
            if pd.notna(h) and h >= thresholds["hospital_risky"]:
                reason_parts.append("Provider flagged with prior risk indicators")
        
        # Check for high risk location
        if 'is_high_risk_location' in data_row:
            loc_risk = data_row["is_high_risk_location"]
            if loc_risk in [1, True, "1", "True", "true"]:
                reason_parts.append("Location is known to be high risk")
        
        # Check for company fraud incidents
        if 'company_fraud_incident_flag' in data_row:
            company_flag = data_row["company_fraud_incident_flag"]
            if company_flag in [1, True, "1", "True", "true"]:
                reason_parts.append("Company has prior fraud incidents")
        
        # Check for model anomalies
        if 'autoencoder_anomaly_score' in data_row and thresholds["ae_high"] is not None:
            ae = pd.to_numeric(data_row["autoencoder_anomaly_score"], errors="coerce")
            if pd.notna(ae) and ae >= thresholds["ae_high"]:
                reason_parts.append("AI model flagged this claim as unusual")
        
        if 'combined_anomaly_score' in data_row and thresholds["combined_high"] is not None:
            ca = pd.to_numeric(data_row["combined_anomaly_score"], errors="coerce")
            if pd.notna(ca) and ca >= thresholds["combined_high"]:
                reason_parts.append("Overall anomaly score is high")
        
        # Check for high claim amounts
        if 'claim_ratio' in data_row:
            claim_ratio = data_row['claim_ratio']
            if pd.notna(claim_ratio) and claim_ratio > 0.8:
                reason_parts.append(f"Claim amount is high relative to coverage limit ({claim_ratio:.0%})")
        
        # Check for frequent visits
        if 'days_since_last_visit' in data_row:
            days_since = data_row['days_since_last_visit']
            if pd.notna(days_since) and days_since < 7:
                reason_parts.append(f"Frequent visits (last visit was only {int(days_since)} days ago)")
        
        # Check for age mismatches
        if 'age_rel_mismatch_flag' in data_row:
            age_mismatch = data_row['age_rel_mismatch_flag']
            if age_mismatch in [1, True, "1", "True", "true"]:
                reason_parts.append("Age relationship mismatch detected")
        
        # Check for gender mismatches
        if 'spouse_gender_mismatch_flag' in data_row:
            gender_mismatch = data_row['spouse_gender_mismatch_flag']
            if gender_mismatch in [1, True, "1", "True", "true"]:
                reason_parts.append("Spouse gender mismatch detected")
        
        # Check for service charge deviations
        if 'service_charge_pct_dev_by_ailment' in data_row:
            dev = data_row['service_charge_pct_dev_by_ailment']
            if pd.notna(dev) and abs(dev) > 1.0:  # More than 100% deviation
                direction = "above" if dev > 0 else "below"
                reason_parts.append(f"Service charge significantly {direction} average for this ailment")
        
        # If no specific reasons but still flagged, provide more specific explanation
        if not reason_parts and row['fraud_score'] > threshold:
            if row['iso_score'] > np.quantile(results['iso_score'], 0.9):
                reason_parts.append("Unusual pattern compared to peers")
            if row['reconstruction_error'] > np.quantile(results['reconstruction_error'], 0.9):
                reason_parts.append("Anomalous claim characteristics")
            if not reason_parts:
                reason_parts.append("Elevated composite risk score from multiple factors")
        
        # Format the reasons with semicolon separator (no bullet points)
        if reason_parts:
            reasons.append("; ".join(reason_parts))
        else:
            reasons.append("No significant risk factors detected")
    
    results['reasons'] = reasons
    
    return results, X_scaled

# Process uploaded file
if uploaded_file is not None:
    try:
        # Read the file based on type
        if uploaded_file.name.endswith('.csv'):
            # Try multiple encodings
            encodings_to_try = ['utf-8', 'ISO-8859-1', 'latin-1', 'cp1252', 'windows-1252']
            df = None
            
            for encoding in encodings_to_try:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding, on_bad_lines='skip', low_memory=False)
                    st.success(f"Successfully read file with {encoding} encoding")
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            
            if df is None:
                # Final attempt with error handling
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding_errors='ignore', on_bad_lines='skip', low_memory=False)
                    st.success("Successfully read file with error handling")
                except Exception as e:
                    st.error(f"Could not read CSV file: {str(e)}")
                    st.stop()
                    
        else:  # Excel file
            try:
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            except:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_excel(uploaded_file, engine='xlrd')
                except Exception as e:
                    st.error(f"Could not read Excel file: {str(e)}")
                    st.stop()
        
        # Check if dataframe was created successfully
        if df is None or df.empty:
            st.error("Failed to read the uploaded file or file is empty.")
            st.stop()
            
        # Show basic info about the data
        st.markdown('<h3 class="sub-header">Data Information</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Number of rows:** {len(df)}")
            st.write(f"**Number of columns:** {len(df.columns)}")
        with col2:
            st.write(f"**File name:** {uploaded_file.name}")
            
        # Show preview
        st.markdown('<h3 class="sub-header">Data Preview</h3>', unsafe_allow_html=True)
        st.dataframe(df.head())
        
        # Process data button
        if st.button("Process Data for Fraud Detection", type="primary"):
            with st.spinner("Processing data and running fraud detection..."):
                try:
                    start_time = time.time()
                    
                    # Preprocess data
                    df_processed = preprocess_data(df.copy())
                    
                    # Create features
                    df_with_features = create_features(df_processed)
                    
                    # Run fraud detection
                    results, X_scaled = run_fraud_detection(df_with_features)
                    
                    if results is not None:
                        # Combine results with original data
                        final_results = pd.concat([df_processed, results], axis=1)
                        
                        processing_time = time.time() - start_time
                        
                        # Display results
                        st.markdown('<h2 class="sub-header">Fraud Detection Results</h2>', unsafe_allow_html=True)
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        total_claims = len(final_results)
                        flagged_claims = final_results['fraud_prediction'].sum()
                        
                        with col1:
                            st.metric("Total Claims", total_claims)
                        with col2:
                            st.metric("Flagged Claims", flagged_claims)
                        with col3:
                            st.metric("Flag Rate", f"{(flagged_claims/total_claims*100):.2f}%")
                        with col4:
                            high_risk = len(final_results[final_results['risk_level'].isin(['High', 'Very High'])])
                            st.metric("High Risk Claims", high_risk)
                        
                        # Display results table
                        display_columns = []
                        for col in ['visit_id', 'gender(claimant)', 'age(claimant)', 'provider', 
                                   'total_payable', 'fraud_prediction', 'risk_level', 'fraud_score', 'reasons']:
                            if col in final_results.columns:
                                display_columns.append(col)
                        
                        st.dataframe(final_results[display_columns].sort_values(by='fraud_score', ascending=False).head(20))
                        
                        # Visualizations
                        st.markdown('<h3 class="sub-header">Visualizations</h3>', unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown('### Risk Level Distribution')
                            fig, ax = plt.subplots(figsize=(8, 6))
                            risk_counts = final_results['risk_level'].value_counts()
                            colors = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']
                            risk_counts.reindex(['Low', 'Medium', 'High', 'Very High']).plot(
                                kind='pie', autopct='%1.1f%%', ax=ax, colors=colors)
                            ax.set_ylabel('')
                            st.pyplot(fig)
                        
                        with col2:
                            st.markdown('### Fraud Score Distribution')
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.histplot(final_results['fraud_score'], bins=20, kde=True, ax=ax, color='#00695c')
                            ax.axvline(final_results['fraud_score'].quantile(0.95), color='r', 
                                      linestyle='--', label='95% Threshold')
                            ax.set_title('Composite Fraud Score')
                            ax.legend()
                            st.pyplot(fig)
                        
                        # Download options
                        st.markdown('<h3 class="sub-header">Download Results</h3>', unsafe_allow_html=True)
                        
                        # Convert to CSV
                        csv = final_results.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name='fraud_detection_results.csv',
                            mime='text/csv'
                        )
                        
                        st.markdown(f"**Processing Time:** {processing_time:.2f} seconds")
                        
                    else:
                        st.error("Fraud detection failed. Please check your data format.")
                        
                except Exception as processing_error:
                    st.error(f"Error during processing: {str(processing_error)}")
                    st.code(traceback.format_exc())
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.code(traceback.format_exc())