import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Styling ---
MINET_RED = "#C0392B"
MINET_BLUE = "#2C3E50"
MINET_LIGHT = "#ECF0F1"
BACKGROUND_COLOR = "#F8F9FA"
CARD_SHADOW = "0 6px 18px rgba(0,0,0,0.08)"

# Apply custom CSS (clean, modern) - no emojis
st.markdown(f"""
<style>
    .main .block-container {{
        padding-top: 2rem;
        background-color: {BACKGROUND_COLOR};
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    }}

    .metric-card {{
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: {CARD_SHADOW};
        border-left: 6px solid {MINET_BLUE};
        margin-bottom: 1rem;
    }}

    .category-card {{
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: {CARD_SHADOW};
        border-left: 4px solid;
        margin-bottom: 1rem;
        transition: transform 0.12s ease;
    }}

    .category-card:hover {{
        transform: translateY(-3px);
    }}

    .risk-high {{
        color: {MINET_RED};
        font-weight: 700;
    }}

    .risk-medium {{
        color: #E67E22;
        font-weight: 700;
    }}

    .risk-low {{
        color: #27AE60;
        font-weight: 700;
    }}

    .header-section {{
        background: linear-gradient(135deg, {MINET_BLUE}, #1A252F);
        padding: 1.6rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }}

    .feature-pill {{
        display: inline-block;
        background: {MINET_LIGHT};
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
        font-size: 0.82rem;
        margin: 0.22rem;
    }}

    .section-divider {{
        border-top: 1px solid {MINET_LIGHT};
        margin: 1.6rem 0;
    }}
</style>
""", unsafe_allow_html=True)

# --- Category definitions (unchanged semantics, ordered and clean) ---
FEATURE_CATEGORIES = [
    {
        "key": "Billing Anomalies",
        "description": "Unusual pricing and extreme claim amounts",
        "features": [
            "provider_claim_zscore",
            "provider_service_pct_dev_charge",
            "service_charge_pct_dev_by_ailment",
            "claim_ratio",
            "high_claim_flag",
        ],
        "color": MINET_RED,
    },
    {
        "key": "Usage Frequency Issues",
        "description": "Consecutive visits and high visit frequency",
        "features": ["days_since_last_visit", "repeat_claimant_count", "frequecy_of_visit", "day_of_week"],
        "color": "#E67E22",
    },
    {
        "key": "Claimant History Patterns",
        "description": "Very frequent claimants and claim variance",
        "features": ["repeat_claimant_count", "average_claim_per_user", "claim_amount_variance_claimant"],
        "color": "#9B59B6",
    },
    {
        "key": "Demographic Mismatches",
        "description": "Age-relationship issues and spouse mismatches",
        "features": ["age_rel_mismatch_flag", "spouse_gender_mismatch_flag", "age"],
        "color": "#3498DB",
    },
    {
        "key": "Medical-Demographic Mismatches",
        "description": "Gender-condition or similar mismatches",
        "features": ["mismatch_score", "age_rel_mismatch_flag"],
        "color": MINET_BLUE,
    },
    {
        "key": "High-Risk Service Types",
        "description": "Maternity, dental, inpatient and other high-risk services",
        "features": ["is_maternity_benefit", "is_dental_benefit", "is_inpatient_benefit", "is_optical_benefit"],
        "color": "#16A085",
    },
    {
        "key": "Company Risk History",
        "description": "Employers with prior fraud or high variance",
        "features": ["company_fraud_incident_flag", "claim_amount_variance_company"],
        "color": "#F39C12",
    },
    {
        "key": "Multiple Detection Flags",
        "description": "Multiple models or detectors agree on an anomaly",
        "features": ["if_component", "kmeans_min_distance", "autoencoder_anomaly_score", "combined_anomaly_score"],
        "color": "#95A5A6",
    },
    {
        "key": "Pattern Anomalies",
        "description": "Unusual structural or temporal patterns",
        "features": ["claim_pattern_anomaly_score", "combined_anomaly_score"],
        "color": "#34495E",
    },
    {
        "key": "Suspicious Characteristics",
        "description": "Miscellaneous suspicious signals",
        "features": ["provider_claim_zscore", "service_charge_pct_dev_by_ailment"],
        "color": "#7F8C8D",
    },
    {
        "key": "Minor Irregularities",
        "description": "Low-risk or borderline anomalies",
        "features": ["average_claim_per_user", "service_charge_pct_dev_by_ailment"],
        "color": "#BDC3C7",
    },
]

# --- Helper: score a single feature value into [0,1] ---
def _score_value(feature_name: str, value) -> float:
    try:
        if pd.isna(value):
            return 0.0
        # binary flags
        if feature_name.endswith("flag") or feature_name.startswith("is"):
            return 1.0 if float(value) > 0 else 0.0
        # z-scores / deviations
        if "zscore" in feature_name or "dev" in feature_name:
            return min(abs(float(value)) / 3.0, 1.0)
        # ratios / percentages
        if "ratio" in feature_name or "pct" in feature_name:
            return min(abs(float(value)), 1.0)
        # anomaly scores already in [0,1]
        if "anomaly" in feature_name or "score" in feature_name:
            val = float(value)
            return val if 0 <= val <= 1 else min(max(val, 0.0), 1.0)
        # numeric magnitude fallback
        return min(abs(float(value)) / 10.0, 1.0)
    except Exception:
        return 0.0

# --- Core categorization function (kept behavior intact) ---
def categorize_claims(flagged_df: pd.DataFrame) -> pd.DataFrame:
    df = flagged_df.copy()
    df.columns = [str(c).strip().lower().replace(' ', '').replace('-', '') for c in df.columns]

    claimant_col = None
    for col in ("claimant_suddo", "claimant_name"):
        if col in df.columns:
            claimant_col = col
            break
    if claimant_col is not None and "combined_anomaly_score" in df.columns:
        idx = df.groupby(claimant_col)["combined_anomaly_score"].idxmax()
        df = df.loc[idx].reset_index(drop=True)

    df["primary_category"] = "Multiple Detection Flags"
    df["triggered_features"] = ""
    df["risk_score"] = 0.0
    df["review_priority"] = "Standard Review"
    df["category_reasons"] = ""

    for i, row in df.iterrows():
        cat_scores = {}
        cat_reasons = {}
        triggered = set()

        for cat in FEATURE_CATEGORIES:
            key = cat["key"]
            features = cat["features"]
            total = 0.0
            count = 0
            reasons = []

            for feat in features:
                if feat in row:
                    val = row[feat]
                    s = _score_value(feat, val)
                    total += s
                    count += 1
                    if s >= 0.3:
                        triggered.add(feat)
                        reasons.append(f"{feat}")

            score = (total / max(count, 1)) if count > 0 else 0.0
            cat_scores[key] = score
            cat_reasons[key] = reasons

        primary = max(cat_scores.items(), key=lambda kv: kv[1])[0]
        primary_score = cat_scores[primary]

        explanations = []
        if "combined_anomaly_score" in row and not pd.isna(row["combined_anomaly_score"]):
            explanations.append(f"Overall anomaly: {row['combined_anomaly_score']:.2f}")
        if cat_reasons.get(primary):
            explanations.extend(cat_reasons[primary])

        if primary_score >= 0.8:
            priority = "Immediate Review"
        elif primary_score >= 0.6:
            priority = "High Priority"
        elif primary_score >= 0.4:
            priority = "Standard Review"
        else:
            priority = "Low Priority"

        df.at[i, "primary_category"] = primary
        df.at[i, "triggered_features"] = ", ".join(sorted(list(triggered)))
        df.at[i, "risk_score"] = float(primary_score)
        df.at[i, "review_priority"] = priority
        df.at[i, "category_reasons"] = "; ".join(explanations) if explanations else ""

    return df

# --- Enhanced Display Components (modern visualizations) ---

def create_risk_gauge(score):
    """Modern gauge for average risk score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'valueformat': '.2f'},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Average Risk Score", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 1], 'tickformat': '.0%'},
            'bar': {'color': MINET_RED, 'thickness': 0.25},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 0.4], 'color': '#E8F8F0'},
                {'range': [0.4, 0.7], 'color': '#FFF4E6'},
                {'range': [0.7, 1.0], 'color': '#FDE7E6'},
            ],
        }
    ))
    fig.update_layout(height=260, margin=dict(l=8, r=8, t=36, b=8), paper_bgcolor=BACKGROUND_COLOR)
    return fig


def create_category_distribution_chart(df):
    """Treemap giving category counts and average risk per category"""
    counts = df.groupby('primary_category').agg(
        count=('primary_category', 'size'),
        avg_risk=('risk_score', 'mean')
    ).reset_index()

    # match colors from FEATURE_CATEGORIES
    color_map = {c['key']: c['color'] for c in FEATURE_CATEGORIES}
    counts['color'] = counts['primary_category'].map(color_map).fillna(MINET_BLUE)

    fig = px.treemap(
        counts,
        path=['primary_category'],
        values='count',
        color='avg_risk',
        color_continuous_scale=['#2ECC71', '#F1C40F', MINET_RED],
        hover_data={'count': True, 'avg_risk': ':.2f'},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=420, coloraxis_showscale=False, paper_bgcolor=BACKGROUND_COLOR)
    return fig


def create_priority_bar_chart(df):
    """Stacked bar breaking down counts by category per priority level"""
    grouped = (
        df.groupby(['review_priority', 'primary_category'])
        .size()
        .reset_index(name='count')
    )
    if grouped.empty:
        fig = go.Figure()
        fig.update_layout(height=250, paper_bgcolor=BACKGROUND_COLOR)
        return fig

    # pivot for stacked bars
    pivot = grouped.pivot(index='primary_category', columns='review_priority', values='count').fillna(0)
    priorities = ['Immediate Review', 'High Priority', 'Standard Review', 'Low Priority']
    # ensure order
    priorities = [p for p in priorities if p in pivot.columns] + [c for c in pivot.columns if c not in priorities]

    fig = go.Figure()
    for p in priorities:
        fig.add_trace(go.Bar(
            x=pivot.index,
            y=pivot[p],
            name=p,
            marker_line_width=0,
        ))

    fig.update_layout(barmode='stack', xaxis_title='Category', yaxis_title='Number of Claims', height=360, legend_title='Review Priority', margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor=BACKGROUND_COLOR)
    return fig


def create_risk_trend_chart(df):
    """Histogram with a boxplot marginal to show distribution and outliers"""
    if 'risk_score' not in df.columns:
        return None
    fig = px.histogram(df, x='risk_score', nbins=20, marginal='box', title='Risk Score Distribution', labels={'risk_score': 'Risk Score'})
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor=BACKGROUND_COLOR)
    fig.update_traces(marker_line_width=0)
    return fig


def create_top_triggered_features_chart(df, top_n=10):
    """Bar chart showing most common triggered features across claims"""
    if 'triggered_features' not in df.columns:
        return None
    all_feats = df['triggered_features'].dropna().astype(str).str.split(',')
    exploded = (pd.Series([f.strip() for sub in all_feats for f in sub if f.strip() != '']) )
    if exploded.empty:
        return None
    top = exploded.value_counts().nlargest(top_n).reset_index()
    top.columns = ['feature', 'count']
    fig = px.bar(top, x='count', y='feature', orientation='h', title='Top Triggered Features', labels={'count': 'Occurrences', 'feature': 'Feature'})
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=36, b=10), paper_bgcolor=BACKGROUND_COLOR)
    return fig


def display_metrics_summary(df):
    """Display key metrics in a clean card layout"""
    total = len(df)
    total_amount = df['total_payable'].sum() if 'total_payable' in df.columns else 0
    avg_risk = df['risk_score'].mean() if 'risk_score' in df.columns else 0
    high_priority = len(df[df['review_priority'].isin(['Immediate Review', 'High Priority'])])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class=\"metric-card\">
            <h4 style=\"margin:0; color: {MINET_BLUE}; font-size: 0.95rem;\">Flagged Claims</h4>
            <div style=\"font-size:1.8rem; font-weight:700; color: {MINET_BLUE};\">{total}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class=\"metric-card\">
            <h4 style=\"margin:0; color: {MINET_BLUE}; font-size: 0.95rem;\">Total Amount</h4>
            <div style=\"font-size:1.25rem; font-weight:700; color: {MINET_BLUE};\">KES {total_amount:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        risk_class = "risk-high" if avg_risk > 0.7 else "risk-medium" if avg_risk > 0.4 else "risk-low"
        st.markdown(f"""
        <div class=\"metric-card\">
            <h4 style=\"margin:0; color: {MINET_BLUE}; font-size: 0.95rem;\">Average Risk</h4>
            <div class=\"{risk_class}\" style=\"font-size:1.6rem; font-weight:700;\">{avg_risk:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class=\"metric-card\">
            <h4 style=\"margin:0; color: {MINET_BLUE}; font-size: 0.95rem;\">High Priority</h4>
            <div style=\"font-size:1.6rem; font-weight:700; color: {MINET_RED};\">{high_priority}</div>
        </div>
        """, unsafe_allow_html=True)


def display_category_cards(df):
    """Display interactive category cards with concise detail panels"""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader('Fraud Pattern Analysis by Category')

    for category in FEATURE_CATEGORIES:
        category_data = df[df['primary_category'] == category['key']]
        if not category_data.empty:
            with st.expander(f"{category['key']} — {len(category_data)} claims", expanded=False):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**Description:** {category['description']}")
                    st.write(f"**Average Risk Score:** {category_data['risk_score'].mean():.2f}")

                    triggered_features = set()
                    for features in category_data['triggered_features']:
                        if features:
                            triggered_features.update([f.strip() for f in features.split(',') if f.strip() != ''])

                    if triggered_features:
                        st.write('**Common Triggered Features:**')
                        feature_cols = st.columns(3)
                        for idx, feature in enumerate(list(triggered_features)[:9]):
                            with feature_cols[idx % 3]:
                                st.markdown(f'<div class="feature-pill">{feature}</div>', unsafe_allow_html=True)

                with col2:
                    priority_counts = category_data['review_priority'].value_counts()
                    if not priority_counts.empty:
                        pie = px.pie(values=priority_counts.values, names=priority_counts.index, hole=0.45)
                        pie.update_traces(textposition='inside', textinfo='percent+label')
                        pie.update_layout(height=230, margin=dict(l=6, r=6, t=6, b=6), paper_bgcolor=BACKGROUND_COLOR)
                        st.plotly_chart(pie, use_container_width=True)

                st.write('**Sample Claims:**')
                display_cols = ["visit_id", "provider", "total_payable", "risk_score", "review_priority"]
                available_cols = [col for col in display_cols if col in category_data.columns]
                if available_cols:
                    st.dataframe(category_data[available_cols].head(6), use_container_width=True)


def display_claims_table(df):
    """Display detailed claims table with filtering options (functionality preserved)"""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader('Detailed Claims Analysis')

    col1, col2, col3 = st.columns(3)
    with col1:
        priority_filter = st.multiselect(
            'Filter by Priority',
            options=df['review_priority'].unique(),
            default=df['review_priority'].unique()
        )
    with col2:
        category_filter = st.multiselect(
            'Filter by Category',
            options=df['primary_category'].unique(),
            default=df['primary_category'].unique()
        )
    with col3:
        risk_range = st.slider(
            'Risk Score Range',
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.01
        )

    filtered_df = df[
        (df['review_priority'].isin(priority_filter)) &
        (df['primary_category'].isin(category_filter)) &
        (df['risk_score'] >= risk_range[0]) &
        (df['risk_score'] <= risk_range[1])
    ]

    st.write(f"Showing {len(filtered_df)} of {len(df)} claims")

    display_columns = [
        'visit_id', 'provider', 'total_payable', 'primary_category',
        'risk_score', 'review_priority', 'triggered_features'
    ]
    available_columns = [col for col in display_columns if col in filtered_df.columns]

    if available_columns:
        st.dataframe(filtered_df[available_columns], use_container_width=True, height=420)

# --- Main Categorization Tab Function (keeps behavior unchanged) ---

def categorization_tab():
    st.header('Fraud Pattern Categorization')

    processed = st.session_state.get('processed_data')
    if processed is None:
        st.warning('No processed data available. Run detection first.')
        return

    flagged = processed[processed.get('needs_review') == 1].copy()
    if flagged.empty:
        st.info('No flagged claims to categorize.')
        return

    with st.spinner('Analyzing fraud patterns...'):
        categorized = categorize_claims(flagged)

    st.success(f'Categorized {len(categorized)} claims')

    display_metrics_summary(categorized)

    st.subheader('Risk Analysis Overview')
    viz_col1, viz_col2 = st.columns([1, 1])

    with viz_col1:
        st.plotly_chart(create_risk_gauge(categorized['risk_score'].mean()), use_container_width=True)

    with viz_col2:
        top_feats = create_top_triggered_features_chart(categorized)
        if top_feats:
            st.plotly_chart(top_feats, use_container_width=True)

    # Larger breakdown by priority and category
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader('Priority by Category')
    st.plotly_chart(create_priority_bar_chart(categorized), use_container_width=True)


    display_category_cards(categorized)
    display_claims_table(categorized)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader('Data Export')
    col1, col2 = st.columns([3, 1])

    with col1:
        st.info('Export the categorized claims data for further analysis or reporting.')

    with col2:
        csv = categorized.to_csv(index=False)
        st.download_button(
            label='Download CSV',
            data=csv,
            file_name='categorized_fraud_claims.csv',
            mime='text/csv'
        )

# --- Integration function for your main app ---

def add_categorization_tab():
    categorization_tab()
