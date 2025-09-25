import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def categorization_tab():
    st.header(" Fraud Reason Categorization")
    
    # Check if we have processed data
    if st.session_state.get('processed_data') is None:
        st.warning("Please process data in the Detection tab first.")
        return
        
    processed_data = st.session_state.processed_data
    
    # Only show flagged claims
    if 'needs_review' not in processed_data.columns:
        st.error("No 'needs_review' column found. Please run fraud detection first.")
        return
        
    flagged_data = processed_data[processed_data['needs_review'] == 1].copy()
    
    if len(flagged_data) == 0:
        st.info("No claims have been flagged for review. Adjust the threshold or process more data.")
        return
    
    st.success(f"**{len(flagged_data)} claims** have been flagged for review")
    
    # Categorize the fraud reasons with better grouping
    categorized_data = _categorize_fraud_reasons(flagged_data)
    
    # Display overview
    _display_overview(categorized_data)
    
    # Detailed breakdown
    _display_detailed_breakdown(categorized_data)
    
    # Provider analysis
    _display_provider_analysis(categorized_data)
    
    # Diagnosis pattern analysis
    _display_diagnosis_analysis(categorized_data)

def _categorize_fraud_reasons(flagged_data):
    """
    Categorize flagged claims into specific fraud reasons with better grouping
    """
    categorized = flagged_data.copy()
    
    # Initialize reason columns
    categorized['reasons'] = "Analyzing claim patterns..."
    categorized['primary_reason'] = "Under review"
    categorized['reason_group'] = "Needs investigation"
    categorized['reason_details'] = ""
    
    # Define thresholds
    high_claim_threshold = 0.8  # 80% of cover limit
    very_high_claim_threshold = 1.0  # 100% of cover limit
    extreme_claim_threshold = 2.0  # 200% of cover limit
    short_gap_threshold = 7     # 7 days since last visit
    high_freq_threshold = 5     # 5+ visits in period
    
    # Categorization logic
    for idx, row in categorized.iterrows():
        reasons = []
        details = []
        
        # 1. Claim amount analysis
        if ('TOTAL_PAYABLE' in row and 'COVER_LIMIT' in row and 
            pd.notna(row['TOTAL_PAYABLE']) and pd.notna(row['COVER_LIMIT']) and
            row['COVER_LIMIT'] > 0):
            
            claim_ratio = row['TOTAL_PAYABLE'] / row['COVER_LIMIT']
            
            if claim_ratio >= extreme_claim_threshold:
                reasons.append("Extreme claim amount")
                details.append(f"Claim amount of KES {row['TOTAL_PAYABLE']:,.0f} exceeds coverage limit by {claim_ratio:.0f} times")
            elif claim_ratio >= very_high_claim_threshold:
                reasons.append("Full limit utilization")
                details.append(f"Claim uses {claim_ratio:.0%} of the KES {row['COVER_LIMIT']:,.0f} coverage limit")
            elif claim_ratio >= high_claim_threshold:
                reasons.append("High claim amount")
                details.append(f"Claim uses {claim_ratio:.0%} of coverage limit")
        
        # 2. Visit pattern analysis
        if ('DAYS_SINCE_LAST_VISIT' in row and pd.notna(row['DAYS_SINCE_LAST_VISIT'])):
            days_gap = row['DAYS_SINCE_LAST_VISIT']
            if days_gap <= short_gap_threshold:
                reasons.append("Frequent visits")
                details.append(f"Only {days_gap} days since last visit")
        
        # 3. High frequency of visits
        if ('FREQUENCY_OF_VISIT' in row and pd.notna(row['FREQUENCY_OF_VISIT'])):
            freq = row['FREQUENCY_OF_VISIT']
            if freq >= high_freq_threshold:
                reasons.append("High visit frequency")
                details.append(f"{freq} visits in a short period")
        
        # 4. High-risk provider
        if ('hospital_risk_score' in row and pd.notna(row['hospital_risk_score'])):
            risk_score = row['hospital_risk_score']
            if risk_score > 0:
                reasons.append("High-risk provider")
                details.append(f"Provider has {risk_score} prior risk indicators")
        
        # 5. Unusual diagnosis patterns
        if ('diagnosis_group' in row and pd.notna(row['diagnosis_group'])):
            diagnosis = row['diagnosis_group']
            if diagnosis in ['Maternity_Related', 'Dental_Related']:
                reasons.append("High-risk diagnosis")
                details.append(f"Diagnosis: {diagnosis.replace('_', ' ')}")
        
        # 6. Demographic anomalies
        if ('AGE(CLAIMANT)' in row and 'RELATIONSHIP' in row and 
            pd.notna(row['AGE(CLAIMANT)']) and pd.notna(row['RELATIONSHIP'])):
            
            relationship = str(row['RELATIONSHIP']).lower()
            age = row['AGE(CLAIMANT)']
            
            if 'child' in relationship and age > 25:
                reasons.append("Age discrepancy")
                details.append(f"Claimant age {age} doesn't match 'child' relationship")
            elif 'spouse' in relationship and age < 18:
                reasons.append("Age discrepancy")
                details.append(f"Claimant age {age} doesn't match 'spouse' relationship")
        
        # 7. If no specific reasons found, provide general analysis
        if not reasons:
            # Create meaningful reasons based on available data
            if 'combined_anomaly_score' in row and pd.notna(row['combined_anomaly_score']):
                score = row['combined_anomaly_score']
                if score > 0.9:
                    reasons.append("High anomaly score")
                    details.append(f"Very high anomaly detection score of {score:.3f}")
                else:
                    reasons.append("Multiple minor anomalies")
                    details.append("Combination of several unusual patterns detected")
            else:
                reasons.append("Unusual claim patterns")
                details.append("Multiple factors contribute to this flag")
        
        # Group similar reasons
        reason_group = "Other anomalies"
        if any(r in reasons for r in ["Extreme claim amount", "Full limit utilization", "High claim amount"]):
            reason_group = "Financial anomalies"
        elif any(r in reasons for r in ["Frequent visits", "High visit frequency"]):
            reason_group = "Utilization patterns"
        elif any(r in reasons for r in ["High-risk provider"]):
            reason_group = "Provider risk"
        elif any(r in reasons for r in ["High-risk diagnosis"]):
            reason_group = "Medical patterns"
        elif any(r in reasons for r in ["Age discrepancy"]):
            reason_group = "Demographic issues"
        
        # Store results
        categorized.at[idx, 'reasons'] = "; ".join(reasons)
        categorized.at[idx, 'primary_reason'] = reasons[0] if reasons else "Pattern analysis"
        categorized.at[idx, 'reason_group'] = reason_group
        categorized.at[idx, 'reason_details'] = ". ".join(details) if details else "Detailed analysis pending"
    
    return categorized

def _display_overview(categorized_data):
    st.subheader(" Fraud Reason Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Reason groups count
        reason_group_counts = categorized_data['reason_group'].value_counts()
        fig = px.pie(
            values=reason_group_counts.values,
            names=reason_group_counts.index,
            title="Fraud Reason Categories"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Primary reasons (top 5)
        primary_reason_counts = categorized_data['primary_reason'].value_counts().head(5)
        fig = px.bar(
            x=primary_reason_counts.values,
            y=primary_reason_counts.index,
            orientation='h',
            title="Top 5 Primary Reasons",
            labels={'x': 'Count', 'y': 'Reason'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Financial impact by reason group
        if 'TOTAL_PAYABLE' in categorized_data.columns:
            financial_data = []
            for group in categorized_data['reason_group'].unique():
                group_data = categorized_data[categorized_data['reason_group'] == group]
                total_amount = group_data['TOTAL_PAYABLE'].sum()
                avg_amount = group_data['TOTAL_PAYABLE'].mean()
                financial_data.append({
                    'Reason Group': group,
                    'Total Amount': total_amount,
                    'Average Amount': avg_amount,
                    'Count': len(group_data)
                })
            
            financial_df = pd.DataFrame(financial_data)
            if not financial_df.empty:
                fig = px.bar(
                    financial_df,
                    x='Reason Group',
                    y='Average Amount',
                    title="Avg Claim Amount by Category",
                    hover_data=['Total Amount', 'Count']
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

def _display_detailed_breakdown(categorized_data):
    st.subheader(" Detailed Breakdown by Category")
    
    # Group by reason group
    reason_groups = categorized_data.groupby('reason_group')
    
    for group_name, group in reason_groups:
        with st.expander(f"{group_name} ({len(group)} claims)"):
            if 'TOTAL_PAYABLE' in group.columns:
                st.write(f"**Total Amount at Risk:** KES {group['TOTAL_PAYABLE'].sum():,.2f}")
                st.write(f"**Average Amount:** KES {group['TOTAL_PAYABLE'].mean():,.2f}")
            
            # Show reason distribution within this group
            st.write("**Reason Distribution:**")
            reason_counts = group['primary_reason'].value_counts()
            for reason, count in reason_counts.items():
                st.write(f"- {reason}: {count} claims")
            
            # Show representative examples
            st.write("**Example Claims:**")
            display_cols = ['VISIT_ID', 'PROVIDER', 'TOTAL_PAYABLE', 'COVER_LIMIT', 'primary_reason']
            
            # Add available columns
            for col in ['DAYS_SINCE_LAST_VISIT', 'FREQUENCY_OF_VISIT']:
                if col in group.columns:
                    display_cols.append(col)
            
            display_cols = [col for col in display_cols if col in group.columns]
            
            if len(display_cols) > 0:
                sample_data = group[display_cols].head(3).copy()
                
                # Format financial values
                if 'TOTAL_PAYABLE' in sample_data.columns:
                    sample_data['TOTAL_PAYABLE'] = sample_data['TOTAL_PAYABLE'].apply(lambda x: f"KES {x:,.0f}")
                if 'COVER_LIMIT' in sample_data.columns:
                    sample_data['COVER_LIMIT'] = sample_data['COVER_LIMIT'].apply(lambda x: f"KES {x:,.0f}" if pd.notna(x) else "N/A")
                
                st.dataframe(sample_data, use_container_width=True)
            
            # Show example reason details
            st.write("**Example Reason Details:**")
            example_details = group['reason_details'].dropna().head(2).tolist()
            for detail in example_details:
                st.write(f"- {detail}")

def _display_provider_analysis(categorized_data):
    if 'PROVIDER' not in categorized_data.columns:
        return
        
    st.subheader(" Provider Analysis")
    
    provider_summary = categorized_data.groupby('PROVIDER').agg({
        'VISIT_ID': 'count',
        'TOTAL_PAYABLE': 'sum',
        'reason_group': lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown'
    }).rename(columns={
        'VISIT_ID': 'Flagged_Claims',
        'TOTAL_PAYABLE': 'Total_Amount',
        'reason_group': 'Most_Common_Category'
    }).sort_values('Flagged_Claims', ascending=False).head(10)
    
    st.write("**Providers with Most Flagged Claims**")
    st.dataframe(provider_summary, use_container_width=True)

def _display_diagnosis_analysis(categorized_data):
    diagnosis_cols = ['ailments', 'diagnosis_group', 'benefit', 'broad_benefit']
    available_diag_cols = [col for col in diagnosis_cols if col in categorized_data.columns]
    
    if not available_diag_cols:
        return
        
    st.subheader("🏥 Medical Pattern Analysis")
    
    # Select which diagnosis-related column to analyze
    diag_col = st.selectbox("Select medical field to analyze", options=available_diag_cols)
    
    if diag_col:
        # Show distribution of reasons by diagnosis
        diag_reason = pd.crosstab(categorized_data[diag_col], categorized_data['reason_group']).head(10)
        
        if not diag_reason.empty:
            fig = px.bar(
                diag_reason,
                title=f"Fraud Reasons by {diag_col}",
                labels={'value': 'Count', diag_col: diag_col}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

# Add this to your main app
def add_categorization_tab():
    categorization_tab()