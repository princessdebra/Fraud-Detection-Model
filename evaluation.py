# evaluation.py - Enhanced for Unlabeled Data & Operational Calibration

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

class EvaluationSuite:
    def __init__(self):
        # Initialize database connection if available
        self.db = self._initialize_db()
        
    def _initialize_db(self):
        """Initialize database connection if available"""
        try:
            from database import FraudDetectionDB
            return FraudDetectionDB()
        except (ImportError, AttributeError):
            st.warning("Database module not available. Some features will be limited.")
            return None
        
    def render(self):
        st.header(" Operational Calibration & Performance Dashboard")
        
        # Check if we have scored data
        if st.session_state.get('scored_data') is None:
            st.warning("Please process data in the Detection tab first.")
            return
            
        scores = st.session_state.scored_data['combined_anomaly_score']
        current_threshold = st.session_state.get('current_threshold', 
                        st.session_state.scorer.info()['threshold'] if hasattr(st.session_state, 'scorer') else 0.5)
        
        # Create tabs for different evaluation sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "Threshold Calibration", 
            "Operational Metrics", 
            "Scenario Planning", 
            "Feedback & Learning"
        ])
        
        with tab1:
            self._threshold_calibration(scores, current_threshold)
            
        with tab2:
            self._operational_metrics(scores, current_threshold)
            
        with tab3:
            self._scenario_planning(scores)
            
        with tab4:
            self._feedback_learning()
    
    def _threshold_calibration(self, scores, current_threshold):
        st.subheader("Capacity-Based Threshold Setting")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Widget: Let the manager set the daily review capacity
            daily_capacity = st.number_input(
                "Enter your team's daily review capacity (number of claims)",
                min_value=1,
                max_value=1000,
                value=50,
                help="Based on your team size and available time"
            )
            
            # Calculate the threshold that would flag exactly that many claims
            total_claims = len(scores)
            if daily_capacity >= total_claims:
                capacity_threshold = scores.min()
            else:
                capacity_percent = 100 - (daily_capacity / total_claims * 100)
                capacity_threshold = np.percentile(scores, capacity_percent)
            
            st.metric("Recommended Threshold", f"{capacity_threshold:.3f}")
            st.metric("Claims to Review", f"{daily_capacity} / {total_claims}")
            st.write(f"This will flag the top **{(daily_capacity / total_claims * 100):.1f}%** of highest-risk claims.")
            
            # Apply threshold button
            if st.button("Apply This Threshold", type="primary"):
                st.session_state.current_threshold = capacity_threshold
                if hasattr(st.session_state, 'scorer'):
                    st.session_state.scorer.update_threshold(capacity_threshold)
                st.success("Threshold updated! Return to Detection tab to see changes.")
        
        with col2:
            # Score distribution visualization
            fig, ax = plt.subplots(figsize=(8, 4))
            n, bins, patches = ax.hist(scores, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
            
            # Color the flagged area
            for i in range(len(patches)):
                if bins[i] >= current_threshold:
                    patches[i].set_facecolor('red')
                    patches[i].set_alpha(0.7)
            
            ax.axvline(current_threshold, color='red', linestyle='--', 
                      label=f'Current Threshold ({current_threshold:.3f})')
            ax.axvline(capacity_threshold, color='green', linestyle='--', 
                      label=f'Capacity Threshold ({capacity_threshold:.3f})')
            
            ax.set_xlabel('Anomaly Score')
            ax.set_ylabel('Number of Claims')
            ax.set_title('Distribution of Anomaly Scores')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
    
    def _operational_metrics(self, scores, current_threshold):
        st.subheader("Operational Performance Metrics")
        
        # Calculate basic metrics
        total_claims = len(scores)
        flagged_claims = (scores >= current_threshold).sum()
        flag_rate = flagged_claims / total_claims
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Claims", total_claims)
        with col2:
            st.metric("Flagged Claims", f"{flagged_claims} ({flag_rate:.1%})")
        with col3:
            # Get feedback stats if available
            if self.db:
                try:
                    feedback_stats = self.db.get_feedback_stats()
                    if not feedback_stats.empty:
                        reviewed = len(feedback_stats)
                        st.metric("Claims Reviewed", reviewed)
                except AttributeError:
                    st.metric("Claims Reviewed", "N/A")
            else:
                st.metric("Claims Reviewed", "N/A")
        
        # Show time-based metrics if we have date information
        processed_data = st.session_state.get('processed_data')
        if processed_data is not None and 'AILMENT_DATE' in processed_data.columns:
            st.subheader("Trend Analysis")
            
            # Try to parse dates
            try:
                processed_data['date'] = pd.to_datetime(processed_data['AILMENT_DATE'], errors='coerce')
                time_based = processed_data.groupby(pd.Grouper(key='date', freq='W')).agg({
                    'combined_anomaly_score': ['count', 'mean']
                }).dropna()
                
                time_based.columns = ['claims_count', 'avg_score']
                time_based['flagged'] = processed_data.groupby(
                    pd.Grouper(key='date', freq='W')
                )['needs_review'].sum().dropna()
                
                time_based['flag_rate'] = time_based['flagged'] / time_based['claims_count']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time_based.index, 
                    y=time_based['flag_rate'],
                    mode='lines+markers',
                    name='Flag Rate',
                    yaxis='y1'
                ))
                fig.add_trace(go.Bar(
                    x=time_based.index,
                    y=time_based['claims_count'],
                    name='Claim Volume',
                    yaxis='y2',
                    opacity=0.3
                ))
                
                fig.update_layout(
                    title='Weekly Flag Rate & Claim Volume',
                    xaxis=dict(title='Week'),
                    yaxis=dict(title='Flag Rate', rangemode='tozero'),
                    yaxis2=dict(title='Claim Volume', overlaying='y', side='right'),
                    legend=dict(x=0, y=1.1),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.info("Could not generate time trends. Date formatting may be inconsistent.")
    
    def _scenario_planning(self, scores):
        st.subheader("Scenario Planning & Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            avg_fraud_value = st.number_input(
                "Estimated average value of a fraudulent claim (KES)", 
                1000, 500000, 25000, 1000,
                help="Based on historical experience with fraudulent claims"
            )
            
        with col2:
            cost_per_review = st.number_input(
                "Estimated cost to review one claim (KES)", 
                50, 5000, 500, 50,
                help="Includes investigator time and resources"
            )
        
        # Model different threshold scenarios
        thresholds = np.percentile(scores, [99, 97, 95, 90, 85, 80, 75, 70, 65, 60])
        scenario_data = []
        
        for t in thresholds:
            num_flagged = (scores >= t).sum()
            flag_rate = num_flagged / len(scores)
            
            # Make reasonable assumptions for planning
            assumed_precision = max(0.05, min(0.5, 0.4 - (flag_rate * 0.3)))  # Higher flag rate → lower precision
            estimated_caught = num_flagged * assumed_precision
            estimated_savings = estimated_caught * avg_fraud_value
            review_cost = num_flagged * cost_per_review
            net_benefit = estimated_savings - review_cost
            
            scenario_data.append({
                'Threshold': t,
                'Flagged Claims': num_flagged,
                'Flag Rate': flag_rate,
                'Assumed Precision': assumed_precision,
                'Estimated Caught': estimated_caught,
                'Estimated Savings (KES)': estimated_savings,
                'Review Cost (KES)': review_cost,
                'Net Benefit (KES)': net_benefit
            })
        
        scenario_df = pd.DataFrame(scenario_data)
        
        # Create visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=scenario_df['Threshold'], 
            y=scenario_df['Net Benefit (KES)'],
            mode='lines+markers',
            name='Net Benefit',
            line=dict(color='green', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=scenario_df['Threshold'], 
            y=scenario_df['Review Cost (KES)'],
            mode='lines',
            name='Review Cost',
            line=dict(color='red', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=scenario_df['Threshold'], 
            y=scenario_df['Estimated Savings (KES)'],
            mode='lines',
            name='Estimated Savings',
            line=dict(color='blue', dash='dash')
        ))
        
        fig.update_layout(
            title='Financial Impact of Different Thresholds',
            xaxis=dict(title='Threshold Value'),
            yaxis=dict(title='KES'),
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show optimal threshold
        optimal_idx = scenario_df['Net Benefit (KES)'].idxmax()
        optimal_row = scenario_df.iloc[optimal_idx]
        
        st.success(
            f"**Optimal Threshold Recommendation:** {optimal_row['Threshold']:.3f}\n\n"
            f"- Flags {optimal_row['Flagged Claims']} claims ({optimal_row['Flag Rate']:.1%})\n"
            f"- Estimated net benefit: KES {optimal_row['Net Benefit (KES)']:,.0f}\n"
            f"- Review cost: KES {optimal_row['Review Cost (KES)']:,.0f}\n"
            f"- Estimated savings: KES {optimal_row['Estimated Savings (KES)']:,.0f}"
        )
    
    def _feedback_learning(self):
        st.subheader("Claim Review & Feedback System")
        
        # Check if we have processed data
        if st.session_state.get('processed_data') is None:
            st.info("Process data in the Detection tab to enable feedback.")
            return
            
        processed_data = st.session_state.processed_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Claim selection for review
            visit_ids = processed_data['VISIT_ID'].tolist() if 'VISIT_ID' in processed_data.columns else []
            if not visit_ids:
                st.info("No VISIT_ID column found in the data.")
                return
                
            selected_visit = st.selectbox("Select Visit ID to review", options=visit_ids[:100])
            
            if selected_visit:
                claim_data = processed_data[processed_data['VISIT_ID'] == selected_visit].iloc[0]
                
                st.write("**Claim Details:**")
                st.write(f"Visit ID: {selected_visit}")
                if 'PROVIDER' in claim_data:
                    st.write(f"Provider: {claim_data['PROVIDER']}")
                if 'TOTAL_PAYABLE' in claim_data:
                    st.write(f"Amount: KES {claim_data['TOTAL_PAYABLE']:,.2f}")
                if 'combined_anomaly_score' in claim_data:
                    st.write(f"Anomaly Score: {claim_data['combined_anomaly_score']:.3f}")
                if 'needs_review' in claim_data:
                    st.write(f"Flagged: {'Yes' if claim_data.get('needs_review', False) else 'No'}")
        
        with col2:
            # Feedback form
            st.write("**Review Outcome:**")
            verdict = st.radio("Verdict", options=[
                "Confirmed Fraud", 
                "False Alarm", 
                "Insufficient Information",
                "Not Reviewed"
            ], index=3)
            
            comments = st.text_area("Review Comments", 
                                  placeholder="Add details about your investigation...")
            
            if st.button("Submit Review", type="primary"):
                if self.db:
                    try:
                        # Save to database
                        self.db.add_feedback(
                            visit_id=selected_visit,
                            verdict=verdict,
                            comments=comments,
                            reviewer=user["email"]  # Would normally come from authentication
                        )
                        st.success("Review submitted! This data will help improve the system.")
                    except AttributeError:
                        st.error("Database method not available. Please check your database implementation.")
                else:
                    st.info("Database not available. Feedback would be saved in a production environment.")
        
        # Show feedback statistics if available
        if self.db:
            try:
                feedback_stats = self.db.get_feedback_stats()
                if not feedback_stats.empty:
                    st.subheader("Review Statistics")
                    
                    col1, col2, col3 = st.columns(3)
                    total_reviews = len(feedback_stats)
                    
                    with col1:
                        confirmed = len(feedback_stats[feedback_stats['verdict'] == 'Confirmed Fraud'])
                        st.metric("Confirmed Fraud", confirmed)
                    
                    with col2:
                        false_alarms = len(feedback_stats[feedback_stats['verdict'] == 'False Alarm'])
                        st.metric("False Alarms", false_alarms)
                    
                    with col3:
                        if confirmed + false_alarms > 0:
                            precision = confirmed / (confirmed + false_alarms)
                            st.metric("Current Precision", f"{precision:.1%}")
                        else:
                            st.metric("Current Precision", "N/A")
                    
                    # Visualize verdict distribution
                    verdict_counts = feedback_stats['verdict'].value_counts()
                    fig = px.pie(
                        values=verdict_counts.values, 
                        names=verdict_counts.index,
                        title="Review Verdict Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except AttributeError:
                st.info("Feedback statistics not available. Database methods may need implementation.")

# Main function to run the evaluation tab
def evaluation_tab():
    evaluator = EvaluationSuite()
    evaluator.render()

# For direct script execution
if __name__ == "__main__":
    evaluation_tab()