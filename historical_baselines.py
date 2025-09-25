import pandas as pd
import numpy as np
import sqlite3

class HistoricalBaselines:
    def __init__(self, db_path="historical_data.db"):
        self.db_path = db_path
    
    def load_provider_baselines(self):
        """Load historical provider medians"""
        conn = sqlite3.connect(self.db_path)
        provider_baselines = pd.read_sql(
            "SELECT provider, service, median_cost, sample_size FROM provider_baselines", 
            conn
        )
        conn.close()
        return provider_baselines
    
    def calculate_typical_cost(self, row, provider_baselines):
        """Calculate typical cost using historical data"""
        provider = row.get('PROVIDER')
        service = row.get('BENEFIT')
        
        if provider and service:
            baseline = provider_baselines[
                (provider_baselines['provider'] == provider) & 
                (provider_baselines['service'] == service)
            ]
            if not baseline.empty:
                return baseline['median_cost'].iloc[0]
        
        # Fallback to global median if no provider-service match
        return provider_baselines['median_cost'].median()