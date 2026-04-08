import pandas as pd
import datetime as dt
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    """Clean raw data: Obtain return value and return or abnormal orders"""
    df = df.dropna(subset=['CustomerID']) 
    df = df[df['Quantity'] > 0] 
    df = df[df['UnitPrice'] > 0] 
    return df

def calculate_rfm(df):
    """Transform the original transaction records into an RFM feature matrix."""
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    
   
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',                                  
        'TotalPrice': 'sum'                                      
    }).reset_index()
    
    rfm.rename(columns={'InvoiceDate': 'Recency', 
                        'InvoiceNo': 'Frequency', 
                        'TotalPrice': 'Monetary'}, inplace=True)
    return rfm

def scale_rfm_data(rfm_df):
    """Feature scalin"""
    
    rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']]
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)
    
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'], index=rfm_df.index)
    
    return rfm_scaled_df, scaler
