import pandas as pd
import datetime as dt
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    """清洗原始数据：去除缺失值和退货/异常订单"""
    df = df.dropna(subset=['CustomerID']) 
    df = df[df['Quantity'] > 0] 
    df = df[df['UnitPrice'] > 0] 
    return df

def calculate_rfm(df):
    """将原始交易记录转化为 RFM 特征矩阵"""
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
    """特征缩放 (至关重要！因为聚类对数值大小敏感)"""
    
    rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']]
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)
    
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'], index=rfm_df.index)
    
    return rfm_scaled_df, scaler
