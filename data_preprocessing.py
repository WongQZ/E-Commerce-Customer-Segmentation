import pandas as pd
import datetime as dt
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    """清洗原始数据：去除缺失值和退货/异常订单"""
    # 假设我们使用的是类似 UCI Online Retail 的数据集
    # 必须包含: CustomerID, InvoiceDate, InvoiceNo, Quantity, UnitPrice
    df = df.dropna(subset=['CustomerID']) # 去除没有客户ID的行
    df = df[df['Quantity'] > 0] # 去除退货订单 (数量为负)
    df = df[df['UnitPrice'] > 0] # 去除免费或异常价格
    return df

def calculate_rfm(df):
    """将原始交易记录转化为 RFM 特征矩阵"""
    # 计算每笔订单的总价
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # 确保日期格式正确
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # 设定分析基准日为数据集中最后一天 + 1天
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    
    # 按照 CustomerID 聚合计算 RFM
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency: 距离上次购买的天数
        'InvoiceNo': 'nunique',                                  # Frequency: 购买次数
        'TotalPrice': 'sum'                                      # Monetary: 总消费金额
    }).reset_index()
    
    # 重命名列
    rfm.rename(columns={'InvoiceDate': 'Recency', 
                        'InvoiceNo': 'Frequency', 
                        'TotalPrice': 'Monetary'}, inplace=True)
    return rfm

def scale_rfm_data(rfm_df):
    """特征缩放 (至关重要！因为聚类对数值大小敏感)"""
    # 提取需要的列 (去掉 CustomerID)
    rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']]
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)
    
    # 转回 DataFrame 方便后续操作
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'], index=rfm_df.index)
    
    return rfm_scaled_df, scaler
