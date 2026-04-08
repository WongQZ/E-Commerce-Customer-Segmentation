import streamlit as st
import pandas as pd

# 导入我们刚刚写好的独立模块
import data_preprocessing as dp
import kmeans_clustering as km
import dbscan_clustering as db
import meanshift_clustering as ms

st.set_page_config(page_title="E-Commerce Customer Segmentation", page_icon="🛒", layout="wide")

st.title("🛒 E-Commerce Customer Segmentation System")
st.markdown("---")

st.sidebar.title("Navigation")
menu = ["1. Data Overview & Preprocessing", "2. K-Means Clustering (Member A)", "3. DBSCAN Clustering (Member B)", "4. MeanShift Clustering (Member C)"]
choice = st.sidebar.radio("Go to", menu)

st.sidebar.header("📁 Data Source")
data_option = st.sidebar.radio("Select your dataset:", ["Upload Your Own Dataset"])

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file, encoding='unicode_escape')
    else:
        return pd.read_excel(uploaded_file)

# 初始化 session_state 来保存处理后的数据，避免在切换页面时数据丢失
if 'rfm_df' not in st.session_state:
    st.session_state.rfm_df = None
if 'rfm_scaled_df' not in st.session_state:
    st.session_state.rfm_scaled_df = None

df = None
uploaded_file = st.sidebar.file_uploader("Upload E-commerce data (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
    
    # 自动进行数据清洗和 RFM 特征提取，存入 session_state
    cleaned_df = dp.clean_data(df)
    st.session_state.rfm_df = dp.calculate_rfm(cleaned_df)
    st.session_state.rfm_scaled_df, scaler = dp.scale_rfm_data(st.session_state.rfm_df)
else:
    st.sidebar.info("👈 Please upload a dataset to begin.")

# --- 页面路由逻辑 ---
if choice == "1. Data Overview & Preprocessing":
    st.header("📊 Data Overview")
    if df is not None:
        st.write(f"Raw Dataset Shape: **{df.shape[0]} rows**, **{df.shape[1]} columns**")
        st.dataframe(df.head())
        
        st.subheader("🛠️ RFM Feature Extraction & Scaling")
        st.success(f"Successfully extracted RFM features for **{st.session_state.rfm_df.shape[0]}** unique customers.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Calculated RFM Data (Before Scaling):**")
            st.dataframe(st.session_state.rfm_df.head())
        with col2:
            st.write("**Scaled RFM Data (For Machine Learning):**")
            st.dataframe(st.session_state.rfm_scaled_df.head())
    else:
        st.warning("Please upload a dataset from the sidebar to start.")

elif choice == "2. K-Means Clustering (Member A)":
    if st.session_state.rfm_scaled_df is not None:
        km.run_kmeans_app(st.session_state.rfm_df.copy(), st.session_state.rfm_scaled_df.copy())
    else:
        st.warning("Please upload data in the 'Data Overview' tab first.")

elif choice == "3. DBSCAN Clustering (Member B)":
    if st.session_state.rfm_scaled_df is not None:
        db.run_dbscan_app(st.session_state.rfm_df.copy(), st.session_state.rfm_scaled_df.copy())
    else:
        st.warning("Please upload data in the 'Data Overview' tab first.")

elif choice == "4. MeanShift Clustering (Member C)":
    if st.session_state.rfm_scaled_df is not None:
        ms.run_meanshift_app(st.session_state.rfm_df.copy(), st.session_state.rfm_scaled_df.copy())
    else:
        st.warning("Please upload data in the 'Data Overview' tab first.")
