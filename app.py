import streamlit as st
import pandas as pd

# 设置页面配置 (这会让你们的 Demo 看起来更专业)
st.set_page_config(page_title="E-Commerce Customer Segmentation", page_icon="🛒", layout="wide")

st.title("🛒 E-Commerce Customer Segmentation System")
st.markdown("---")

# 侧边栏：导航菜单
st.sidebar.title("Navigation")
menu = ["1. Data Overview & Preprocessing", "2. K-Means Clustering (Member A)", "3. DBSCAN Clustering (Member B)", "4. MeanShift Clustering (Member C)"]
choice = st.sidebar.radio("Go to", menu)

# 侧边栏：数据上传区域
st.sidebar.header("📁 Data Source")
data_option = st.sidebar.radio("Select your dataset:", ["Upload Your Own Dataset", "Use Default (UCI Retail)"])

@st.cache_data # 使用缓存加速加载
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file, encoding='unicode_escape')
        else:
            return pd.read_excel(uploaded_file)
    else:
        # 这里假设你们本地放了一个 sample_data.csv 作为默认数据
        # 实际演示时请确保文件存在
        return pd.DataFrame({"Message": ["Please upload a dataset or provide a default file."]})

# 数据加载逻辑
df = None
if data_option == "Upload Your Own Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload E-commerce data (CSV/Excel)", type=["csv", "xlsx"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
elif data_option == "Use Default (UCI Retail)":
    # df = load_data() # 等你们有了默认文件再取消注释
    st.sidebar.info("Default dataset selected (Placeholder).")

# --- 页面路由逻辑 ---
if choice == "1. Data Overview & Preprocessing":
    st.header("📊 Data Overview")
    if df is not None:
        st.write(f"Dataset Shape: **{df.shape[0]} rows**, **{df.shape[1]} columns**")
        st.dataframe(df.head())
        
        st.subheader("🛠️ RFM Feature Extraction")
        st.info("Here we will show how raw transaction data is converted into Recency, Frequency, and Monetary (RFM) features for clustering.")
        # 这里之后可以引入 data_preprocessing.py 的逻辑
    else:
        st.warning("Please upload a dataset from the sidebar to start.")

elif choice == "2. K-Means Clustering (Member A)":
    st.header("🎯 K-Means Clustering")
    st.write("This module is developed by Member A.")
    # 这里之后引入 kmeans_clustering.py

elif choice == "3. DBSCAN Clustering (Member B)":
    st.header("🔍 DBSCAN Clustering")
    st.write("This module is developed by Member B.")
    # 这里之后引入 dbscan_clustering.py

elif choice == "4. MeanShift Clustering (Member C)":
    st.header("🌌 MeanShift Clustering")
    st.write("This module is developed by Member C.")
    # 这里之后引入 meanshift_clustering.py
