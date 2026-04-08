import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

import data_preprocessing as dp
import kmeans_clustering as km
import dbscan_clustering as db
import meanshift_clustering as ms

st.set_page_config(page_title="AI Customer Insight", page_icon="🛒", layout="wide")

with st.sidebar:
    st.markdown("""
        <div style='text-align: center;'>
            <h2 style='color: #FF4B4B;'>🛍️ AI Segmenter</h2>
            <p style='color: #666;'>E-commerce Analytics Pro</p>
        </div>
    """, unsafe_allow_html=True)
    
   choice = option_menu(
        menu_title="Main Menu", 
        options=["Data Overview", "K-Means (A)", "DBSCAN (B)", "MeanShift (C)"],
        icons=['database-fill-check', 'pie-chart-fill', 'water', 'cpu-fill'], 
        menu_icon="cast", 
        default_index=0,
        styles={
            "container": {
                "padding": "5!important", 
                "background-color": "#262730", 
                "border": "1px solid #41434d"
            },
            "icon": {
                "color": "#00FFCC", 
                "font-size": "20px"
            }, 
            "nav-link": {
                "color": "#FFFFFF", 
                "font-size": "16px", 
                "text-align": "left", 
                "margin":"5px", 
                "--hover-color": "#3e3f4b" 
            },
            "nav-link-selected": {
                "background-color": "#FF4B4B", 
                "font-weight": "bold",
                "color": "white"
            },
        }
    )
    
    st.markdown("---")
    uploaded_file = st.file_uploader("📂 Upload Dataset (CSV/Excel)", type=["csv", "xlsx"])


if 'rfm_df' not in st.session_state:
    st.session_state.rfm_df = None
if 'rfm_scaled_df' not in st.session_state:
    st.session_state.rfm_scaled_df = None

df = None
if uploaded_file:
 
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, encoding='unicode_escape')
    else:
        df = pd.read_excel(uploaded_file)
    

    cleaned_df = dp.clean_data(df)
    st.session_state.rfm_df = dp.calculate_rfm(cleaned_df)
    st.session_state.rfm_scaled_df, _ = dp.scale_rfm_data(st.session_state.rfm_df)
    st.sidebar.success("✅ Data Processed!")

if choice == "Data Overview":
    st.title("📊 Data Overview & Preprocessing")
    if df is not None:
        st.write("### Raw Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### 🔢 Calculated RFM")
            st.dataframe(st.session_state.rfm_df.head(), use_container_width=True)
        with col2:
            st.write("### 📏 Scaled Features")
            st.dataframe(st.session_state.rfm_scaled_df.head(), use_container_width=True)
    else:
        st.info("👋 Welcome! Please upload your e-commerce CSV file from the sidebar to begin.")

elif choice == "K-Means (A)":
    st.title("🎯 K-Means Clustering Analysis")
    if st.session_state.rfm_scaled_df is not None:
        km.run_kmeans_app(st.session_state.rfm_df.copy(), st.session_state.rfm_scaled_df.copy())
    else:
        st.warning("⚠️ Please upload data first.")

elif choice == "DBSCAN (B)":
    st.title("🔍 DBSCAN Density-Based Clustering")
    if st.session_state.rfm_scaled_df is not None:
        db.run_dbscan_app(st.session_state.rfm_df.copy(), st.session_state.rfm_scaled_df.copy())
    else:
        st.warning("⚠️ Please upload data first.")

elif choice == "MeanShift (C)":
    st.title("🌌 MeanShift Autonomous Clustering")
    if st.session_state.rfm_scaled_df is not None:
        ms.run_meanshift_app(st.session_state.rfm_df.copy(), st.session_state.rfm_scaled_df.copy())
    else:
        st.warning("⚠️ Please upload data first.")
