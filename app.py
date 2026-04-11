import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

import data_preprocessing as dp
import kmeans_clustering as km
import dbscan_clustering as db
import meanshift_clustering as ms

st.set_page_config(page_title="AI Customer Insight", page_icon="🛒", layout="wide")

    st.markdown("""
        <div style='text-align: center; 
                    padding: 20px 10px; 
                    background-color: #1A1C24; 
                    border-radius: 12px; 
                    border: 1px solid rgba(255, 75, 75, 0.2); 
                    box-shadow: 0 4px 20px rgba(255, 75, 75, 0.15); 
                    margin-bottom: 20px;'>
            <h2 style='margin: 0; 
                       font-family: "Arial Black", sans-serif; 
                       background: linear-gradient(45deg, #FF4B4B, #FFA0A0); 
                       -webkit-background-clip: text; 
                       -webkit-text-fill-color: transparent; 
                       font-size: 26px; 
                       letter-spacing: -0.5px;'>
                🛍️ Customers Segmenter
            </h2>
            <p style='margin: 8px 0 0 0; 
                      color: #8892B0; 
                      font-size: 11px; 
                      font-weight: 700; 
                      letter-spacing: 1.5px; 
                      text-transform: uppercase;'>
                E-commerce Analytics Pro
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    choice = option_menu(
        menu_title=None, 
        options=["Data Overview", "K-Means (A)", "DBSCAN (B)", "MeanShift (C)"],
        icons=['database-fill-check', 'pie-chart-fill', 'water', 'cpu-fill'], 
        menu_icon="cast", 
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#FF4B4B", "font-size": "18px"}, 
            "nav-link": {
                "font-size": "15px", 
                "text-align": "left", 
                "margin":"5px 0", 
                "--hover-color": "#333" 
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
if uploaded_file is not None:

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
        raw_df = df
        rfm_df = st.session_state.rfm_df
        
        st.markdown("### 💡 Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="👥 Total Customers", value=f"{len(rfm_df):,}")
        with col2:
            st.metric(label="🛒 Total Transactions", value=f"{len(raw_df):,}")
        with col3:
            avg_spend = rfm_df['Monetary'].mean()
            st.metric(label="💰 Avg Spend/Customer", value=f"${avg_spend:.2f}")
        with col4:
            st.metric(label="📅 Data Rows", value=f"{len(raw_df):,}")
            
        st.divider() 
        
        with st.expander("📂 View Raw Data Preview (Click to expand)", expanded=False):
            st.info("This is the original transactional data uploaded to the system. Displaying top 100 rows.")
            st.dataframe(raw_df.head(100), use_container_width=True) 
            
        st.markdown("### ⚙️ Feature Engineering Results")
        tab1, tab2 = st.tabs(["🔢 Calculated RFM Variables", "📏 Scaled Features (For AI)"])
        
        with tab1:
            st.caption("Recency, Frequency, and Monetary values calculated per customer.")
            st.dataframe(rfm_df, use_container_width=True)
            
        with tab2:
            st.caption("Data normalized for distance-based clustering algorithms.")
            st.dataframe(st.session_state.rfm_scaled_df, use_container_width=True) 

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
