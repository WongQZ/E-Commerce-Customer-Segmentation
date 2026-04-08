import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import pandas as pd

def run_kmeans_app(rfm_df, rfm_scaled_df):
    st.markdown("### ⚙️ Algorithm Configuration")
    
    k_value = st.slider("Select the number of clusters (K):", min_value=2, max_value=8, value=3, step=1)
    
    if st.button("Run K-Means Clustering"):
        with st.spinner('Calculating clusters...'):

            kmeans = KMeans(n_clusters=k_value, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(rfm_scaled_df)
            

            rfm_df['Cluster'] = cluster_labels
            rfm_df['Cluster'] = rfm_df['Cluster'].astype(str) 
            
            silhouette_avg = silhouette_score(rfm_scaled_df, cluster_labels)
            

            st.markdown("### 📈 Clustering Results & Evaluation")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Model Inertia (WCSS)", value=f"{kmeans.inertia_:,.2f}", help="Lower is better.")
            with col2:
                st.metric(label="Silhouette Score", value=f"{silhouette_avg:.4f}", help="Closer to 1 is better. Evaluates cluster separation.")
                
            st.markdown("### 🌐 3D Cluster Visualization")
            st.info("Rotate and zoom the 3D plot to explore the customer segments.")
            
            fig = px.scatter_3d(
                rfm_df, 
                x='Recency', y='Frequency', z='Monetary',
                color='Cluster', 
                hover_name='CustomerID',
                title=f'Customer Segments (K={k_value})',
                opacity=0.7,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 📊 Business Insights (Cluster Averages)")
            cluster_summary = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
            st.dataframe(cluster_summary, use_container_width=True)
