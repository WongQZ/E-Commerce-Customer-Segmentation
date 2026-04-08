import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import plotly.express as px
import pandas as pd
import numpy as np

def run_dbscan_app(rfm_df, rfm_scaled_df):
    st.markdown("### Algorithm Configuration (DBSCAN)")
    st.write("DBSCAN is great at finding arbitrarily shaped clusters and identifying outliers (noise).")
    
    col1, col2 = st.columns(2)
    with col1:
        eps_value = st.slider("Epsilon (eps):", min_value=0.1, max_value=3.0, value=0.5, step=0.1, 
                              help="The maximum distance between two samples for one to be considered as in the neighborhood of the other.")
    with col2:
        min_samples_value = st.slider("Minimum Samples:", min_value=2, max_value=20, value=5, step=1,
                                      help="The number of samples in a neighborhood for a point to be considered as a core point.")
        
    if st.button("Run DBSCAN Clustering"):
        with st.spinner('Scanning for clusters...'):
            dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
            cluster_labels = dbscan.fit_predict(rfm_scaled_df)
            
            rfm_df['Cluster'] = cluster_labels
            rfm_df['Cluster'] = rfm_df['Cluster'].astype(str)
            
            n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise_ = list(cluster_labels).count(-1)
            
            st.markdown("### 📈 Clustering Results & Evaluation")
            
            if n_clusters_ > 1:
                silhouette_avg = silhouette_score(rfm_scaled_df, cluster_labels)
                st.success(f"DBSCAN found **{n_clusters_}** distinct clusters and **{n_noise_}** noise points/outliers.")
                
                c1, c2 = st.columns(2)
                c1.metric(label="Estimated Number of Clusters", value=n_clusters_)
                c2.metric(label="Silhouette Score", value=f"{silhouette_avg:.4f}")
            else:
                st.warning(f"DBSCAN only found **{n_clusters_}** clusters and **{n_noise_}** noise points. Try adjusting Eps and Min Samples. (Silhouette Score requires at least 2 clusters).")
            
            st.markdown("### 🌐 3D Cluster Visualization")
            
            unique_clusters = rfm_df['Cluster'].unique()
            color_map = {str(c): px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, c in enumerate(unique_clusters) if c != '-1'}
            color_map['-1'] = 'black' 
            
            fig = px.scatter_3d(
                rfm_df, 
                x='Recency', y='Frequency', z='Monetary',
                color='Cluster', 
                hover_name='CustomerID',
                title=f'DBSCAN Segments',
                opacity=0.9,
                
                color_discrete_map={
                    "-1": "#FF4B4B",  
                    "0": "#00FFCC",  
                    "1": "#B14BFF",   
                    "2": "#FFD700",   
                    "3": "#00BFFF"    
                }
            )
            
            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=30),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            if n_noise_ > 0:
                st.info("💡 **Business Insight**: The black dots (Cluster -1) represent outliers. In e-commerce, these could be extremely rare 'whale' customers or anomalous transactions that need manual review.")
