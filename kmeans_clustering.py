import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

def plot_radar_chart(rfm_df):
    
    cluster_avg = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
    
    scaler = MinMaxScaler()
    cluster_avg[['Recency', 'Frequency', 'Monetary']] = scaler.fit_transform(cluster_avg[['Recency', 'Frequency', 'Monetary']])
    
    fig = go.Figure()
    categories = ['Recency', 'Frequency', 'Monetary']
    
    colors = px.colors.qualitative.Pastel
    
    for i, row in cluster_avg.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['Recency'], row['Frequency'], row['Monetary'], row['Recency']],
            theta=categories + [categories[0]],
            fill='toself',
            name=f'Cluster {row["Cluster"]}',
            line=dict(color=colors[i % len(colors)])
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title="Customer Segment Profiles (Radar Chart)",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

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
            
            fig_3d = px.scatter_3d(
                rfm_df, 
                x='Recency', y='Frequency', z='Monetary',
                color='Cluster', 
                hover_name='CustomerID',
                title=f'Customer Segments (K={k_value})',
                opacity=0.7,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig_3d, use_container_width=True)
    
            st.markdown("### 📊 Business Insights (Cluster Averages)")
            cluster_summary = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
            st.dataframe(cluster_summary, use_container_width=True)
            

            st.markdown("### 🎯 DNA Profiling: Segment Characteristics")
            st.info("The Radar Chart normalizes the values (0 to 1) to visually compare the 'shape' of each segment.")
            
            fig_radar = plot_radar_chart(rfm_df)
            st.plotly_chart(fig_radar, use_container_width=True)
