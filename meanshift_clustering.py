import streamlit as st
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
import plotly.express as px
import pandas as pd

def run_meanshift_app(rfm_df, rfm_scaled_df):
    st.markdown("### ⚙️ Algorithm Configuration (MeanShift)")
    st.write("MeanShift is a density-based algorithm that **automatically discovers the number of clusters** without needing a pre-defined K.")
    
    # 交互式选项：自动估算 vs 手动调节
    auto_bandwidth = st.checkbox("Automatically estimate bandwidth (Recommended)", value=True)
    
    bandwidth_value = None
    if not auto_bandwidth:
        bandwidth_value = st.slider("Bandwidth (Smoothing parameter):", min_value=0.5, max_value=5.0, value=1.5, step=0.1,
                                    help="Dictates the size of the region to search through. Smaller values create more clusters.")
        
    if st.button("Run MeanShift Clustering"):
        with st.spinner('Calculating density centers (This might take a moment)...'):
            
            # 1. 带宽处理逻辑
            if auto_bandwidth:
                # 使用 sklearn 自动估算最优带宽，quantile 越小，聚类越多
                estimated_bw = estimate_bandwidth(rfm_scaled_df, quantile=0.2, n_samples=min(1000, len(rfm_scaled_df)))
                st.info(f"Auto-estimated bandwidth: **{estimated_bw:.4f}**")
                ms = MeanShift(bandwidth=estimated_bw, bin_seeding=True)
            else:
                ms = MeanShift(bandwidth=bandwidth_value, bin_seeding=True)
                
            # 2. 训练与预测
            cluster_labels = ms.fit_predict(rfm_scaled_df)
            
            rfm_df['Cluster'] = cluster_labels
            rfm_df['Cluster'] = rfm_df['Cluster'].astype(str)
            
            n_clusters_ = len(set(cluster_labels))
            
            # 3. 评估指标展示
            st.markdown("### 📈 Clustering Results & Evaluation")
            
            if n_clusters_ > 1:
                silhouette_avg = silhouette_score(rfm_scaled_df, cluster_labels)
                c1, c2 = st.columns(2)
                c1.metric(label="Automatically Discovered Clusters", value=n_clusters_)
                c2.metric(label="Silhouette Score", value=f"{silhouette_avg:.4f}")
            else:
                st.warning("MeanShift combined all data into a single cluster. Try unchecking auto-bandwidth and using a smaller manual bandwidth.")
            
            # 4. 3D 可视化
            st.markdown("### 🌐 3D Cluster Visualization")
            fig = px.scatter_3d(
                rfm_df, x='Recency', y='Frequency', z='Monetary',
                color='Cluster', hover_name='CustomerID',
                title='MeanShift Customer Segments',
                opacity=0.7, color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig, use_container_width=True)
            
            # 展示核心数据
            if n_clusters_ > 1:
                st.markdown("### 📊 Business Insights (Cluster Averages)")
                cluster_summary = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
                st.dataframe(cluster_summary, use_container_width=True)
