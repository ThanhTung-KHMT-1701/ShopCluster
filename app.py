# =============================================================================
# SHOPCLUSTER DASHBOARD - STREAMLIT APPLICATION
# =============================================================================
# Dashboard trực quan hóa kết quả phân cụm khách hàng dựa trên luật kết hợp
# Mini Project: Khai Phá Dữ Liệu - Nhóm 09
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os

# =============================================================================
# CẤU HÌNH TRANG
# =============================================================================
st.set_page_config(
    page_title="ShopCluster Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# ĐƯỜNG DẪN DỮ LIỆU
# =============================================================================
DATA_DIR = "data/mini_project"
IMAGES_DIR = "images"

# =============================================================================
# HÀM LOAD DỮ LIỆU (có cache để tối ưu hiệu suất)
# =============================================================================
@st.cache_data
def load_rules():
    """Load luật kết hợp đã lọc"""
    return pd.read_csv(f"{DATA_DIR}/rules_fpgrowth_filtered.csv")

@st.cache_data
def load_cluster_profiles():
    """Load profile đầy đủ của các cụm"""
    return pd.read_csv(f"{DATA_DIR}/cluster_profiles_complete.csv")

@st.cache_data
def load_marketing_strategies():
    """Load chiến lược marketing"""
    return pd.read_csv(f"{DATA_DIR}/cluster_marketing_strategies.csv")

@st.cache_data
def load_rfm_stats():
    """Load thống kê RFM theo cụm"""
    return pd.read_csv(f"{DATA_DIR}/cluster_rfm_stats.csv")

@st.cache_data
def load_feature_comparison():
    """Load so sánh các biến thể feature"""
    return pd.read_csv(f"{DATA_DIR}/feature_variants_comparison.csv")

@st.cache_data
def load_customer_clusters():
    """Load phân cụm khách hàng V4"""
    return pd.read_csv(f"{DATA_DIR}/customer_clusters_v4_k5.csv")

def load_image(image_name):
    """Load hình ảnh từ thư mục images"""
    image_path = f"{IMAGES_DIR}/{image_name}"
    if os.path.exists(image_path):
        return Image.open(image_path)
    return None

# =============================================================================
# SIDEBAR - NAVIGATION
# =============================================================================
st.sidebar.title("🛒 ShopCluster")
st.sidebar.markdown("---")

# Menu điều hướng
menu = st.sidebar.radio(
    "📌 Điều hướng",
    [
        "🏠 Tổng quan",
        "📜 Luật Kết Hợp",
        "🎨 Feature Engineering",
        "🔬 Kết quả Clustering",
        "👥 Phân Khúc Khách Hàng",
        "📈 Chiến Lược Marketing",
        "📊 Bổ sung"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
Nhóm 09\n
Họ và tên: Lưu Thanh Tùng
""")

# =============================================================================
# TAB 1: TỔNG QUAN
# =============================================================================
if menu == "🏠 Tổng quan":
    st.title("🛒 ShopCluster Dashboard")
    st.markdown("### Phân Cụm Khách Hàng Dựa Trên Luật Kết Hợp")
    
    st.markdown("---")
    
    # KPIs tổng quan
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Tổng khách hàng",
            value="3,921",
            delta="UK Market"
        )
    
    with col2:
        st.metric(
            label="📜 Luật kết hợp",
            value="200",
            delta="Top-K theo Lift"
        )
    
    with col3:
        st.metric(
            label="👥 Số cụm (V4)",
            value="5",
            delta="Silhouette: 0.809"
        )
    
    with col4:
        st.metric(
            label="🎯 Avg Lift",
            value="42.19",
            delta="+70 max"
        )
    
    st.markdown("---")
    
    # Pipeline tổng quan
    st.subheader("🔄 Pipeline Phân Tích")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        | Bước | Mô tả | Kết quả |
        |------|-------|---------|
        | **1. Association Rules** | Khai thác luật kết hợp bằng FP-Growth | 200 luật (Lift > 20) |
        | **2. Feature Engineering** | Tạo 4 biến thể đặc trưng | V1, V2, V3, V4 |
        | **3. K-Means Clustering** | Chọn K tối ưu bằng Elbow + Silhouette | K=5 cho V4 |
        | **4. Visualization** | PCA/SVD giảm chiều về 2D | 73.3% variance |
        | **5. Comparison** | So sánh các biến thể | V4 tốt nhất |
        | **6. Profiling** | Đặt tên cụm + Chiến lược | 5 segments |
        """)
    
    with col2:
        st.markdown("""
        **📁 Dữ liệu:**
        - Online Retail Dataset
        - 18,021 hóa đơn
        - 4,007 sản phẩm
        - 3,921 khách hàng UK
        
        **⚙️ Tham số:**
        - min_support: 1%
        - min_confidence: 30%
        - min_lift: 1.5
        """)
    
    st.markdown("---")
    
    # Kết quả chính
    st.subheader("🎯 Kết Quả Phân Khúc (V4_Antecedent2)")
    
    try:
        df_profiles = load_cluster_profiles()
        
        # Hiển thị bảng tóm tắt
        display_cols = ['Cluster', 'Name_EN', 'Segment_Type', 'N_Customers', 'Pct', 'R_Mean', 'F_Mean', 'M_Mean']
        available_cols = [c for c in display_cols if c in df_profiles.columns]
        
        if available_cols:
            df_display = df_profiles[available_cols].copy()
            df_display['Pct'] = df_display['Pct'].round(1).astype(str) + '%'
            df_display['R_Mean'] = df_display['R_Mean'].round(0).astype(int).astype(str) + ' days'
            df_display['F_Mean'] = df_display['F_Mean'].round(1)
            df_display['M_Mean'] = df_display['M_Mean'].round(0).astype(int).astype(str) + ' GBP'
            
            st.dataframe(df_display)
    except Exception as e:
        st.warning(f"Chưa có dữ liệu cluster profiles: {e}")

# =============================================================================
# TAB 2: LUẬT KẾT HỢP
# =============================================================================
elif menu == "📜 Luật Kết Hợp":
    st.title("📜 Luật Kết Hợp (Association Rules)")
    st.markdown("### Yêu cầu 1: Khai thác luật kết hợp bằng FP-Growth")
    
    st.markdown("---")
    
    # Load dữ liệu
    try:
        df_rules = load_rules()
        
        # Filters
        st.subheader("🔍 Bộ lọc")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            min_lift = st.slider("Min Lift", 1.0, 75.0, 20.0, 1.0)
        with col2:
            min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.3, 0.05)
        with col3:
            min_sup = st.slider("Min Support", 0.0, 0.03, 0.01, 0.001)
        with col4:
            top_k = st.slider("Top K Rules", 10, 200, 50, 10)
        
        # Lọc dữ liệu
        df_filtered = df_rules[
            (df_rules['lift'] >= min_lift) &
            (df_rules['confidence'] >= min_conf) &
            (df_rules['support'] >= min_sup)
        ].head(top_k)
        
        st.markdown("---")
        
        # Hiển thị bảng luật
        st.subheader(f"📋 Top {len(df_filtered)} Luật Kết Hợp")
        
        # Chọn cột hiển thị
        display_cols = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']
        available_cols = [c for c in display_cols if c in df_filtered.columns]
        
        if available_cols:
            df_show = df_filtered[available_cols].copy()
            df_show.columns = ['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift']
            df_show['Support'] = df_show['Support'].round(4)
            df_show['Confidence'] = (df_show['Confidence'] * 100).round(1).astype(str) + '%'
            df_show['Lift'] = df_show['Lift'].round(2)
            
            st.dataframe(df_show)
        
        st.markdown("---")
        
        # Hiển thị biểu đồ
        st.subheader("📊 Trực quan hóa")
        
        col1, col2 = st.columns(2)
        
        with col1:
            img = load_image("Req1_Top15RulesByLift.png")
            if img:
                st.image(img, caption="Top 15 Rules by Lift", use_column_width=True)
        
        with col2:
            img = load_image("Req1_SupportConfidenceScatter.png")
            if img:
                st.image(img, caption="Support vs Confidence", use_column_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            img = load_image("Req1_LiftDistribution.png")
            if img:
                st.image(img, caption="Lift Distribution", use_column_width=True)
        
        with col4:
            img = load_image("Req1_MetricsDistribution.png")
            if img:
                st.image(img, caption="Metrics Distribution", use_column_width=True)
        
    except Exception as e:
        st.error(f"Lỗi load dữ liệu: {e}")

# =============================================================================
# TAB 3: FEATURE ENGINEERING
# =============================================================================
elif menu == "🎨 Feature Engineering":
    st.title("🎨 Feature Engineering")
    st.markdown("### Yêu cầu 2: Xây dựng các biến thể đặc trưng")
    
    st.markdown("---")
    
    # Mô tả 4 variants
    st.subheader("📦 4 Biến thể Feature Matrix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **V1_Binary (Baseline)**
        - Kích thước: 3,921 × 200
        - Giá trị: 0 hoặc 1
        - Logic: 1 nếu khách mua TẤT CẢ sản phẩm trong antecedents
        
        **V2_Weighted (Trọng số)**
        - Kích thước: 3,921 × 200
        - Giá trị: lift × confidence (7.45 - 71.15)
        - Phản ánh độ mạnh của luật
        """)
    
    with col2:
        st.markdown("""
        **V3_Binary_RFM (Kết hợp)**
        - Kích thước: 3,921 × 203
        - Binary rules + 3 cột RFM scaled
        - Kết hợp hành vi mua kèm + giá trị khách hàng
        
        **V4_Antecedent2 (Lọc phức tạp)**
        - Kích thước: 3,921 × 63
        - Chỉ giữ luật có antecedent ≥ 2 sản phẩm
        - Tập trung pattern mua kèm phức tạp
        """)
    
    st.markdown("---")
    
    # Bảng so sánh
    st.subheader("📊 So sánh các biến thể")
    
    try:
        df_comparison = load_feature_comparison()
        st.dataframe(df_comparison)
    except:
        st.info("Bảng so sánh sẽ hiển thị sau khi có dữ liệu")
    
    st.markdown("---")
    
    # Biểu đồ
    st.subheader("📈 Trực quan hóa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        img = load_image("Req2_FeatureVariantComparison.png")
        if img:
            st.image(img, caption="Feature Variant Comparison", use_column_width=True)
    
    with col2:
        img = load_image("Req2_RFMDistribution.png")
        if img:
            st.image(img, caption="RFM Distribution", use_column_width=True)

# =============================================================================
# TAB 4: KẾT QUẢ CLUSTERING
# =============================================================================
elif menu == "🔬 Kết quả Clustering":
    st.title("🔬 Kết quả Clustering")
    st.markdown("### Yêu cầu 3-5: Phân cụm K-Means và Trực quan hóa")
    
    st.markdown("---")
    
    # Sub-tabs
    sub_tab = st.radio(
        "Chọn phần xem:",
        ["Chọn K (Elbow & Silhouette)", "Trực quan 2D (PCA/SVD)", "So sánh Variants"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if sub_tab == "Chọn K (Elbow & Silhouette)":
        st.subheader("📉 Phương pháp Elbow và Silhouette Score")
        
        # Bảng K tối ưu
        st.markdown("""
        | Variant | K được chọn | Silhouette | Lý do |
        |---------|-------------|------------|-------|
        | V1_Binary | 2 | 0.7039 | Silhouette cao nhất |
        | V2_Weighted | 2 | 0.8920 | Silhouette cao nhất |
        | V3_Binary_RFM | 2 | 0.9622* | *Có outlier RFM |
        | V4_Antecedent2 | **5** | 0.8091 | Ưu tiên K>2, chênh <20% |
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            img = load_image("Req3_ElbowMethod.png")
            if img:
                st.image(img, caption="Elbow Method", use_column_width=True)
        
        with col2:
            img = load_image("Req3_SilhouetteScore.png")
            if img:
                st.image(img, caption="Silhouette Score", use_column_width=True)
        
        img = load_image("Req3_BestKComparison.png")
        if img:
            st.image(img, caption="Best K Comparison", use_column_width=True)
    
    elif sub_tab == "Trực quan 2D (PCA/SVD)":
        st.subheader("🎯 Giảm chiều về 2D")
        
        col1, col2 = st.columns(2)
        
        with col1:
            img = load_image("Req4_PCA_ClusterSeparation.png")
            if img:
                st.image(img, caption="PCA Cluster Separation", use_column_width=True)
        
        with col2:
            img = load_image("Req4_SVD_ClusterSeparation.png")
            if img:
                st.image(img, caption="SVD Cluster Separation", use_column_width=True)
        
        st.markdown("""
        **Nhận xét:**
        - **SVD phù hợp hơn** cho dữ liệu rule-based features (sparse, binary)
        - V4 có variance ratio 73.3% trên SVD, clusters tách biệt tốt
        - V3 có outlier gây méo visualization
        """)
    
    else:  # So sánh Variants
        st.subheader("⚖️ So sánh các biến thể")
        
        st.markdown("""
        ### Kết luận so sánh:
        
        | So sánh | Winner | Lý do |
        |---------|--------|-------|
        | **Binary vs Weighted** | V2_Weighted | Silhouette 0.892 vs 0.704 |
        | **Rule-only vs Rule+RFM** | V1_Binary | V3 có outlier không đáng tin |
        | **Top-K Large vs Small** | V4_Antecedent2 | 5 cụm đa dạng, Silhouette 0.809 |
        
        ### Khuyến nghị:
        - **Marketing Segmentation**: V4_Antecedent2 (5 cụm)
        - **Phân tích hành vi**: V2_Weighted
        - **Baseline**: V1_Binary
        """)

# =============================================================================
# TAB 5: PHÂN KHÚC KHÁCH HÀNG (QUAN TRỌNG NHẤT)
# =============================================================================
elif menu == "👥 Phân Khúc Khách Hàng":
    st.title("👥 Phân Khúc Khách Hàng")
    st.markdown("### Yêu cầu 6: Profiling và Diễn giải Cụm")
    
    st.markdown("---")
    
    # Load dữ liệu
    try:
        df_profiles = load_cluster_profiles()
        df_strategies = load_marketing_strategies()
        df_rules = load_rules()
        
        # Dropdown chọn cluster
        cluster_options = df_profiles['Cluster'].unique().tolist()
        cluster_names = {
            row['Cluster']: f"Cluster {row['Cluster']}: {row['Name_EN']}"
            for _, row in df_profiles.iterrows()
        }
        
        selected_cluster = st.selectbox(
            "🎯 Chọn Cluster để xem chi tiết:",
            options=cluster_options,
            format_func=lambda x: cluster_names.get(x, f"Cluster {x}")
        )
        
        st.markdown("---")
        
        # Lấy thông tin cluster được chọn
        cluster_info = df_profiles[df_profiles['Cluster'] == selected_cluster].iloc[0]
        strategy_info = df_strategies[df_strategies['Cluster'] == selected_cluster].iloc[0] if len(df_strategies[df_strategies['Cluster'] == selected_cluster]) > 0 else None
        
        # Hiển thị profile
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"### 📛 {cluster_info['Name_EN']}")
            st.markdown(f"**Tên Việt:** {cluster_info['Name_VN']}")
            st.markdown(f"**Segment Type:** {cluster_info['Segment_Type']}")
        
        with col2:
            st.metric("👥 Số khách hàng", f"{int(cluster_info['N_Customers']):,}")
            st.metric("📊 Tỷ lệ", f"{cluster_info['Pct']:.1f}%")
        
        with col3:
            st.metric("📅 Recency (Mean)", f"{cluster_info['R_Mean']:.0f} days")
            st.metric("🔄 Frequency (Mean)", f"{cluster_info['F_Mean']:.1f} orders")
            st.metric("💰 Monetary (Mean)", f"£{cluster_info['M_Mean']:,.0f}")
        
        st.markdown("---")
        
        # Persona
        st.subheader("🎭 Persona")
        if 'Persona' in cluster_info:
            st.info(cluster_info['Persona'])
        
        st.markdown("---")
        
        # Top Rules cho cluster này
        st.subheader("📜 Top Rules được kích hoạt")
        
        if strategy_info is not None and 'Top_Rules' in strategy_info:
            top_rules_str = strategy_info['Top_Rules']
            if pd.notna(top_rules_str):
                rule_ids = [r.strip() for r in str(top_rules_str).split(',')][:5]
                st.markdown(f"**Rules:** {', '.join(rule_ids)}")
        
        # Hiển thị 5 luật đầu tiên từ bộ rules
        st.dataframe(
            df_rules[['antecedents_str', 'consequents_str', 'lift', 'confidence']].head(5)
        )
        
        st.markdown("---")
        
        # Bundle Recommendations
        st.subheader("🎁 Bundle/Cross-sell Recommendations")
        
        if strategy_info is not None and 'Bundle_From_Rules' in strategy_info:
            bundle_str = strategy_info['Bundle_From_Rules']
            if pd.notna(bundle_str):
                bundles = str(bundle_str).split(' | ')
                for i, bundle in enumerate(bundles[:3], 1):
                    st.markdown(f"**Bundle {i}:** {bundle}")
        
        st.markdown("---")
        
        # Biểu đồ
        st.subheader("📊 Trực quan hóa")
        
        col1, col2 = st.columns(2)
        
        with col1:
            img = load_image("Req6_ClusterDistribution.png")
            if img:
                st.image(img, caption="Cluster Distribution", use_column_width=True)
        
        with col2:
            img = load_image("Req6_ClusterProfileSummary.png")
            if img:
                st.image(img, caption="Cluster Profile Summary", use_column_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            img = load_image("Req6_RuleActivationHeatmap.png")
            if img:
                st.image(img, caption="Rule Activation Heatmap", use_column_width=True)
        
        with col4:
            img = load_image(f"Req6_RFMByCluster_V4_Antecedent2.png")
            if img:
                st.image(img, caption="RFM by Cluster (V4)", use_column_width=True)
        
    except Exception as e:
        st.error(f"Lỗi load dữ liệu: {e}")
        st.info("Vui lòng chạy các bước phân tích trước để có dữ liệu.")

# =============================================================================
# TAB 6: CHIẾN LƯỢC MARKETING
# =============================================================================
elif menu == "📈 Chiến Lược Marketing":
    st.title("📈 Chiến Lược Marketing")
    st.markdown("### Đề xuất chiến lược cho từng phân khúc khách hàng")
    
    st.markdown("---")
    
    try:
        df_strategies = load_marketing_strategies()
        df_profiles = load_cluster_profiles()
        df_customers = load_customer_clusters()
        
        # Filter theo Segment Type
        segment_types = ['Tất cả'] + df_strategies['Segment_Type'].unique().tolist()
        selected_segment = st.selectbox("🎯 Lọc theo Segment Type:", segment_types)
        
        if selected_segment != 'Tất cả':
            df_filtered = df_strategies[df_strategies['Segment_Type'] == selected_segment]
        else:
            df_filtered = df_strategies
        
        st.markdown("---")
        
        # Hiển thị danh sách K khách hàng thuộc Segment Type đã chọn
        st.subheader("👥 Khách hàng theo Segment Type")
        k_customers = st.number_input("Số lượng khách hàng (K)", min_value=1, max_value=1000, value=10, step=1)
        
        try:
            if selected_segment != 'Tất cả':
                # Lấy danh sách cluster thuộc Segment Type đã chọn
                # Ưu tiên dùng profiles để đảm bảo mapping Cluster -> Segment_Type
                clusters_for_segment = df_profiles.loc[
                    df_profiles['Segment_Type'] == selected_segment, 'Cluster'
                ].dropna().unique().tolist()
            else:
                clusters_for_segment = df_profiles['Cluster'].dropna().unique().tolist()

            # Lọc khách hàng theo các cụm tương ứng
            df_segment_customers = df_customers[df_customers['Cluster'].isin(clusters_for_segment)].copy()

            # Loại bỏ giá trị ID không hợp lệ nếu có (ví dụ '000nan')
            df_segment_customers['CustomerID'] = df_segment_customers['CustomerID'].astype(str)
            df_segment_customers = df_segment_customers[
                df_segment_customers['CustomerID'].str.match(r'^\d+$')
            ]

            # Lấy K khách hàng đầu tiên
            df_show_customers = df_segment_customers.head(int(k_customers))

            # Hiển thị bảng khách hàng
            if len(df_show_customers) > 0:
                st.dataframe(df_show_customers.rename(columns={
                    'CustomerID': 'Customer ID',
                    'Cluster': 'Cluster'
                }))
            else:
                st.info("Không tìm thấy khách hàng cho Segment Type đã chọn.")
        except Exception as e:
            st.warning(f"Không thể hiển thị danh sách khách hàng: {e}")

        st.markdown("---")
        
        # Hiển thị bảng chiến lược
        for _, row in df_filtered.iterrows():
            with st.expander(f"🎯 Cluster {row['Cluster']}: {row['Name_EN']} ({row['Segment_Type']})", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**📋 Strategy Type:** {row['Strategy_Type']}")
                    st.markdown(f"**🎁 Offer:** {row['Offer']}")
                    st.markdown(f"**📱 Channel:** {row['Channel']}")
                
                with col2:
                    st.markdown(f"**⏰ Timing:** {row['Timing']}")
                    st.markdown(f"**📊 KPI Target:** {row['KPI_Target']}")
                
                if 'Strategy_Detail' in row and pd.notna(row['Strategy_Detail']):
                    st.info(f"**Chi tiết:** {row['Strategy_Detail']}")
        
        st.markdown("---")
        
        # Biểu đồ phân bố strategy
        st.subheader("📊 Phân bố Chiến lược")
        
        img = load_image("Req6_StrategyDistribution.png")
        if img:
            st.image(img, caption="Distribution of Marketing Strategies", use_column_width=True)
        
        st.markdown("---")
        
        # Export button
        st.subheader("💾 Export Chiến lược")
        
        csv = df_strategies.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="marketing_strategies.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Lỗi load dữ liệu: {e}")

# =============================================================================
# TAB 7: BỔ SUNG - THỬ NGHIỆM TOPK VÀ DBSCAN
# =============================================================================
elif menu == "📊 Bổ sung":
    st.title("📊 C. Bổ sung")
    st.markdown("### Thử nghiệm bổ sung: TopK và So sánh thuật toán DBSCAN")
    
    st.markdown("---")
    
    # Sub-tabs
    sub_tab = st.radio(
        "📌 Chọn nội dung:",
        ["1️⃣ Thử nghiệm TopK", "2️⃣ So sánh K-Means vs DBSCAN"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # ==========================================================================
    # SUB-TAB 1: THỬ NGHIỆM TOPK
    # ==========================================================================
    if sub_tab == "1️⃣ Thử nghiệm TopK":
        st.subheader("🔬 Thử nghiệm giá trị TopK hợp lý")
        
        st.markdown("""
        **Mục tiêu:** Xác định giá trị TopK tối ưu cho việc chọn số luật kết hợp làm đầu vào cho phân cụm.
        
        **Các tiêu chí đánh giá:**
        - Chất lượng luật (Lift, Confidence)
        - Độ phủ khách hàng (Coverage)
        - Khả năng phân cụm (Silhouette Score)
        """)
        
        st.markdown("---")
        
        # Hiển thị bảng kết quả
        st.markdown("#### 📋 Kết quả thử nghiệm các giá trị TopK")
        
        try:
            df_topk = pd.read_csv(f"{DATA_DIR}/topk_experiment_results.csv")
            
            # Định dạng hiển thị
            df_display = df_topk[['TopK', 'Avg_Lift', 'Min_Lift', 'Avg_Confidence', 'Coverage', 'Best_Silhouette', 'Silhouette_K5']].copy()
            df_display['Avg_Lift'] = df_display['Avg_Lift'].round(2)
            df_display['Min_Lift'] = df_display['Min_Lift'].round(2)
            df_display['Avg_Confidence'] = (df_display['Avg_Confidence'] * 100).round(1).astype(str) + '%'
            df_display['Coverage'] = (df_display['Coverage'] * 100).round(1).astype(str) + '%'
            df_display['Best_Silhouette'] = df_display['Best_Silhouette'].round(3)
            df_display['Silhouette_K5'] = df_display['Silhouette_K5'].round(3)
            
            st.dataframe()
            
        except Exception as e:
            st.info("Dữ liệu thử nghiệm TopK chưa có. Vui lòng chạy notebook phần C. Bổ sung.")
        
        st.markdown("---")
        
        # Hiển thị biểu đồ
        st.markdown("#### 📈 Biểu đồ phân tích TopK")
        
        img_topk = load_image("TopK_Experiment_Results.png")
        if img_topk:
            st.image(img_topk, caption="Kết quả thử nghiệm các giá trị TopK", use_column_width=True)
        else:
            st.warning("Chưa có biểu đồ TopK_Experiment_Results.png")
        
        st.markdown("---")
        
        # Kết luận
        st.markdown("#### ✅ Kết luận")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **Chọn TopK = 200 vì:**
            1. ✅ **Độ phủ cao nhất**: 56.8% khách hàng
            2. ✅ **Điểm bão hòa tự nhiên**: TopK > 200 không thêm luật
            3. ✅ **Min Lift = 20.04**: Vẫn là liên kết mạnh
            4. ✅ **Giá trị thực tiễn cao**
            """)
        
        with col2:
            st.info("""
            **Trade-off:**
            - TopK=50: Silhouette cao (0.906) nhưng Coverage chỉ 3.5%
            - TopK=200: Coverage 56.8%, Silhouette 0.223 (K=5)
            - **Ưu tiên Coverage** cho marketing thực tế
            """)
    
    # ==========================================================================
    # SUB-TAB 2: SO SÁNH K-MEANS VS DBSCAN
    # ==========================================================================
    elif sub_tab == "2️⃣ So sánh K-Means vs DBSCAN":
        st.subheader("🔬 So sánh K-Means và DBSCAN trên biến thể V4")
        
        st.markdown("""
        **Mục tiêu:** So sánh hiệu quả phân cụm giữa **K-Means (V4, K=5)** và **DBSCAN** dựa trên:
        - Metrics thống kê: Silhouette, Davies-Bouldin, Calinski-Harabasz
        - Mức độ "Actionable" - khả năng áp dụng vào thực tế marketing
        """)
        
        st.markdown("---")
        
        # 1. Tìm tham số DBSCAN
        st.markdown("#### 🔍 Bước 1: Tìm tham số tối ưu cho DBSCAN")
        
        img_param = load_image("DBSCAN_ParameterSearch.png")
        if img_param:
            st.image(img_param, caption="K-Distance Graph và Grid Search cho DBSCAN", use_column_width=True)
        else:
            st.warning("Chưa có biểu đồ DBSCAN_ParameterSearch.png")
        
        st.info("""
        **Kết quả Grid Search:**
        - Tham số tối ưu: **eps = 0.15, min_samples = 5**
        - Silhouette Score: 0.484
        - Số cụm: 2 (+ noise points)
        """)
        
        st.markdown("---")
        
        # 2. So sánh Metrics
        st.markdown("#### 📊 Bước 2: So sánh Metrics")
        
        img_compare = load_image("KMeans_vs_DBSCAN_Comparison.png")
        if img_compare:
            st.image(img_compare, caption="So sánh K-Means vs DBSCAN", use_column_width=True)
        else:
            st.warning("Chưa có biểu đồ KMeans_vs_DBSCAN_Comparison.png")
        
        # Bảng so sánh metrics
        st.markdown("##### 📋 Bảng so sánh chi tiết")
        
        metrics_data = {
            'Metric': ['Silhouette Score ↑', 'Davies-Bouldin Index ↓', 'Calinski-Harabasz ↑', 'Số cụm có ý nghĩa', 'Coverage'],
            'K-Means (V4, K=5)': ['0.223', '1.53', '341.2', '5', '100%'],
            'DBSCAN': ['0.484', '0.82', '587.8', '2', '76.9%'],
            'Winner': ['DBSCAN', 'DBSCAN', 'DBSCAN', 'K-Means', 'K-Means']
        }
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe()
        
        st.markdown("---")
        
        # 3. Đánh giá Actionable
        st.markdown("#### 🎯 Bước 3: Đánh giá mức độ 'Actionable'")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**K-Means - RFM by Cluster:**")
            kmeans_rfm = {
                'Cluster': [0, 1, 2, 3, 4],
                'N': [297, 124, 251, 1443, 113],
                'R_Mean': [28, 61, 38, 79, 52],
                'F_Mean': [5.8, 21.3, 6.1, 4.7, 10.6],
                'M_Mean': ['2,113', '17,366', '3,043', '1,990', '6,074']
            }
            st.dataframe()
        
        with col2:
            st.markdown("**DBSCAN - RFM by Cluster:**")
            dbscan_rfm = {
                'Cluster': [0, 1, 'Noise'],
                'N': [1672, 41, 515],
                'R_Mean': [75, 100, '-'],
                'F_Mean': [4.4, 4.7, '-'],
                'M_Mean': ['1,900', '1,704', '-']
            }
            st.dataframe()
        
        st.markdown("---")
        
        # 4. Kết luận cuối cùng
        st.markdown("#### 🏆 Bước 4: Kết luận và Khuyến nghị")
        
        img_verdict = load_image("KMeans_vs_DBSCAN_FinalVerdict.png")
        if img_verdict:
            st.image(img_verdict, caption="Final Verdict: K-Means vs DBSCAN", use_column_width=True)
        else:
            st.warning("Chưa có biểu đồ KMeans_vs_DBSCAN_FinalVerdict.png")
        
        # Actionable Score comparison
        st.markdown("##### 📊 Điểm Actionable Score")
        
        actionable_data = {
            'Metric': ['Meaningful Clusters (>1%)', 'RFM Discrimination (CV)', 'Coverage', 'Cluster Balance (Entropy)', 'TOTAL ACTIONABLE SCORE'],
            'K-Means': ['5', '0.718', '100%', '0.688', '**0.853**'],
            'DBSCAN': ['2', '0.107', '76.9%', '0.163', '**0.357**']
        }
        df_actionable = pd.DataFrame(actionable_data)
        st.dataframe()
        
        st.markdown("---")
        
        # Final verdict
        col1, col2 = st.columns(2)
        
        with col1:
            st.error("""
            **DBSCAN thắng về thống kê:**
            - Silhouette cao hơn 117%
            - DBI thấp hơn 46%
            - CH Index cao hơn 72%
            """)
        
        with col2:
            st.success("""
            **K-Means thắng về ứng dụng:**
            - Coverage 100% (không bỏ sót)
            - 5 cụm đa dạng cho marketing
            - Actionable Score cao hơn **139%**
            """)
        
        st.markdown("---")
        
        st.info("""
        ### 🏆 KHUYẾN NGHỊ: Chọn K-Means (V4, K=5)
        
        **Lý do:**
        1. ✅ **Coverage 100%** - Không bỏ sót khách hàng nào
        2. ✅ **5 cụm đa dạng** - Đủ chi tiết để tạo 5 chiến lược marketing khác biệt
        3. ✅ **RFM discrimination cao (0.718)** - Phân biệt rõ ràng hành vi khách hàng
        4. ✅ **Actionable Score 0.853** - Khả năng áp dụng thực tế cao
        """)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🛒 ShopCluster Dashboard | Mini Project: Phân Cụm Khách Hàng Dựa Trên Luật Kết Hợp</p>
    <p>Môn: Khai Phá Dữ Liệu | Nhóm: 09 | GV: Cô Lê Thị Thùy Trang</p>
</div>
""", unsafe_allow_html=True)
