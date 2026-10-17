import io
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
DATA_DIR = BASE.parent / "data"

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CLUSTER_NAMES = {
    0: "Đơn thông thường",
    1: "Đơn thông thường",
    2: "Đơn mua nhiều",
    3: "Sản phẩm giá cao",
}

st.set_page_config(page_title="Coffee Sales - Stage 2", layout="wide")
st.title("Coffee Sales - Dự đoán và phân cụm giao dịch")


@st.cache_resource
def load_models():
    reg = joblib.load(MODEL_DIR / "LeGiaBao_regression_model.pkl")
    clf = joblib.load(MODEL_DIR / "PhamVietTien_classification_model.pkl")
    km = joblib.load(MODEL_DIR / "LeDangKhoa_kmeans_model.pkl")
    return reg, clf, km


@st.cache_data
def load_cluster_data():
    cols = ["Quantity", "Price", "Sales Amount", "PC1", "PC2", "Cluster_Label", "Cluster_Label_PCA"]
    return pd.read_csv(DATA_DIR / "LeDangKhoa_clustered.csv", usecols=lambda c: c in cols)

reg_model, clf_model, km_bundle = load_models()

#Lấy danh sách lựa chọn cho dropdown từ chính model (đảm bảo khớp dữ liệu lúc train)
encoder = reg_model.named_steps["preprocessor"].named_transformers_["categorical"].named_steps["encoder"]
PRODUCTS, LOCATIONS, SOURCES, _ = [list(c) for c in encoder.categories_]

tab1, tab2, tab3, tab4 = st.tabs(
    ["Dự đoán giá", "Giao dịch giá trị cao", "Phân cụm", "Biểu đồ"]
)

#Tab1: Regression
with tab1:
    st.subheader("Dự đoán giá sản phẩm (Linear Regression)")
    c1, c2 = st.columns(2)
    product = c1.selectbox("Sản phẩm", PRODUCTS, key="r_product")
    location = c1.selectbox("Địa điểm / kênh bán", LOCATIONS, key="r_location")
    source = c1.selectbox("Nguồn dữ liệu", SOURCES, key="r_source")
    quantity = c2.slider("Số lượng", 1, 10, 1, key="r_quantity")
    month = c2.slider("Tháng", 1, 12, 6, key="r_month")
    weekday = c2.selectbox("Thứ", WEEKDAYS, key="r_weekday")

    if st.button("Dự đoán giá", key="r_btn"):
        row = pd.DataFrame([{
            "Quantity": quantity, "Product Name": product, "Store Location": location,
            "Source": source, "Month": month, "Weekday": weekday, 
        }])[list(reg_model.feature_names_in_)]
        price = reg_model.predict(row)[0]
        m1, m2 = st.columns(2)
        m1.metric("Giá dự đoán / sản phẩm", f"{price:.2f}")
        m2.metric("Doanh thu dự kiến", f"{price * quantity:.2f}")


#Tab2: Classification
with tab2:
    st.subheader("Dự đoán giao dịch giá trị cao (Logistic Regression)")
    st.caption("Giao dịch giá trị cao: Sales Amount lớn hơn phân vị 75 (khoảng 6,0).")
    c1, c2 = st.columns(2)
    product = c1.selectbox("Sản phẩm", PRODUCTS, key="c_product")
    location = c1.selectbox("Địa điểm / kênh bán", LOCATIONS, key="c_location")
    source = c1.selectbox("Nguồn dữ liệu", SOURCES, key="c_source")
    month = c2.slider("Tháng", 1, 12, 6, key="c_month")
    weekday = c2.selectbox("Thứ", WEEKDAYS, key="c_weekday")

    if st.button("Phân loại", key="c_btn"):
        row = pd.DataFrame([{
            "Product Name": product, "Store Location": location, "Source": source,
            "Month": month, "Weekday": weekday,
        }])[list(clf_model.feature_names_in_)]
        label = clf_model.predict(row)[0]
        proba = clf_model.predict_proba(row)[0, 1]
        if label == 1:
            st.warning(f"Giá trị cao (xác suất {proba:.1%})")
        else:
            st.success(f"Bình thường (xác suất giá trị cao chỉ {proba:.1%})")

#Tab3: Clustering
with tab3:
    st.subheader("Xác định cụm giao dịch (K-Means)")
    c1, c2 = st.columns(2)
    product = c1.selectbox("Sản phẩm", PRODUCTS, key="k_product")
    location = c1.selectbox("Địa điểm / kênh bán", LOCATIONS, key="k_location")
    source = c1.selectbox("Nguồn dữ liệu", SOURCES, key="k_source")
    quantity = c2.slider("Số lượng", 1, 10, 1, key="k_quantity")
    price = c2.number_input("Giá mỗi sản phẩm", min_value=0.0, max_value=60.0, value=3.0, step=0.5, key="k_price")
    month = c2.slider("Tháng", 1, 12, 6, key="k_month")
    weekday = c2.selectbox("Thứ", WEEKDAYS, key="k_weekday")

    if st.button("Xác định cụm", key="k_btn"):
        row = pd.DataFrame([{
            "Quantity": quantity, "Price": price, "Month": month, "Product Name": product,
            "Store Location": location, "Source": source, "Weekday": weekday,
        }])[km_bundle["features"]]
        cluster = int(km_bundle["kmeans"].predict(km_bundle["preprocessor"].transform(row))[0])
        st.metric("Cụm", f"{cluster} - {CLUSTER_NAMES[cluster]}")

#Tab4: Visualization
with tab4:
    try:
        df = load_cluster_data()
    except FileNotFoundError:
        st.warning("Chưa tìm thấy data/LeDangKhoa_clustered.csv")
    else:
        choice = st.radio("Nhãn cụm", ["Không PCA", "Có PCA"], horizontal=True)
        col = "Cluster_Label" if choice == "Không PCA" else "Cluster_Label_PCA"

        left, right = st.columns([3, 2])
        
        with left:
            st.subheader("Các cụm trên PC1 - PC2")
            sample = df.sample(min(10000, len(df)), random_state=42)
            fig, ax = plt.subplots(figsize=(5, 3.4))
            sc = ax.scatter(sample["PC1"], sample["PC2"], c=sample[col], cmap="tab10", s=5, alpha=0.5)
            ax.set_title(f"K-Means ({choice}) trên 10.000 điểm mẫu", fontsize=10)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.legend(*sc.legend_elements(), title="Cụm")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            st.image(buf, width=480)
        
        with right:
            st.subheader("Đặc điểm từng cụm (không PCA)")
            profile = df.groupby("Cluster_Label")[["Quantity", "Price", "Sales Amount"]].mean().round(2)
            profile.insert(0, "Số giao dịch", df['Cluster_Label'].value_counts().sort_index())
            st.dataframe(profile)

        pca_image = BASE / "images" / "pca_variance.png"
        if pca_image.exists():
            st.subheader("Phương sai tích lũy của PCA")
            st.image(str(pca_image))