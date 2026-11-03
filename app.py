
import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import os
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================
# 1. CẤU HÌNH WEBSITE
# =========================================

st.set_page_config(
    page_title="Dự đoán kết quả học tập",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 HỆ THỐNG DỰ ĐOÁN KẾT QUẢ HỌC TẬP")

st.write(
    "Sử dụng Logistic Regression và Decision Tree "
    "để dự đoán học sinh Đạt hoặc Chưa đạt."
)

st.markdown("---")


# =========================================
# 2. TẢI DỮ LIỆU TỪ KAGGLE
# =========================================

@st.cache_data
def load_data():

    path = kagglehub.dataset_download(
        "spscientist/students-performance-in-exams"
    )

    csv_files = glob.glob(
        os.path.join(path, "**", "*.csv"),
        recursive=True
    )

    file_path = next(
        (
            f for f in csv_files
            if os.path.basename(f).lower()
            == "studentsperformance.csv"
        ),
        csv_files[0] if csv_files else None
    )

    if file_path is None:
        raise FileNotFoundError("Không tìm thấy file CSV!")

    return pd.read_csv(file_path)


try:
    df = load_data()

except Exception as e:
    st.error(f"Lỗi tải dữ liệu: {e}")
    st.stop()


# =========================================
# 3. TIỀN XỬ LÝ DỮ LIỆU
# =========================================

df = df.drop_duplicates().copy()

features = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course"
]

# Tính điểm trung bình 3 môn
df["Average"] = df[
    ["math score", "reading score", "writing score"]
].mean(axis=1)

# Tạo nhãn kết quả
df["Pass"] = (df["Average"] >= 50).astype(int)

# X là dữ liệu đầu vào
X = df[features]

# Y là kết quả cần dự đoán
y = df["Pass"]


# =========================================
# 4. CHIA DỮ LIỆU TRAIN VÀ TEST
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================
# 5. XÂY DỰNG HAI MÔ HÌNH
# =========================================

@st.cache_resource
def train_models(X_train, y_train):

    # Tiền xử lý dữ liệu cho Logistic Regression
    preprocess_logistic = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                features
            )
        ]
    )

    # Mô hình Logistic Regression
    logistic_model = Pipeline([
        ("preprocessing", preprocess_logistic),
        ("scaler", StandardScaler(with_mean=False)),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    logistic_model.fit(X_train, y_train)

    # Tiền xử lý dữ liệu cho Decision Tree
    preprocess_tree = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                features
            )
        ]
    )

    # Mô hình Decision Tree
    tree_model = Pipeline([
        ("preprocessing", preprocess_tree),
        (
            "classifier",
            DecisionTreeClassifier(
                criterion="gini",
                max_depth=5,
                random_state=42
            )
        )
    ])

    tree_model.fit(X_train, y_train)

    return logistic_model, tree_model


with st.spinner("Đang huấn luyện Logistic Regression và Decision Tree..."):

    logistic_model, tree_model = train_models(
        X_train,
        y_train
    )

st.success("Hai mô hình đã được huấn luyện thành công!")


# =========================================
# 6. TỔNG QUAN DỮ LIỆU
# =========================================

st.header("1. Tổng quan dữ liệu")

col1, col2, col3 = st.columns(3)

col1.metric("Tổng số học sinh", len(df))
col2.metric("Dữ liệu Train", len(X_train))
col3.metric("Dữ liệu Test", len(X_test))

st.subheader("10 dòng dữ liệu đầu tiên")

st.dataframe(
    df.head(10),
    use_container_width=True
)

st.subheader("Kiểm tra dữ liệu thiếu")

st.dataframe(
    df.isnull().sum().rename("Số dữ liệu thiếu"),
    use_container_width=True
)

st.subheader("Phân bố kết quả học tập")

result_count = df["Pass"].value_counts().sort_index()

chart_data = pd.DataFrame({
    "Kết quả": ["Chưa đạt", "Đạt"],
    "Số học sinh": [
        result_count.get(0, 0),
        result_count.get(1, 0)
    ]
})

st.bar_chart(
    chart_data.set_index("Kết quả")
)


# =========================================
# 7. ĐÁNH GIÁ HAI MÔ HÌNH
# =========================================

st.markdown("---")
st.header("2. Đánh giá hiệu quả mô hình")

# Dự đoán trên tập Test
logistic_pred = logistic_model.predict(X_test)
tree_pred = tree_model.predict(X_test)


def evaluate_model(y_true, y_pred):

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1-score": f1_score(
            y_true, y_pred, zero_division=0
        )
    }


logistic_scores = evaluate_model(
    y_test, logistic_pred
)

tree_scores = evaluate_model(
    y_test, tree_pred
)

comparison = pd.DataFrame(
    [logistic_scores, tree_scores],
    index=["Logistic Regression", "Decision Tree"]
)

st.subheader("Bảng so sánh 4 chỉ số")

st.dataframe(
    comparison.style.format("{:.4f}"),
    use_container_width=True
)

st.subheader("Biểu đồ so sánh")

st.bar_chart(comparison)


# Ma trận nhầm lẫn
col1, col2 = st.columns(2)

with col1:

    st.subheader("Ma trận nhầm lẫn Logistic Regression")

    st.dataframe(
        pd.DataFrame(
            confusion_matrix(
                y_test,
                logistic_pred,
                labels=[0, 1]
            ),
            index=["Thực tế: Chưa đạt", "Thực tế: Đạt"],
            columns=["Dự đoán: Chưa đạt", "Dự đoán: Đạt"]
        )
    )

with col2:

    st.subheader("Ma trận nhầm lẫn Decision Tree")

    st.dataframe(
        pd.DataFrame(
            confusion_matrix(
                y_test,
                tree_pred,
                labels=[0, 1]
            ),
            index=["Thực tế: Chưa đạt", "Thực tế: Đạt"],
            columns=["Dự đoán: Chưa đạt", "Dự đoán: Đạt"]
        )
    )


# =========================================
# 8. NHẬP THÔNG TIN HỌC SINH
# =========================================

st.markdown("---")
st.header("3. Nhập thông tin học sinh mới")

st.write(
    "Người dùng chọn thông tin học sinh. "
    "Hệ thống sẽ sử dụng hai thuật toán để dự đoán."
)

with st.form("student_form"):

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Giới tính",
            sorted(df["gender"].dropna().unique())
        )

        race = st.selectbox(
            "Nhóm dân tộc",
            sorted(df["race/ethnicity"].dropna().unique())
        )

        education = st.selectbox(
            "Trình độ học vấn của phụ huynh",
            sorted(
                df["parental level of education"]
                .dropna().unique()
            )
        )

    with col2:

        lunch = st.selectbox(
            "Loại bữa ăn",
            sorted(df["lunch"].dropna().unique())
        )

        preparation = st.selectbox(
            "Khóa học chuẩn bị thi",
            sorted(
                df["test preparation course"]
                .dropna().unique()
            )
        )

    submit = st.form_submit_button(
        "🔍 DỰ ĐOÁN KẾT QUẢ",
        use_container_width=True
    )


# =========================================
# 9. HIỂN THỊ KẾT QUẢ DỰ ĐOÁN
# =========================================

if submit:

    new_student = pd.DataFrame([{
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": education,
        "lunch": lunch,
        "test preparation course": preparation
    }])

    # Dự đoán Logistic Regression
    logistic_result = logistic_model.predict(
        new_student
    )[0]

    # Dự đoán Decision Tree
    tree_result = tree_model.predict(
        new_student
    )[0]

    st.markdown("---")
    st.header("4. Kết quả dự đoán")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 Logistic Regression")

        if logistic_result == 1:
            st.success("KẾT QUẢ: ĐẠT")
        else:
            st.error("KẾT QUẢ: CHƯA ĐẠT")

    with col2:

        st.subheader("🌳 Decision Tree")

        if tree_result == 1:
            st.success("KẾT QUẢ: ĐẠT")
        else:
            st.error("KẾT QUẢ: CHƯA ĐẠT")

    st.subheader("So sánh kết quả")

    if logistic_result == tree_result:

        if logistic_result == 1:
            st.success(
                "Cả hai mô hình đều dự đoán học sinh ĐẠT."
            )
        else:
            st.warning(
                "Cả hai mô hình đều dự đoán học sinh CHƯA ĐẠT."
            )

    else:

        st.warning(
            "Hai mô hình đưa ra kết quả khác nhau. "
            "Bạn nên xem xét thêm thông tin học tập thực tế."
        )

    st.info(
        "Đây là kết quả dự đoán của mô hình, "
        "không phải kết quả học tập chính thức."
    )


# =========================================
# 10. THÔNG TIN CUỐI TRANG
# =========================================

st.markdown("---")

st.caption(
    "Đề tài: Dự đoán kết quả học tập bằng "
    "Logistic Regression và Decision Tree"
)

st.caption(
    "Nguồn dữ liệu: Kaggle - Students Performance in Exams"
)

st.markdown(
    "[Truy cập bộ dữ liệu Kaggle]"
    "(https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)"
)
