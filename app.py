import json
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

API_BASE_URL = st.sidebar.text_input("Backend URL", value="http://127.0.0.1:8000")

st.set_page_config(
    page_title="Loan Approval Intelligence",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top right, rgba(59,130,246,0.16), transparent 30%),
                radial-gradient(circle at top left, rgba(16,185,129,0.10), transparent 25%),
                linear-gradient(180deg, #07111d 0%, #0b1727 100%);
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.5rem;
}
.hero {
    padding: 1.6rem 1.9rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(17,24,39,0.90));
    border: 1px solid rgba(148,163,184,0.18);
    box-shadow: 0 20px 60px rgba(0,0,0,0.30);
    margin-bottom: 1rem;
}
.metric-card {
    padding: 1rem 1.1rem;
    border-radius: 18px;
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(148,163,184,0.15);
}
.section-card {
    padding: 1.2rem;
    border-radius: 18px;
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid rgba(148,163,184,0.12);
    margin-top: 0.8rem;
}
.small-muted { color: #94a3b8; font-size: 0.95rem; }
h1, h2, h3, label, p, li { color: white !important; }
.stTabs [data-baseweb="tab"] { font-size: 15px; }
</style>
""", unsafe_allow_html=True)

def api_get(path):
    resp = requests.get(f"{API_BASE_URL}{path}", timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(resp.text)
    return resp.json()

def api_post(path, files=None, json_body=None):
    resp = requests.post(f"{API_BASE_URL}{path}", files=files, json=json_body, timeout=180)
    if resp.status_code != 200:
        raise RuntimeError(resp.text)
    return resp.json()

st.markdown("""
<div class="hero">
    <h1 style="margin:0;color:white;">Loan Approval Intelligence Platform</h1>
    <p style="margin-top:0.6rem;color:#cbd5e1;">
        Premium Phase 1 + 2 + 3 + 4 workspace for upload, preprocessing, EDA,
        model training, prediction, batch scoring, downloadable outputs, and history.
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Overview", "Upload Dataset", "Dataset Summary", "Preprocess Dataset",
    "EDA Dashboard", "Model Training", "Prediction", "Batch Prediction", "History & Downloads"
])

with tab1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="metric-card"><h3 style="margin:0;">13</h3><div class="small-muted">Expected Columns</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="metric-card"><h3 style="margin:0;">614</h3><div class="small-muted">Known Rows</div></div>', unsafe_allow_html=True)
    c3.markdown('<div class="metric-card"><h3 style="margin:0;">3</h3><div class="small-muted">Core Models</div></div>', unsafe_allow_html=True)
    c4.markdown('<div class="metric-card"><h3 style="margin:0;">Batch + History</h3><div class="small-muted">Phase 4 Added</div></div>', unsafe_allow_html=True)
    st.markdown("### Included so far")
    st.write("- Data upload and schema validation")
    st.write("- Summary and missing-value reporting")
    st.write("- Preprocessing with feature engineering")
    st.write("- Full EDA analysis")
    st.write("- Model training, evaluation, comparison, and single prediction")
    st.write("- Batch prediction, downloadable artifacts, model history, and prediction history")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload loan dataset CSV", type=["csv"])
    if uploaded is not None and st.button("Send to Backend", type="primary"):
        with st.spinner("Uploading dataset..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
            try:
                data = api_post("/upload-data", files=files)
                st.success(data["message"])
                st.session_state["dataset_id"] = data["dataset_id"]
                st.json(data)
            except Exception as exc:
                st.error(str(exc))
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    dataset_id = st.number_input("Dataset ID", min_value=1, step=1, value=int(st.session_state.get("dataset_id", 1)))
    c1, c2 = st.columns(2)
    if c1.button("Fetch Summary", type="primary"):
        try:
            st.session_state["summary_data"] = api_get(f"/summary/{dataset_id}")
        except Exception as exc:
            st.error(str(exc))
    if c2.button("Fetch Missing Values"):
        try:
            st.session_state["missing_data"] = api_get(f"/missing-values/{dataset_id}")
        except Exception as exc:
            st.error(str(exc))

    if "summary_data" in st.session_state:
        data = st.session_state["summary_data"]
        a, b, c = st.columns(3)
        a.metric("Rows", data["shape"]["rows"])
        b.metric("Columns", data["shape"]["columns"])
        c.metric("Dataset ID", data["dataset_id"])
        st.subheader("Columns")
        st.write(data["columns"])
        st.subheader("Data Types")
        st.dataframe(pd.DataFrame([{"column": k, "dtype": v} for k, v in data["dtypes"].items()]), use_container_width=True)
        st.subheader("Target Distribution")
        target_df = pd.DataFrame(list(data["target_distribution"].items()), columns=["Loan_Status", "Count"])
        st.dataframe(target_df, use_container_width=True)
        if not target_df.empty:
            st.plotly_chart(px.bar(target_df, x="Loan_Status", y="Count", title="Loan Status Distribution"), use_container_width=True)
        st.subheader("Numerical Summary")
        st.dataframe(pd.DataFrame(data["numerical_summary"]), use_container_width=True)

    if "missing_data" in st.session_state:
        data = st.session_state["missing_data"]
        miss_df = pd.DataFrame({
            "Column": list(data["missing_counts"].keys()),
            "Missing Count": list(data["missing_counts"].values()),
            "Missing %": list(data["missing_percentages"].values())
        }).sort_values("Missing Count", ascending=False)
        st.subheader("Missing Value Report")
        st.dataframe(miss_df, use_container_width=True)
        st.plotly_chart(px.bar(miss_df, x="Column", y="Missing Count", title="Missing Values by Column"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    dataset_id_pre = st.number_input("Dataset ID for Preprocessing", min_value=1, step=1, value=int(st.session_state.get("dataset_id", 1)), key="dataset_id_pre")
    if st.button("Run Preprocessing", type="primary"):
        try:
            st.session_state["preprocess_data"] = api_post(f"/preprocess/{dataset_id_pre}")
            st.success(st.session_state["preprocess_data"]["message"])
        except Exception as exc:
            st.error(str(exc))

    if "preprocess_data" in st.session_state:
        data = st.session_state["preprocess_data"]
        left, right = st.columns(2)
        left.metric("Original Rows", data["original_shape"]["rows"])
        left.metric("Original Columns", data["original_shape"]["columns"])
        right.metric("Processed Rows", data["processed_shape"]["rows"])
        right.metric("Processed Columns", data["processed_shape"]["columns"])
        st.subheader("Engineered Features")
        st.write(data["engineered_features"])
        st.subheader("Dropped Columns")
        st.write(data["dropped_columns"] or ["None"])
        processed_download = f"{API_BASE_URL}/download/processed/{dataset_id_pre}"
        st.markdown(f"[Download Processed CSV]({processed_download})")
    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    eda_dataset_id = st.number_input("Dataset ID for EDA", min_value=1, step=1, value=int(st.session_state.get("dataset_id", 1)), key="eda_dataset_id")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Load Basic EDA"):
        try:
            st.session_state["eda_basic"] = api_get(f"/eda/basic/{eda_dataset_id}")["data"]
        except Exception as exc:
            st.error(str(exc))
    if c2.button("Load Univariate"):
        try:
            st.session_state["eda_uni"] = api_get(f"/eda/univariate/{eda_dataset_id}")["data"]
        except Exception as exc:
            st.error(str(exc))
    if c3.button("Load Bivariate"):
        try:
            st.session_state["eda_bi"] = api_get(f"/eda/bivariate/{eda_dataset_id}")["data"]
        except Exception as exc:
            st.error(str(exc))
    if c4.button("Load Advanced EDA"):
        try:
            st.session_state["eda_multi"] = api_get(f"/eda/multivariate/{eda_dataset_id}")["data"]
            st.session_state["eda_class"] = api_get(f"/eda/class-imbalance/{eda_dataset_id}")["data"]
            st.session_state["eda_out"] = api_get(f"/eda/outliers/{eda_dataset_id}")["data"]
            st.session_state["eda_corr"] = api_get(f"/eda/correlation/{eda_dataset_id}")["data"]
            st.session_state["eda_find"] = api_get(f"/eda/findings/{eda_dataset_id}")["data"]
        except Exception as exc:
            st.error(str(exc))

    if "eda_basic" in st.session_state:
        data = st.session_state["eda_basic"]
        st.subheader("Basic Inspection")
        aa, bb = st.columns(2)
        aa.metric("Rows", data["shape"]["rows"])
        bb.metric("Columns", data["shape"]["columns"])
        st.write("Columns:", data["columns"])
        st.write("Data Types")
        st.dataframe(pd.DataFrame([{"column": k, "dtype": v} for k, v in data["dtypes"].items()]), use_container_width=True)
        st.write("Head")
        st.dataframe(pd.DataFrame(data["head"]), use_container_width=True)

    if "eda_uni" in st.session_state:
        data = st.session_state["eda_uni"]
        st.subheader("Univariate Analysis")
        for col, counts in data.get("categorical", {}).items():
            df_plot = pd.DataFrame(list(counts.items()), columns=[col, "Count"])
            st.plotly_chart(px.bar(df_plot, x=col, y="Count", title=f"{col} Distribution"), use_container_width=True)
        num = data.get("numerical", {})
        selected_num = st.selectbox("Select numerical feature", options=list(num.keys()), key="num_feature_select") if num else None
        if selected_num:
            vals = num[selected_num]["histogram_bins"]
            if vals:
                st.plotly_chart(px.histogram(pd.DataFrame({selected_num: vals}), x=selected_num, nbins=30, title=f"{selected_num} Histogram"), use_container_width=True)
                stats_view = {k:v for k,v in num[selected_num].items() if k != "histogram_bins"}
                st.dataframe(pd.DataFrame([stats_view]), use_container_width=True)

    if "eda_bi" in st.session_state:
        data = st.session_state["eda_bi"]
        st.subheader("Bivariate Analysis")
        for key, value in data.items():
            if key.endswith("_vs_Loan_Status"):
                df_plot = pd.DataFrame(value).fillna(0)
                st.plotly_chart(px.bar(df_plot, barmode="group", title=key.replace("_", " ")), use_container_width=True)
            elif key.endswith("_mean_by_Loan_Status"):
                df_plot = pd.DataFrame(list(value.items()), columns=["Loan_Status", "Mean"])
                st.plotly_chart(px.bar(df_plot, x="Loan_Status", y="Mean", title=key.replace("_", " ")), use_container_width=True)
            elif key == "income_vs_loanamount_scatter":
                df_plot = pd.DataFrame(value)
                if not df_plot.empty:
                    st.plotly_chart(px.scatter(df_plot, x="ApplicantIncome", y="LoanAmount", title="Applicant Income vs Loan Amount"), use_container_width=True)

    if "eda_multi" in st.session_state:
        st.subheader("Multivariate Analysis")
        data = st.session_state["eda_multi"]
        if "income_loan_credit_vs_approval" in data:
            df_plot = pd.DataFrame(data["income_loan_credit_vs_approval"])
            if not df_plot.empty:
                st.plotly_chart(px.scatter_3d(df_plot, x="TotalIncome", y="LoanAmount", z="Credit_History", color="Loan_Status", title="Income + Loan Amount + Credit History vs Approval"), use_container_width=True)
        if "dependents_married_income_vs_approval" in data:
            df_plot = pd.DataFrame(data["dependents_married_income_vs_approval"])
            if not df_plot.empty:
                st.plotly_chart(px.scatter(df_plot, x="Dependents", y="ApplicantIncome", color="Loan_Status", symbol="Married", title="Dependents + Married + Income vs Approval"), use_container_width=True)

    if "eda_class" in st.session_state:
        data = st.session_state["eda_class"]
        st.subheader("Class Imbalance Check")
        df_plot = pd.DataFrame({"Loan_Status": list(data["counts"].keys()), "Count": list(data["counts"].values()), "Percentage": list(data["percentages"].values())})
        st.plotly_chart(px.pie(df_plot, names="Loan_Status", values="Count", title="Loan Status Class Balance"), use_container_width=True)
        st.dataframe(df_plot, use_container_width=True)

    if "eda_out" in st.session_state:
        st.subheader("Outlier Analysis")
        for col, info in st.session_state["eda_out"].items():
            df_plot = pd.DataFrame({col: info["values"]})
            st.plotly_chart(px.box(df_plot, y=col, title=f"{col} Outlier Boxplot"), use_container_width=True)
            st.dataframe(pd.DataFrame([{
                "Feature": col, "Q1": info["q1"], "Q3": info["q3"], "IQR": info["iqr"],
                "Lower Bound": info["lower_bound"], "Upper Bound": info["upper_bound"], "Outlier Count": info["outlier_count"]
            }]), use_container_width=True)

    if "eda_corr" in st.session_state:
        st.subheader("Correlation Analysis")
        corr = pd.DataFrame(st.session_state["eda_corr"]["matrix"])
        if not corr.empty:
            st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix"), use_container_width=True)

    if "eda_find" in st.session_state:
        st.subheader("EDA Findings")
        for item in st.session_state["eda_find"]["findings"]:
            st.write(f"- {item}")
    st.markdown('</div>', unsafe_allow_html=True)

with tab6:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    train_dataset_id = st.number_input("Dataset ID for Model Training", min_value=1, step=1, value=int(st.session_state.get("dataset_id", 1)), key="train_dataset_id")
    if st.button("Train and Compare Models", type="primary"):
        try:
            st.session_state["train_data"] = api_post(f"/train-models/{train_dataset_id}")
            st.success(f"Training completed. Best model: {st.session_state['train_data']['best_model_name']}")
        except Exception as exc:
            st.error(str(exc))

    if "train_data" in st.session_state:
        train_data = st.session_state["train_data"]
        st.subheader("Best Model")
        a, b = st.columns(2)
        a.metric("Best Model", train_data["best_model_name"])
        best_metrics = train_data["metrics"][train_data["best_model_name"]]
        b.metric("Best Model F1", best_metrics["f1_score"])

        rows = []
        for model_name, m in train_data["metrics"].items():
            rows.append({
                "Model": model_name,
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1 Score": m["f1_score"],
                "ROC AUC": m["roc_auc"],
                "CV F1 Mean": m["cross_validation_f1_mean"],
                "CV F1 Std": m["cross_validation_f1_std"],
            })
        comp_df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)
        st.subheader("Model Comparison")
        st.dataframe(comp_df, use_container_width=True)
        st.plotly_chart(px.bar(comp_df, x="Model", y=["Accuracy", "Precision", "Recall", "F1 Score"], barmode="group", title="Model Metrics Comparison"), use_container_width=True)

        selected_model = st.selectbox("Select model for deeper analysis", options=list(train_data["metrics"].keys()))
        m = train_data["metrics"][selected_model]
        st.subheader(f"{selected_model} Confusion Matrix")
        cm = pd.DataFrame(m["confusion_matrix"], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        st.dataframe(cm, use_container_width=True)
        st.plotly_chart(px.imshow(cm, text_auto=True, aspect="auto", title=f"{selected_model} Confusion Matrix"), use_container_width=True)

        st.subheader(f"{selected_model} Classification Report")
        rep = pd.DataFrame(m["classification_report"]).transpose()
        st.dataframe(rep, use_container_width=True)

        fi = m.get("feature_importance", {})
        if fi:
            st.subheader(f"{selected_model} Feature Importance")
            fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=False)
            st.dataframe(fi_df, use_container_width=True)
            st.plotly_chart(px.bar(fi_df, x="Feature", y="Importance", title=f"{selected_model} Top Features"), use_container_width=True)

        st.markdown(f"[Download Best Model Artifact]({API_BASE_URL}/download/model/{train_dataset_id})")
    st.markdown('</div>', unsafe_allow_html=True)

with tab7:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    pred_dataset_id = st.number_input("Dataset ID for Prediction", min_value=1, step=1, value=int(st.session_state.get("dataset_id", 1)), key="pred_dataset_id")

    c1, c2, c3 = st.columns(3)
    gender = c1.selectbox("Gender", ["Male", "Female"])
    married = c2.selectbox("Married", ["Yes", "No"])
    dependents = c3.selectbox("Dependents", ["0", "1", "2", "3+"])

    c4, c5, c6 = st.columns(3)
    education = c4.selectbox("Education", ["Graduate", "Not Graduate"])
    self_emp = c5.selectbox("Self Employed", ["Yes", "No"])
    property_area = c6.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    n1, n2, n3 = st.columns(3)
    applicant_income = n1.number_input("Applicant Income", min_value=0.0, value=5000.0, step=100.0)
    coapplicant_income = n2.number_input("Coapplicant Income", min_value=0.0, value=0.0, step=100.0)
    loan_amount = n3.number_input("Loan Amount", min_value=0.0, value=120.0, step=1.0)

    n4, n5 = st.columns(2)
    loan_term = n4.number_input("Loan Amount Term", min_value=0.0, value=360.0, step=12.0)
    credit_history = n5.selectbox("Credit History", [1.0, 0.0])

    if st.button("Predict Loan Status", type="primary"):
        payload = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_emp,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area
        }
        try:
            st.session_state["prediction_data"] = api_post(f"/predict/{pred_dataset_id}", json_body=payload)
            st.success("Prediction completed.")
        except Exception as exc:
            st.error(str(exc))

    if "prediction_data" in st.session_state:
        pred = st.session_state["prediction_data"]
        a, b, c = st.columns(3)
        a.metric("Best Model", pred["best_model_name"])
        b.metric("Prediction", pred["prediction_label"])
        c.metric("Approval Probability", pred["probability_approved"] if pred["probability_approved"] is not None else "N/A")
        st.subheader("Processed Input Sent to Model")
        st.dataframe(pd.DataFrame([pred["processed_input"]]), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab8:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    batch_dataset_id = st.number_input("Dataset ID for Batch Prediction", min_value=1, step=1, value=int(st.session_state.get("dataset_id", 1)), key="batch_dataset_id")
    batch_file = st.file_uploader("Upload batch CSV for prediction", type=["csv"], key="batch_file")
    if batch_file is not None and st.button("Run Batch Prediction", type="primary"):
        with st.spinner("Scoring batch file..."):
            files = {"file": (batch_file.name, batch_file.getvalue(), "text/csv")}
            try:
                st.session_state["batch_data"] = api_post(f"/predict-batch/{batch_dataset_id}", files=files)
                st.success("Batch prediction completed.")
            except Exception as exc:
                st.error(str(exc))

    if "batch_data" in st.session_state:
        data = st.session_state["batch_data"]
        a, b, c = st.columns(3)
        a.metric("Best Model", data["best_model_name"])
        b.metric("Total Rows", data["total_rows"])
        c.metric("Dataset ID", data["dataset_id"])
        st.subheader("Batch Prediction Preview")
        st.dataframe(pd.DataFrame(data["preview"]), use_container_width=True)
        label_counts = pd.DataFrame(data["preview"])["Prediction_Label"].value_counts().reset_index() if data["preview"] else pd.DataFrame()
        if not label_counts.empty:
            label_counts.columns = ["Prediction_Label", "Count"]
            st.plotly_chart(px.pie(label_counts, names="Prediction_Label", values="Count", title="Preview Prediction Mix"), use_container_width=True)
        st.markdown(f"[Download Batch Prediction CSV]({API_BASE_URL}/download/batch/{batch_dataset_id})")
    st.markdown('</div>', unsafe_allow_html=True)

with tab9:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    hist_dataset_id = st.number_input("Dataset ID for History", min_value=1, step=1, value=int(st.session_state.get("dataset_id", 1)), key="hist_dataset_id")
    c1, c2 = st.columns(2)
    if c1.button("Load Model History"):
        try:
            st.session_state["model_history"] = api_get(f"/history/models/{hist_dataset_id}")["records"]
        except Exception as exc:
            st.error(str(exc))
    if c2.button("Load Prediction History"):
        try:
            st.session_state["prediction_history"] = api_get(f"/history/predictions/{hist_dataset_id}")["records"]
        except Exception as exc:
            st.error(str(exc))

    if "model_history" in st.session_state:
        st.subheader("Model Run History")
        model_hist_df = pd.DataFrame(st.session_state["model_history"])
        st.dataframe(model_hist_df, use_container_width=True)
        if not model_hist_df.empty:
            st.plotly_chart(px.line(model_hist_df, x="created_at", y="best_model_f1", color="best_model_name", markers=True, title="Model F1 Over Time"), use_container_width=True)

    if "prediction_history" in st.session_state:
        st.subheader("Prediction History")
        pred_hist_df = pd.DataFrame(st.session_state["prediction_history"])
        st.dataframe(pred_hist_df, use_container_width=True)
        if not pred_hist_df.empty and "prediction" in pred_hist_df.columns:
            counts = pred_hist_df["prediction"].astype(str).value_counts().reset_index()
            counts.columns = ["Prediction", "Count"]
            st.plotly_chart(px.bar(counts, x="Prediction", y="Count", title="Prediction History Distribution"), use_container_width=True)

    st.markdown("### Quick Downloads")
    st.markdown(f"- [Download Processed Dataset]({API_BASE_URL}/download/processed/{hist_dataset_id})")
    st.markdown(f"- [Download Best Model]({API_BASE_URL}/download/model/{hist_dataset_id})")
    st.markdown(f"- [Download Latest Batch Prediction CSV]({API_BASE_URL}/download/batch/{hist_dataset_id})")
    st.markdown('</div>', unsafe_allow_html=True)
