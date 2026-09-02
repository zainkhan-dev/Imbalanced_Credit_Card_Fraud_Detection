import io
import joblib
import pandas as pd
import streamlit as st

# --- Global Schema Definition (30 Total Features) ---
FEATURE_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

st.set_page_config(
    page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide"
)


# --- Load Model & Scaler Artifacts ---
@st.cache_resource
def load_artifacts():
    model = joblib.load("baseline_logistic_regression.joblib")
    amount_scaler = joblib.load("amount_scaler.joblib")
    time_scaler = joblib.load("time_scaler.joblib")
    return model, amount_scaler, time_scaler


try:
    model, amount_scaler, time_scaler = load_artifacts()
except Exception as e:
    st.error(
        f"Failed to load model artifacts. Ensure 'baseline_logistic_regression.joblib', 'amount_scaler.joblib', and 'time_scaler.joblib' exist: {e}"
    )
    st.stop()


# --- Centralized Preprocessing & Scoring Helper ---
def predict_fraud(df, model, amount_scaler, time_scaler, threshold):
    """Scale raw Time and Amount values before model prediction."""
    df_processed = df[FEATURE_COLS].copy()

    df_processed["Amount"] = amount_scaler.transform(df_processed[["Amount"]])
    df_processed["Time"] = time_scaler.transform(df_processed[["Time"]])

    probas = model.predict_proba(df_processed)[:, 1]
    flags = (probas >= threshold).astype(int)
    return probas, flags


# --- Sidebar Configuration ---
st.sidebar.header("Model Configuration")
threshold = st.sidebar.slider(
    "Decision Threshold (T)",
    min_value=0.01,
    max_value=0.90,
    value=0.17,
    step=0.01,
    help="Optimized T=0.17 maximizes F1-Score under natural class imbalance.",
)

with st.sidebar.expander("ℹ️ How Decision Thresholding Works"):
    st.markdown(
        """
        * **Default Probability (0.50):** Standard classifiers use $T=0.50$, which misses high-risk fraud cases in heavily imbalanced datasets.
        * **Calibrated Threshold ($T=0.17$):** Lowering $T$ increases recall (catching more fraud) while maintaining acceptable precision.
        """
    )

# --- Main App Interface ---
st.title("💳 Credit Card Fraud Detection Platform")
st.caption(
    "Baseline Logistic Regression — Production Pipeline with Calibrated Feature Scaling"
)

tab1, tab2 = st.tabs(["⚡ Single Sample Demo", "📁 CSV Batch Scoring"])

# --- TAB 1: Single Sample Scoring ---
with tab1:
    st.subheader("Interactive Sample Scoring (Real Test Set Rows)")

    try:
        sample_df = pd.read_csv("sample_transactions.csv")
        sample_labels = [
            f"Sample #{idx+1} ({row['label']}) - Amount: ${row['Amount']:.2f}"
            for idx, row in sample_df.iterrows()
        ]

        selected_idx = st.selectbox(
            "Select Test Scenario:",
            range(len(sample_labels)),
            format_func=lambda x: sample_labels[x],
        )

        selected_row = sample_df.iloc[[selected_idx]]
        true_label = selected_row["label"].values[0]

        if st.button("Score Selected Transaction", type="primary"):
            probas, flags = predict_fraud(
                selected_row, model, amount_scaler, time_scaler, threshold
            )
            proba = float(probas[0])
            is_flagged = bool(flags[0])

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Fraud Probability", f"{proba:.4f}")
            col2.metric("Operating Threshold", f"{threshold:.2f}")
            col3.metric("True Label in Dataset", true_label)

            if is_flagged:
                col4.error("🚨 FLAGGED FOR REVIEW")
            else:
                col4.success("✅ APPROVED")

    except FileNotFoundError:
        st.warning(
            "Please export 'sample_transactions.csv' from your notebook to run the interactive demo."
        )

# --- TAB 2: Batch CSV Upload with Explicit Validation ---
with tab2:
    st.subheader("Batch Transaction Scoring")
    uploaded_file = st.file_uploader(
        "Upload CSV file containing Time, V1-V28, and Amount", type=["csv"]
    )

    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        missing_cols = set(FEATURE_COLS) - set(df_batch.columns)

        # Explicit schema validation handling
        if missing_cols:
            st.error(
                f"Uploaded CSV is missing required columns: {sorted(list(missing_cols))}"
            )
        else:
            probas, flags = predict_fraud(
                df_batch, model, amount_scaler, time_scaler, threshold
            )

            df_batch["Fraud_Probability"] = probas
            df_batch["Flagged_Fraud"] = flags

            st.dataframe(df_batch.head(10))

            flagged_total = int(flags.sum())
            st.info(
                f"Scored {len(df_batch)} transactions. Total Flagged at T={threshold:.2f}: **{flagged_total}**"
            )

            # File Download Buffer
            csv_buffer = io.BytesIO()
            df_batch.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Scored Predictions CSV",
                data=csv_buffer.getvalue(),
                file_name="scored_predictions.csv",
                mime="text/csv",
            )
