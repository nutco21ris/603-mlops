import streamlit as st
import joblib
import pandas as pd

# Load model (move this before any st calls)
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("reddit_model_pipeline.joblib")

model = load_model()

# Page config
st.set_page_config(
    page_title="Reddit Comment Classifier",
    page_icon="🤖",
    layout="centered"  # changed to centered for better mobile view
)

# Main UI
st.title("Reddit Comment Classifier")
st.write("Enter a comment to predict if it will be removed.")

# Single comment prediction
comment = st.text_area("Enter your comment:", height=100)

if st.button("Predict", use_container_width=True):
    if comment.strip():
        prediction = model.predict_proba([comment])[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Removal Probability",
                f"{prediction[1]:.1%}"
            )
        with col2:
            st.metric(
                "Keep Probability",
                f"{prediction[0]:.1%}"
            )
    else:
        st.warning("Please enter a comment")

# Batch prediction
st.divider()
st.subheader("Batch Prediction")
file = st.file_uploader("Upload CSV file (comments in first column)", type="csv")

if file:
    try:
        df = pd.read_csv(file)
        st.write("Preview:", df.head())
        
        if st.button("Run Batch Prediction", use_container_width=True):
            with st.spinner("Processing..."):
                predictions = model.predict_proba(df.iloc[:, 0].values)
                
                # Prepare results
                results = df.copy()
                results['removal_prob'] = predictions[:, 1]
                results['keep_prob'] = predictions[:, 0]
                
                # Show results
                st.success("Done!")
                st.dataframe(results)
                
                # Download option
                st.download_button(
                    "Download Results",
                    results.to_csv(index=False),
                    "predictions.csv",
                    "text/csv"
                )
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

st.divider()
st.caption("Reddit Comment Classifier • Built with Streamlit") 