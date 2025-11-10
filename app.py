import io
import joblib
import streamlit as st

st.set_page_config(page_title="Text Classification Demo", page_icon="🔤", layout="centered")

# ----------------------------
# Utilities
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline_from_path(path: str):
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_pipeline_from_bytes(bytes_io: io.BytesIO):
    # joblib can load from a file-like object
    return joblib.load(bytes_io)

def get_classes(pipe):
    # Try to infer class labels from pipeline or its final estimator
    # Works for most scikit-learn pipelines
    if hasattr(pipe, "classes_"):
        return pipe.classes_
    try:
        # If it's a Pipeline, final estimator is steps[-1][1]
        final_est = getattr(pipe, "steps", [[None, None]])[-1][1]
        return getattr(final_est, "classes_", None)
    except Exception:
        return None

def safe_predict(pipe, text: str):
    pred = pipe.predict([text])[0]
    classes = get_classes(pipe)
    proba = None
    # Not all models implement predict_proba
    if hasattr(pipe, "predict_proba"):
        try:
            probs = pipe.predict_proba([text])[0]
            proba = list(zip(classes if classes is not None else range(len(probs)), probs))
            # Sort by probability (desc)
            proba.sort(key=lambda x: x[1], reverse=True)
        except Exception:
            proba = None
    return pred, proba, classes

# ----------------------------
# Sidebar: model source
# ----------------------------
st.sidebar.title("Model Settings")
source = st.sidebar.radio(
    "Load model from:",
    options=["Default path", "Upload .joblib"],
    index=0
)

pipe = None
if source == "Default path":
    model_path = st.sidebar.text_input(
        "Model path",
        value="models/lr_tfidf_model.joblib",
        help="Path to your saved *tfidf* pipeline model (.joblib)"
    )
    if st.sidebar.button("Load model"):
        with st.spinner(f"Loading model from {model_path}..."):
            pipe = load_pipeline_from_path(model_path)
else:
    uploaded = st.sidebar.file_uploader("Upload a .joblib model", type=["joblib"])
    if uploaded is not None:
        if st.sidebar.button("Load uploaded model"):
            bytes_io = io.BytesIO(uploaded.read())
            with st.spinner("Loading uploaded model..."):
                pipe = load_pipeline_from_bytes(bytes_io)

# If nothing pressed yet but default path exists, try lazy-load to be friendly
if pipe is None and source == "Default path" and model_path:
    try:
        pipe = load_pipeline_from_path(model_path)
    except Exception:
        pass  # Wait until user clicks "Load model" or fixes path

# ----------------------------
# Main UI
# ----------------------------
st.title("🔤 Text Classification Demo")
st.markdown(
    "Enter text below and click **Predict** to see the model’s label. "
    "You can load a model via the sidebar (default path or upload)."
)

disabled = pipe is None
if disabled:
    st.info("Load a model from the sidebar to begin.")

user_text = st.text_area("Your text", placeholder="Type or paste text here...", height=180, disabled=disabled)

col1, col2 = st.columns([1, 2])
with col1:
    predict_clicked = st.button("Predict", disabled=disabled or not user_text.strip())
with col2:
    clear_clicked = st.button("Clear")

if clear_clicked:
    # Streamlit will just rerun; text_area will clear if you pass a key, but we keep it simple.
    st.experimental_rerun()

if predict_clicked:
    try:
        pred, proba, classes = safe_predict(pipe, user_text.strip())
        st.success(f"**Prediction:** {pred}")

        if proba is not None:
            st.subheader("Class probabilities")
            # Nice table
            st.dataframe(
                {"class": [str(c) for c, _ in proba], "probability": [float(p) for _, p in proba]},
                use_container_width=True
            )
        else:
            st.caption("This model does not expose probability scores (`predict_proba`).")

        if classes is not None:
            st.caption(f"Known classes: {', '.join(map(str, classes))}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

st.markdown("---")
with st.expander("ℹ️ Tips"):
    st.markdown(
        "- Ensure your saved model is a scikit-learn **Pipeline** that includes text preprocessing and a classifier.\n"
        "- If you trained with TF-IDF + Logistic Regression/SVM, saving via `joblib.dump(pipe, path)` is ideal.\n"
        "- Large models can be uploaded via the sidebar if you don’t want to commit them to your repo."
    )
