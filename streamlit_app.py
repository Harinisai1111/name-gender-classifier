import joblib
from pathlib import Path
import streamlit as st

MODEL_PATH = Path("models/name_gender_clf.joblib")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python train.py` first.")
    return joblib.load(MODEL_PATH)

st.set_page_config(page_title="Name Gender Classifier", page_icon="👤")
st.title("👤 Name Gender Classifier")

name = st.text_input("Type a name")
if name:
    model = load_model()
    pred = model.predict([name])[0]
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        proba = model.predict_proba([name])[0]
        conf = dict(zip(classes, proba))[pred]
        st.metric("Prediction", pred, f"confidence {conf:.1%}")
    else:
        st.metric("Prediction", pred)
else:
    st.write("Enter a name above to get a prediction.")

st.caption("Demo model; may reflect cultural/linguistic biases of the training data.")
