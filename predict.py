import sys
import joblib
from pathlib import Path

MODEL_PATH = Path("models/name_gender_clf.joblib")

def load_model():
    if not MODEL_PATH.exists():
        print("Model not found. Run: python train.py")
        sys.exit(1)
    return joblib.load(MODEL_PATH)

def predict_names(names):
    model = load_model()
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        for n in names:
            pred = model.predict([n])[0]
            proba = model.predict_proba([n])[0]
            conf = dict(zip(classes, proba))[pred]
            print(f"{n} → {pred}  (confidence: {conf:.2%})")
    else:
        for n in names:
            pred = model.predict([n])[0]
            print(f"{n} → {pred}")

if __name__ == "__main__":
    # Usage: python predict.py Priya Alex Sam
    if len(sys.argv) > 1:
        predict_names(sys.argv[1:])
    else:
        try:
            while True:
                name = input("Enter a name (or blank to quit): ").strip()
                if not name:
                    break
                predict_names([name])
        except KeyboardInterrupt:
            pass
