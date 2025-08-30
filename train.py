import os
import random
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def load_dataset():
    """
    Try a solid built-in dataset from NLTK.
    If NLTK data isn't available (e.g., offline), fall back to a small seed list.
    """
    csv_path = os.path.join("data", "names_dataset.csv")
    if os.path.exists(csv_path):
        data = pd.read_csv(csv_path)
        # Ensure columns are named correctly
        data = data.rename(columns=lambda x: x.strip().lower())
        if "name" in data.columns and "gender" in data.columns:
            data = data[["name", "gender"]]
            data = data.dropna()
            data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)
            return data
        else:
            raise ValueError("CSV must have 'name' and 'gender' columns.")
    try:
        import nltk
        from nltk.corpus import names
        # Ensure the corpus is available (first time needs download)
        try:
            names.words('male.txt')
        except LookupError:
            nltk.download('names')
        male = [(n, "male") for n in names.words('male.txt')]
        female = [(n, "female") for n in names.words('female.txt')]
        data = pd.DataFrame(male + female, columns=["name", "gender"])
        # Shuffle to avoid any ordering bias
        data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)
        return data
    except Exception:
        # Minimal fallback so you can still run offline
        seed = [
            ("Arun","male"),("Rahul","male"),("Vikram","male"),("Aman","male"),("Ravi","male"),
            ("Karthik","male"),("John","male"),("Michael","male"),("David","male"),("Sanjay","male"),
            ("Priya","female"),("Sneha","female"),("Anjali","female"),("Meena","female"),("Aishwarya","female"),
            ("Neha","female"),("Emma","female"),("Olivia","female"),("Sophia","female"),("Harini","female")
        ]
        data = pd.DataFrame(seed, columns=["name","gender"])
        return data

def build_pipeline():
    """
    Char-level n-gram TF-IDF + Logistic Regression works very well for names.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2,4),   # bi-grams to 4-grams
            lowercase=True,
            min_df=1
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=None
        ))
    ])

def main():
    data = load_dataset()
    X = data["name"]
    y = data["gender"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique())==2 else None
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.3f}\n")
    print("Classification report:\n", classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/name_gender_clf.joblib"
    joblib.dump(pipe, model_path)
    print(f"\n✓ Saved model to {model_path}")

    # Try a few demo names
    demo = ["Aarav","Isha","Rahul","Sneha","Jordan","Taylor","Alex","Harini","Nini"]
    print("\nSample predictions:")
    if hasattr(pipe, "predict_proba"):
        classes = list(pipe.classes_)
        for n in demo:
            proba = pipe.predict_proba([n])[0]
            cm = dict(zip(classes, proba))
            pred = pipe.predict([n])[0]
            conf = cm[pred]
            print(f"  {n:10s} → {pred:6s}  (confidence: {conf:.2%})")
    else:
        for n in demo:
            pred = pipe.predict([n])[0]
            print(f"  {n:10s} → {pred}")

if __name__ == "__main__":
    main()
