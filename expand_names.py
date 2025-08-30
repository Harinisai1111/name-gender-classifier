import pandas as pd
import nltk
nltk.download('names')
from nltk.corpus import names

# Load your existing CSV
df = pd.read_csv("data/names_dataset.csv")

# Get NLTK names
male = [(n, "male") for n in names.words('male.txt')]
female = [(n, "female") for n in names.words('female.txt')]
nltk_df = pd.DataFrame(male + female, columns=["name", "gender"])

# Combine and drop duplicates
combined = pd.concat([df, nltk_df]).drop_duplicates().reset_index(drop=True)

# Save back to CSV
combined.to_csv("data/names_dataset.csv", index=False)