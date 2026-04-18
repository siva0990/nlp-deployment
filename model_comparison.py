import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.preprocess import clean_text

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv("D:/Final_year_project/Twitter_Data.csv")

# Clean text column
df['clean_text'] = df['clean_text'].fillna('')
df = df.dropna(subset=['category'])

# Features & labels
X = df['clean_text']
y = df['category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Define Pipeline Models
# ---------------------------
models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000))
    ]),
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", MultinomialNB())
    ]),
    "Random Forest": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
}
df['clean_text'] = df['clean_text'].fillna('').apply(clean_text)
# ---------------------------
# Train & Evaluate
# ---------------------------
results = []
best_model = None
best_f1 = 0

for name, model in models.items():
    model.fit(X_train, y_train)   # No manual vectorization
    y_pred = model.predict(X_test)

    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    prec = round(precision_score(y_test, y_pred, average="weighted"), 2)
    rec = round(recall_score(y_test, y_pred, average="weighted"), 2)
    f1 = round(f1_score(y_test, y_pred, average="weighted"), 2)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })

    # Track best model using F1-score
    if f1 > best_f1:
        best_f1 = f1
        best_model = model

# ---------------------------
# Results
# ---------------------------
df_results = pd.DataFrame(results)
print(df_results)

# Save results
df_results.to_csv("model_comparison.csv", index=False)

# ---------------------------
# Train Best Model on FULL data
# ---------------------------
print(f"\nBest Model Selected based on F1-score: {best_model}")

best_model.fit(X, y)

# Save final pipeline model
joblib.dump(best_model, "final_pipeline.joblib")

print("\n✅ Final pipeline model saved as 'final_pipeline.joblib'")