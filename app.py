# Version 1.0.1 - Live Fix
import streamlit as st
import pandas as pd
import joblib
import spacy
import os
from utils.preprocess import clean_text

# ---------------------------
# Load models with caching
# ---------------------------
@st.cache_resource
def load_models():
    sentiment_model = joblib.load(os.getenv("SENTIMENT_MODEL", "final_pipeline.joblib"))
    classification_model = joblib.load(os.getenv("CLASS_MODEL", "text_classification_model.joblib"))
    classification_vectorizer = joblib.load(os.getenv("VECTORIZER_MODEL", "text_classification_vectorizer.joblib"))
    nlp = spacy.load("en_core_web_sm")

    return sentiment_model, classification_model, classification_vectorizer, nlp


sentiment_model, classification_model, classification_vectorizer, nlp = load_models()

# ---------------------------
# Mapping
# ---------------------------
classification_mapping = {
    0: 'Teaching',
    1: 'Infrastructure',
    2: 'Exams',
    3: 'Placements',
    -1: 'Miscellaneous'
}

# ---------------------------
# Prediction Functions
# ---------------------------
def predict_sentiment(text):
    text = clean_text(text)
    prediction = sentiment_model.predict([text])[0]

    if prediction == 1:
        return 'Positive'
    elif prediction == -1:
        return 'Negative'
    else:
        return 'Neutral'


def predict_classification(text):
    text = clean_text(text)
    text_vectorized = classification_vectorizer.transform([text])
    prediction = classification_model.predict(text_vectorized)
    return classification_mapping.get(prediction[0], "Unknown")


def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.title("🚀 NLP Analytics Dashboard")

    # ---------------------------
    # Single Text Prediction
    # ---------------------------
    st.subheader("🔍 Try Single Text Prediction")

    user_input = st.text_area("Enter text here:")

    if st.button("Predict"):
        if user_input:
            sentiment = predict_sentiment(user_input)
            category = predict_classification(user_input)
            entities = extract_entities(user_input)

            st.success(f"Sentiment: {sentiment}")
            st.info(f"Category: {category}")
            st.write(f"Entities: {entities}")
        else:
            st.warning("Please enter text")

    st.divider()

    # ---------------------------
    # CSV Upload Section
    # ---------------------------
    uploaded_file = st.file_uploader("📂 Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("📄 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        column_name = st.selectbox("Select Text Column", df.columns)

        sentiment_type = st.selectbox(
            "Filter by Sentiment",
            ['All', 'Positive', 'Negative', 'Neutral']
        )

        classification_category = st.selectbox(
            "Filter by Category",
            ['All', 'Teaching', 'Infrastructure', 'Exams', 'Placements', 'Miscellaneous']
        )

        if st.button("Analyze NLP Pipeline"):

            with st.spinner("Analyzing data..."):

                df['Predicted Sentiment'] = df[column_name].apply(predict_sentiment)
                df['Predicted Category'] = df[column_name].apply(predict_classification)
                df['Named Entities'] = df[column_name].apply(extract_entities)

            # ---------------------------
            # Metrics
            # ---------------------------
            st.subheader("📊 Summary")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Records", len(df))
            col2.metric("Positive", (df['Predicted Sentiment'] == 'Positive').sum())
            col3.metric("Negative", (df['Predicted Sentiment'] == 'Negative').sum())

            # ---------------------------
            # Charts
            # ---------------------------
            st.subheader("📈 Sentiment Distribution")
            st.bar_chart(df['Predicted Sentiment'].value_counts())

            st.subheader("📊 Category Distribution")
            st.bar_chart(df['Predicted Category'].value_counts())

            # ---------------------------
            # Filtering
            # ---------------------------
            filtered_df = df.copy()

            if sentiment_type != 'All':
                filtered_df = filtered_df[
                    filtered_df['Predicted Sentiment'] == sentiment_type
                ]

            if classification_category != 'All':
                filtered_df = filtered_df[
                    filtered_df['Predicted Category'] == classification_category
                ]

            # ---------------------------
            # Results Table
            # ---------------------------
            st.subheader("📋 Filtered Results")

            if len(filtered_df) > 0:
                st.dataframe(
                    filtered_df[
                        ['Predicted Sentiment', 'Predicted Category', 'Named Entities', column_name]
                    ],
                    use_container_width=True
                )
            else:
                st.warning("No results found for selected filters.")


if __name__ == "__main__":
    main()