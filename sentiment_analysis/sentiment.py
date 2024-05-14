# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:46:20 2024

@author: AMMU
"""
import pickle
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load trained models using pickle
with open("C:/Users/amukt/OneDrive/Documents/sentiment_analysis/logistic_count_model.pkl", "rb") as f:
    loj_model_count = pickle.load(f)

with open("C:/Users/amukt/OneDrive/Documents/sentiment_analysis/count_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Function to predict sentiment using loaded models
def predict_count_with_pickle(model, vectorizer, new_comment):
    new_comment = [new_comment]
    new_comment = vectorizer.transform(new_comment)
    result = model.predict(new_comment)
    if result == 1:
        return "The entered Comment is Positive"
    else:
        return "The entered Comment is Negative"

# Create a Streamlit app
def main():
    st.title("Amazon Reviews Sentiment Analysis")
    comment = st.text_input("Enter your review:")
    if st.button("Predict"):
        prediction = predict_count_with_pickle(loj_model_count, vectorizer, comment)
        st.write(prediction)
# Run the Streamlit app
if __name__ == "__main__":
    main()

