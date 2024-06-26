import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Streamlit setup
st.title('Product Review Sentimental Analysis')
st.write('Review output.')

# Load the dataset from a CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

    # Define a function to categorize sentiment based on star rating
    def categorize_sentiment(rating):
        if rating >= 4:
            return 'positive'
        elif rating == 3:
            return 'neutral'
        else:
            return 'negative'

    # Apply the function to the rating column to create a sentiment column
    data['sentiment'] = data['Rating'].apply(categorize_sentiment)
    # Map sentiment to numerical values
    sentiment_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
    data['sentiment_num'] = data['sentiment'].map(sentiment_mapping)
    # Handle missing values in 'Rating' column before splitting
    X = data[['Rating']].fillna('')  # Replace NaN with empty string, or choose another appropriate strategy

    # Convert 'Rating' column to string type
    X['Rating'] = X['Rating'].astype(str)  # Convert all values to strings

    # Convert 'Rating' column to numerical representation (e.g., using one-hot encoding)
    encoder = OneHotEncoder(handle_unknown='ignore')  # Handle potential new categories in test data
    X_encoded = encoder.fit_transform(X)  # Fit and transform on the entire dataset

    # Define your target variable y based on the 'sentiment_num' column
    y = data['sentiment_num']  # Use the correct target variable

    # Split the data into features (X) and labels (y)
    # Use the encoded features for splitting
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)  # Now the model should work with numerical data

    # Predict sentiments
    y_pred = model.predict(X_test)

    # Display unique values in predictions and true labels
    st.write("Unique predicted labels:", np.unique(y_pred))
    st.write("Unique true labels:", np.unique(y_test))

    # Generate the classification report
    labels = np.unique(y_test)
    target_names = [list(sentiment_mapping.keys())[list(sentiment_mapping.values()).index(label)] for label in labels]
    st.write(classification_report(y_test, y_pred, target_names=target_names, labels=labels))

    # Calculate sentiment distribution
    sentiment_counts = data['sentiment'].value_counts()

    # Plot pie chart
    st.subheader('Sentiment Distribution')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=['pink', 'yellow', 'green'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    # Plot bar graph
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(sentiment_counts.index, sentiment_counts.values, color=['red', 'green', 'yellow'])
    ax.set_title('Sentiment Distribution')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Counts')
    st.pyplot(fig)

# To run the Streamlit app, save this script as app.py and run `streamlit run app.py` in your terminal
