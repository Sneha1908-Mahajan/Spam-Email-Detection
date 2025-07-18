import streamlit as st
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model and vectorizer
@st.cache_resource
def load_model():
    with open('spam_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

@st.cache_resource
def load_vectorizer():
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return vectorizer

model = load_model()
vectorizer = load_vectorizer()

# Initialize session state for prediction history
if 'history' not in st.session_state:
    st.session_state.history = []

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Streamlit app
def main():
    st.title("📧 Email Spam Classifier")
    st.write("This app predicts whether an email is spam or not using a trained machine learning model.")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Classify Email", "Prediction History"])
    
    with tab1:
        # Input options
        input_option = st.radio("Input method:", ("Enter text", "Upload file"))
        
        email_text = ""
        
        if input_option == "Enter text":
            email_text = st.text_area("Enter the email text:", height=200)
        else:
            uploaded_file = st.file_uploader("Upload email file:", type=['txt', 'eml'])
            if uploaded_file is not None:
                email_text = uploaded_file.getvalue().decode("utf-8")
                st.text_area("Email content:", value=email_text, height=200)
        
        if st.button("Predict"):
            if email_text.strip():
                # Preprocess and predict
                processed_text = preprocess_text(email_text)
                text_vector = vectorizer.transform([processed_text]).toarray()
                prediction = model.predict(text_vector)
                
                # Display results
                if prediction[0] == 1:
                    st.error("This email is classified as SPAM.")
                else:
                    st.success("This email is classified as NOT SPAM.")
                
                # Add to history
                st.session_state.history.append({
                    'text': email_text[:100] + "..." if len(email_text) > 100 else email_text,
                    'prediction': "Spam" if prediction[0] == 1 else "Not Spam",
                    'full_text': email_text
                })
            else:
                st.warning("Please enter some email text to analyze.")
    
    with tab2:
        if st.session_state.history:
            st.subheader("Previous Predictions")
            
            # Show recent predictions in a table without the action icons
            history_df = pd.DataFrame(st.session_state.history[::-1])
            st.dataframe(
                history_df[['text', 'prediction']],
                column_config={
                    'text': 'Email Snippet',
                    'prediction': 'Prediction'
                },
                hide_index=True,
                use_container_width=True,
                # disabled=True   This disables the interactive features including the icons
            )
            
            # Option to view details of any prediction
            selected_index = st.selectbox(
                "View details of a previous prediction:",
                range(len(st.session_state.history)),
                format_func=lambda x: f"Prediction {len(st.session_state.history)-x} - {st.session_state.history[::-1][x]['prediction']}"
            )
            
            if selected_index is not None:
                selected_pred = st.session_state.history[::-1][selected_index]
                st.write("Full email text:")
                st.text(selected_pred['full_text'])
                
                # Show the analysis again for this email
                st.write("Analysis:")
                if selected_pred['prediction'] == "Spam":
                    st.error("This email was classified as SPAM.")
                else:
                    st.success("This email was classified as NOT SPAM.")
            
            # Clear history button
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("No prediction history yet. Make some predictions on the 'Classify Email' tab!")

if __name__ == '__main__':
    main()