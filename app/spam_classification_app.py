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
from wordcloud import WordCloud
import re
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

nltk.download('punkt')
nltk.download('stopwords')

@st.cache_resource
def load_model():
    with open('models/spam_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

@st.cache_resource
def load_vectorizers():
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('models/count_vectorizer.pkl', 'rb') as f:
        count_vec = pickle.load(f)
    return tfidf, count_vec

model = load_model()
tfidf, count_vec = load_vectorizers()

if 'history' not in st.session_state:
    st.session_state.history = []

def transform_messages(text):
    ps = PorterStemmer()
    
    text = text.lower()
    
    text = re.sub(r'[^\w\s\$\%\!\*]', '', text)
    
    text = text.split()
    
    custom_stopwords = set(stopwords.words('english')) - {'free', 'win', 'prize', 'urgent', 'click', 'offer'}
    text = [word for word in text if word not in custom_stopwords]
    
    text = [ps.stem(word) for word in text]
    
    return " ".join(text)

def generate_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=15)
    ax.axis('off')
    st.pyplot(fig)

def plot_prediction_distribution(history):
    if not history:
        return
    
    df = pd.DataFrame(history)
    counts = df['prediction'].value_counts()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', 
           colors=['#4CAF50', '#FF5252'], startangle=90)
    ax.set_title('Your Prediction Results', fontsize=15)
    st.pyplot(fig)

# Streamlit app
def main():
    st.title("📧 Email Spam Classifier")
    st.write("This app predicts whether an email is spam or not using a trained machine learning model.")
    
    tab1, tab2 = st.tabs(["Classify Email", "Prediction History"])
    
    with tab1:
        input_option = st.radio("Input Method", ("Type text", "Upload file"))
        
        email_text = ""
        
        if input_option == "Type text":
            email_text = st.text_area("Enter the email text here:", height=200,
                                    placeholder="Paste the email content you want to check...")
        else:
            uploaded_file = st.file_uploader("Choose an email file:", type=['txt', 'eml'])
            if uploaded_file is not None:
                email_text = uploaded_file.getvalue().decode("utf-8")
                st.text_area("Email content:", value=email_text, height=200)
        
        if st.button("Predict"):
            if email_text.strip():
                transformed_text = transform_messages(email_text)
                
                X_tfidf = tfidf.transform([transformed_text])
                X_keywords = count_vec.transform([transformed_text])
                X_input = sp.hstack([X_tfidf, X_keywords])
                
                prediction = model.predict(X_input)
                prediction_proba = model.predict_proba(X_input)[0]
                
                if prediction[0] == 1:
                    st.error(f"This email is classified as Spam (confidence: {prediction_proba[1]*100:.1f}%)")
                    generate_word_cloud(email_text, "Spam Indicators")
                else:
                    st.success(f"This email is classified as Non-Spam (confidence: {prediction_proba[0]*100:.1f}%)")
                    generate_word_cloud(email_text, "Normal Email Content")

                st.session_state.history.append({
                    'text': email_text[:100] + "..." if len(email_text) > 100 else email_text,
                    'prediction': "Spam" if prediction[0] == 1 else "Not Spam",
                    'confidence': prediction_proba[1] if prediction[0] == 1 else prediction_proba[0],
                    'full_text': email_text
                })
                
            else:
                st.warning("Please enter some email text to check.")
    
    with tab2:
        if st.session_state.history:
            st.subheader("Previous Predictions")
            
            plot_prediction_distribution(st.session_state.history)
            
            history_df = pd.DataFrame(st.session_state.history[::-1])
            st.dataframe(
                history_df[['text', 'prediction', 'confidence']],
                column_config={
                    'text': 'Email Preview',
                    'prediction': 'Result',
                    'confidence': st.column_config.ProgressColumn(
                        "Confidence",
                        format="%.1f%%",
                        min_value=0,
                        max_value=1,
                    )
                },
                hide_index=True,
                use_container_width=True,
            )
            
            selected_index = st.selectbox(
                "View details of a previous predictions:",
                range(len(st.session_state.history)),
                format_func=lambda x: f"Check {len(st.session_state.history)-x} - {st.session_state.history[::-1][x]['prediction']}"
            )
            
            if selected_index is not None:
                selected_pred = st.session_state.history[::-1][selected_index]
                st.write("Full email text:")
                st.text(selected_pred['full_text'])
                
                st.write("Result:")
                if selected_pred['prediction'] == "Spam":
                    st.error(f"Spam detected ({selected_pred['confidence']*100:.1f}% confidence)")
                    generate_word_cloud(selected_pred['full_text'], "Spam Indicators")
                else:
                    st.success(f"Not spam ({selected_pred['confidence']*100:.1f}% confidence)")
                    generate_word_cloud(selected_pred['full_text'], "Normal Email Content")
            
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("No prediction history yet. Make some predictions on the 'Classify Email' tab!")

if __name__ == '__main__':
    main()