import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(
    page_title="Movie Sentiment AI",
    page_icon="🎬",
    layout="centered"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .stTextArea textarea {
        font-size: 16px !important;
    }
    .stButton button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State for the text area
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Function to handle button clicks for examples
def set_text(text):
    st.session_state.user_input = text

# Download NLTK data
nltk.download('stopwords')

# --- 2. LOAD RESOURCES (ROBUST) ---
@st.cache_resource
def load_resources():
    # Use absolute paths to prevent "File Not Found" errors
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'sentiment_model.keras')
    token_path = os.path.join(current_dir, 'tokenizer.pickle')
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file missing: {model_path}")
        st.stop()
        
    model = tf.keras.models.load_model(model_path)
    with open(token_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_resources()

# --- 3. CLEANING FUNCTION ---
def clean_text(text):
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    if 'not' in all_stopwords:
        all_stopwords.remove('not')
    stop_words_set = set(all_stopwords)
    
    text = re.sub(r'<.*?>', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words_set]
    return ' '.join(text)

# --- 4. THE SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2503/2503508.png", width=100)
    st.title("Movie AI 🎬")
    st.info("This model uses a **LSTM** neural network trained on 25,000 IMDB reviews.")
    st.write("---")
    st.caption("Created by **Yoosuf Ahamed**")

# --- 5. MAIN UI ---
st.title("🎬 Movie Review Classifier")
st.markdown("### How did you like the movie?")
st.write("Type your own review or click one of the examples below:")

# Quick Test Buttons (Columns)
col1, col2, col3 = st.columns(3)
with col1:
    st.button("Try a Masterpiece", on_click=set_text, args=("I absolutely loved this movie! The acting was incredible and the ending blew my mind.",))
with col2:
    st.button("Try a Disaster", on_click=set_text, args=("This was the worst film I have ever seen. Total waste of time and money.",))
with col3:
    st.button("Try Tricky", on_click=set_text, args=("The acting was great, but the story was really boring and slow.",))

# Text Input (Linked to Session State)
user_text = st.text_area(
    "Enter your review here:", 
    value=st.session_state.user_input, 
    height=150, 
    placeholder="Type something like: 'The visual effects were stunning...'"
)

# Analyze Button
if st.button("🚀 Analyze Sentiment", type="primary"):
    if user_text:
        with st.spinner('Reading review...'):
            # Process
            cleaned = clean_text(user_text)
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=200, padding='pre', truncating='post')
            prediction = model.predict(padded)[0][0]
            
            # Logic
            is_positive = prediction > 0.5
            confidence = prediction if is_positive else 1 - prediction
            label = "POSITIVE" if is_positive else "NEGATIVE"
            emoji = "🍿" if is_positive else "🍅"
            color = "green" if is_positive else "red"

            # --- RESULT SECTION ---
            st.write("---")
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                # Big Metric Display
                st.metric(label="Sentiment", value=f"{label} {emoji}", delta=f"{confidence:.1%}")
            
            with res_col2:
                # Visual Gauge
                st.write(f"**Confidence Score:** {confidence:.1%}")
                if is_positive:
                    st.progress(int(confidence * 100))
                    st.balloons()
                else:
                    # Invert progress for negative so it looks 'full'
                    st.progress(int(confidence * 100))
            
            # Debug Dropdown
            with st.expander("🔍 See Model Internals"):
                st.code(f"Cleaned Input: {cleaned}")
                st.write(f"Raw Sigmoid Output: `{prediction:.4f}`")
                
    else:
        st.warning("⚠️ Please enter some text first!")