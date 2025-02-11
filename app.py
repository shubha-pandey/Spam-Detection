# for streamlit application

import nltk

# run these only once and after the packages are installed no need to re-run them everytime
#nltk.download('punkt')
#nltk.download('punkt_tab')
#nltk.download('stopwords')

import streamlit as st                              # st is used to create UI
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from langdetect import detect


# Initialize PorterStemmer
ps = PorterStemmer()


# Function to preprocess text
def transform_text(text):                                # same function as in the data preprocessing
    text = text.lower()
    useful_words = []
    words = nltk.word_tokenize(text)

    for word in words :
        if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation:
            useful_words.append(ps.stem(word))    
    
    return " ".join(useful_words)


# Function to generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    return wordcloud


# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tk = pickle.load(open('vectorizer.pkl', 'rb'))


# image file
image = Image.open("spam_detector_logo.png")     # for adding images


# Streamlit page configuration
st.set_page_config(page_title="Spam Detection App", page_icon="📧", layout="centered")


# App header
st.title("📧 Spam Detection System")     # title

st.markdown("""
#### Your reliable tool for identifying spam messages! 🤖
""")


with st.sidebar:
    st.header("About")
    st.image(image, use_container_width=False, width=200)     # image
    st.markdown("""
    **Welcome to the Spam Detection App!**
    This app classifies sms or email messages as **Spam** or **Ham (Not Spam)**.
                      
    **How It Works?**
    - Input your message in the provided text area.
    - Click on Check button to classify it.
    - Voila now you know whether it is genuine or not!
                
    Want to see some fun visualizations of your message ? Click on this button🔻
    """)
    generate_wordcloud_button = st.button("Generate Word Cloud")     # button
    

input_sms = st.text_area("📩 Paste your message below to see if it's spam or not:", height=200)     # where to paste text 


# Generate word cloud if button is clicked
if generate_wordcloud_button:
    if input_sms.strip():
        wordcloud = generate_wordcloud(input_sms)
        st.markdown("### Word Cloud of Your Message:")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.warning("⚠️ Please enter a valid message to generate a Word Cloud 📝")


# Prediction section
if st.button("Check⚙️") :
    if input_sms :
        with st.spinner("Analyzing..."):
            
            # Preprocess and classify
            transformed_sms = transform_text(input_sms)
            vectorized_data = tk.transform([transformed_sms])     # convert data into vectors
            result = model.predict(vectorized_data)
        
            # Display results 
            if result[0] == 0 :
                st.success(f"✅ Your message is **NOT SPAM!**")
            else :
                st.error(f"❗Warning: Your message is **likely SPAM!**\nExercise caution! 🚨")
    
    else :
        st.warning("⚠️ Please enter some text to analyze! 📝")


# Footer
st.markdown("---")
st.markdown("© 2025 Spam Detection App | Created by Shubha Pandey 🕵🏻‍♀️")
