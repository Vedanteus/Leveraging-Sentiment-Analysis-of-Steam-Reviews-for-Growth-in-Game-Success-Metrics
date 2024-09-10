import streamlit as st
from textblob import TextBlob
import cleantext
import pandas as pd

st.header('Sentiment Analysis Tool')

# Section for Text Analysis
with st.expander('Analyze Individual Text'):
    user_input = st.text_input('Enter your text:')
    if user_input:
        sentiment = TextBlob(user_input)
        st.write('Polarity Score:', round(sentiment.sentiment.polarity, 2))
        st.write('Subjectivity Score:', round(sentiment.sentiment.subjectivity, 2))

    clean_input = st.text_input('Text to Clean:')
    if clean_input:
        cleaned_text = cleantext.clean(clean_input, clean_all=False, 
                                       extra_spaces=True, stopwords=True, 
                                       lowercase=True, numbers=True, punct=True)
        st.write('Cleaned Text:', cleaned_text)

# Section for CSV File Analysis
with st.expander('Analyze CSV File'):
    uploaded_file = st.file_uploader('Upload CSV File')

    # Function to calculate polarity score
    def calculate_polarity(text):
        sentiment_analysis = TextBlob(text)
        return sentiment_analysis.sentiment.polarity

    # Function to determine sentiment category
    def determine_sentiment(polarity_score):
        if polarity_score >= 0.5:
            return 'Positive'
        elif polarity_score <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])
        data['Polarity Score'] = data['review'].apply(calculate_polarity)
        data['Sentiment'] = data['Polarity Score'].apply(determine_sentiment)
        st.write(data.head(10))

        @st.cache_data
        def convert_to_csv(dataframe):
            return dataframe.to_csv().encode('utf-8')

        csv_data = convert_to_csv(data)

        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name='sentiment_analysis.csv',
            mime='text/csv',
        )





