import pandas as pd
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
import torch

# Download VADER lexicon
nltk.download('vader_lexicon')

# Load the CSV file
file_path = r'C:\Users\hp\Desktop\Code\steam_reviews_all_games.csv'
df = pd.read_csv(file_path)

# Ensure all reviews are strings and handle missing values
df['review'] = df['review'].astype(str).fillna('')

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Load pre-trained fine-tuned DistilBERT model for sentiment analysis
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

# Load sentiment-analysis pipeline with DistilBERT
sentiment_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Function to analyze sentiment using TextBlob
def analyze_sentiment_textblob(review):
    analysis = TextBlob(review)
    return analysis.sentiment.polarity

# Function to analyze sentiment using VADER
def analyze_sentiment_vader(review):
    scores = sid.polarity_scores(review)
    return scores['compound']

# Function to analyze sentiment using DistilBERT
def analyze_sentiment_bert(review):
    # Ensure text length does not exceed model's max length
    inputs = tokenizer(review, truncation=True, padding=True, max_length=512, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)
    return 'POSITIVE' if prediction.item() == 1 else 'NEGATIVE'

# Apply sentiment analysis with TextBlob
df['sentiment_textblob'] = df['review'].apply(analyze_sentiment_textblob)

# Apply sentiment analysis with VADER
df['sentiment_vader'] = df['review'].apply(analyze_sentiment_vader)

# Apply sentiment analysis with DistilBERT
df['sentiment_bert'] = df['review'].apply(analyze_sentiment_bert)

# Save the results to a new CSV file
output_path = r'C:\Users\hp\Desktop\Code\steam_reviews_all_games_with_sentiments.csv'
df.to_csv(output_path, index=False)

print(f"Sentiment analysis completed and saved to '{output_path}'")
