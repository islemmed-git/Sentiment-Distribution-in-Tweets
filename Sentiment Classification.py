import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

# Sample dataset (replace this with your actual dataset)
data = {
    'Tweet': [
        "I love this product! It's amazing.",
        "this is bad.",
        "Neutral tweet here."
    ]
}

df = pd.DataFrame(data)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Apply sentiment analysis to each tweet
df['Sentiment_Score'] = df['Tweet'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Classify sentiments
df['Sentiment'] = df['Sentiment_Score'].apply(
    lambda score: 'Positive' if score >= 0.05 else ('Negative' if score <= -0.05 else 'Neutral')
)

# Display the DataFrame
print(df)

# Visualization: Pie chart of sentiment distribution
sentiment_counts = df['Sentiment'].value_counts()
sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral', 'lightblue'])
plt.title('Sentiment Distribution in Tweets')
plt.show()