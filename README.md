# Sentiment-Distribution-in-Tweets 
**Sentiment Analysis with NLTK**

# Overview :

This code performs sentiment analysis on a given dataset of tweets using the Natural Language Toolkit (NLTK) library. The sentiment of each tweet is determined using the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool. The results are then visualized using a pie chart to display the distribution of sentiments in the dataset.

# Prerequisites :
Before running the code, make sure to install the necessary library:

import nltk
nltk.download('vader_lexicon')

# Usage :

1-Replace the sample dataset (data) with your actual dataset.
2-Ensure that the required libraries are installed.
3-Run the code to perform sentiment analysis and visualize the results.

# Code Explanation :

## Import Libraries
nltk: Natural Language Toolkit library.
pandas: Data manipulation library.
matplotlib.pyplot: Plotting library for data visualization.

## import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

## Sample Dataset:
Replace the 'Tweet' values in the data dictionary with your actual dataset.

 data = {
     'Tweet': [
         "I love this product! It's amazing.",
         "i am sad.",
         "Neutral tweet here."
     ]
 }

 df = pd.DataFrame(data)

## Sentiment Analysis:

Initialize the sentiment analyzer (sia).
Apply sentiment analysis to each tweet and create a new column ('Sentiment_Score') in the DataFrame.

sia = SentimentIntensityAnalyzer()
df['Sentiment_Score'] = df['Tweet'].apply(lambda x: sia.polarity_scores(x)['compound'])

## Sentiment Classification:
Classify sentiments based on the sentiment score:
'Positive' if score >= 0.05
'Negative' if score <= -0.05
'Neutral' otherwise

df['Sentiment'] = df['Sentiment_Score'].apply(
    lambda score: 'Positive' if score >= 0.05 else ('Negative' if score <= -0.05 else 'Neutral')
)


## Display the DataFrame:
Print the DataFrame with the added sentiment information.

print(df)

## Visualization: Pie Chart:
Create a pie chart to visualize the distribution of sentiments in the dataset. 

sentiment_counts = df['Sentiment'].value_counts()
sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral', 'lightblue'])
plt.title('Sentiment Distribution in Tweets')
plt.show()

![Capture d'écran 2024-12-19 023908](https://github.com/user-attachments/assets/5eb8269a-b52c-48b1-b033-a155b0bc4029)
