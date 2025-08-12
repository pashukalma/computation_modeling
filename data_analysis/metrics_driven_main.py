''' 

Metrics-driven report with **Code Refactoring**

Create a metrics-driven report of trends and topics of interest, and how they shift over time
''' 

from pandas import read_csv
import pandas as pd

''' work with engagement data '''
from google.colab import files
files.upload()
import os
print(os.listdir())

''' perform basic data exploration '''
df = pd.read_csv('engagements.csv')
df.head()[:5]
df.info()
display(df.describe())

"""Define functions for data loading and initial exploration as requested by the subtask.


"""

import pandas as pd
import os

def load_data(file_path):
    """Loads data from a CSV file into a pandas DataFrame with error handling.

    Args:
        file_path: The path to the CSV file.

    Returns:
        A pandas DataFrame if the file is loaded successfully, None otherwise.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def explore_data(df):
    """Performs basic data exploration on a pandas DataFrame.

    Args:
        df: The pandas DataFrame to explore.
    """
    if df is not None:
        print("DataFrame Head:")
        display(df.head())
        print("\nDataFrame Info:")
        df.info()
        print("\nDataFrame Description:")
        display(df.describe())
    else:
        print("Cannot explore data: DataFrame is None.")

# Update the notebook to use these new functions
file_path = 'engagements.csv'
df = load_data(file_path)
explore_data(df)

"""### data cleaning and preparation

Define the `clean_data` function to handle missing values and convert data types, and then call this function to clean the dataframe.
"""

def clean_data(df):
    """Cleans and prepares the DataFrame.

    Handles missing values and converts data types.

    Args:
        df: The pandas DataFrame to clean.

    Returns:
        The cleaned pandas DataFrame.
    """
    if df is None:
        print("Cannot clean data: DataFrame is None.")
        return None

    # Convert 'timestamp' to datetime objects, handling potential errors
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    except Exception as e:
        print(f"Error converting timestamp to datetime: {e}")
        # Handle the error, perhaps by returning None or logging
        return None

    # Extract date component
    df['date'] = df['timestamp'].dt.date

    # Fill missing values
    df['media_caption'].fillna('', inplace=True)
    df['comment_text'].fillna('', inplace=True)

    print("Data cleaning and preparation complete.")
    return df

# Update the notebook to call this new function after loading and exploration
df = clean_data(df)

# Display the cleaned data to verify
if df is not None:
    display(df.head())
    display(df.info())

"""### Engagement trends analysis

Define the functions to calculate daily, weekly, and monthly engagement trends and then call them to display the results.
"""

def calculate_daily_engagement(df):
    """Calculates daily engagement count.

    Args:
        df: The pandas DataFrame with a 'date' column.

    Returns:
        A DataFrame with daily engagement counts, sorted by date.
    """
    if df is None or 'date' not in df.columns:
        print("Cannot calculate daily engagement: Invalid DataFrame.")
        return None

    daily_engagement = df.groupby('date').size().reset_index(name='engagement_count')
    daily_engagement['date'] = pd.to_datetime(daily_engagement['date'])
    daily_engagement = daily_engagement.sort_values(by='date')
    return daily_engagement

def calculate_weekly_engagement(daily_engagement_df):
    """Calculates weekly engagement count from daily engagement.

    Args:
        daily_engagement_df: DataFrame with daily engagement counts and a 'date' column.

    Returns:
        A DataFrame with weekly engagement counts.
    """
    if daily_engagement_df is None or 'date' not in daily_engagement_df.columns or 'engagement_count' not in daily_engagement_df.columns:
        print("Cannot calculate weekly engagement: Invalid daily engagement DataFrame.")
        return None

    weekly_engagement = daily_engagement_df.resample('W', on='date')['engagement_count'].sum().reset_index(name='weekly_engagement_count')
    return weekly_engagement

def calculate_monthly_engagement(daily_engagement_df):
    """Calculates monthly engagement count from daily engagement.

    Args:
        daily_engagement_df: DataFrame with daily engagement counts and a 'date' column.

    Returns:
        A DataFrame with monthly engagement counts.
    """
    if daily_engagement_df is None or 'date' not in daily_engagement_df.columns or 'engagement_count' not in daily_engagement_df.columns:
        print("Cannot calculate monthly engagement: Invalid daily engagement DataFrame.")
        return None

    monthly_engagement = daily_engagement_df.resample('M', on='date')['engagement_count'].sum().reset_index(name='monthly_engagement_count')
    return monthly_engagement

# Calculate and display engagement trends using the new functions
daily_engagement = calculate_daily_engagement(df)
weekly_engagement = calculate_weekly_engagement(daily_engagement)
monthly_engagement = calculate_monthly_engagement(daily_engagement)

print("Daily Engagement:")
display(daily_engagement.head())
print("\nWeekly Engagement:")
display(weekly_engagement.head())
print("\nMonthly Engagement:")
display(monthly_engagement.head())

"""## Topic identification

- Functions for text cleaning, word frequency analysis, and keyword-based topic identification.
"""

import string
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

def clean_text(text):
    """Cleans text by converting to lowercase, removing punctuation, and stripping whitespace."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

def remove_stopwords(text):
    """Removes stop words from text."""
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

def analyze_word_frequency(df, text_column='cleaned_text', top_n=50):
    """Analyzes word frequency in a text column.

    Args:
        df: The pandas DataFrame.
        text_column: The name of the column containing cleaned text.
        top_n: The number of top words to display.

    Returns:
        A sorted list of (word, count) tuples.
    """
    if df is None or text_column not in df.columns:
        print(f"Cannot analyze word frequency: Invalid DataFrame or missing '{text_column}' column.")
        return None

    word_counts = {}
    for text in df[text_column]:
        for word in text.split():
            word_counts[word] = word_counts.get(word, 0) + 1

    sorted_word_counts = sorted(
        word_counts.items(), key=lambda item: item[1], reverse=True)

    print(f"Top {top_n} most frequent words:")
    for word, count in sorted_word_counts[:top_n]:
        print(f"{word}: {count}")

    return sorted_word_counts

def identify_topics_keywords(df, topic_keywords, text_column='combined_text'):
    """Identifies topics based on keywords in a text column.

    Args:
        df: The pandas DataFrame.
        topic_keywords: A dictionary mapping topic names to lists of keywords.
        text_column: The name of the column containing the text to analyze.

    Returns:
        The DataFrame with new boolean columns for each topic.
    """
    if df is None or text_column not in df.columns:
        print(f"Cannot identify topics: Invalid DataFrame or missing '{text_column}' column.")
        return None

    df_topics = df.copy() # Create a copy to avoid modifying the original DataFrame inplace
    for topic, keywords in topic_keywords.items():
        df_topics[topic] = df_topics[text_column].apply(
            lambda x: any(keyword in str(x).lower() for keyword in keywords))

    print("Topic identification complete.")
    return df_topics

# Update the notebook to use these new functions
if df is not None:
    # 1. Combine the media_caption and comment_text columns
    df['combined_text'] = df['media_caption'] + ' ' + df['comment_text']

    # 2. Perform text cleaning and remove stop words
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    df['cleaned_text'] = df['cleaned_text'].apply(remove_stopwords)

    # 3. Analyze word frequency
    sorted_word_counts = analyze_word_frequency(df)

    # 4. Identify topics based on keywords
    topic_keywords = {
        'brand_product': ['tree hut', 'scrub', 'butter', 'wash',
                          'scent', 'skin', 'product', 'smell', 'body'],
        'marketing_promotions': ['giveaway', 'contest', 'win', 'enter',
                                 'winners', 'tag', 'share', 'follow',
                                 'chance', 'promo', 'discount'],
        'user_sentiment': ['love', 'good', 'great', 'amazing', 'help',
                           'favorite', 'want', 'need', 'obsessed', 'repurchase']
    }
    df = identify_topics_keywords(df, topic_keywords)

    # Display the first few rows with the new columns
    display(
      df[['media_caption', 'comment_text', 'combined_text', 'cleaned_text'] + list(topic_keywords.keys())].head())

"""### Correlation analysis

Define the `analyze_correlation` function to merge the daily engagement and daily topic popularity dataframes and calculate the correlation matrix.
"""

def analyze_correlation(daily_engagement_df, daily_topic_popularity_df):
    """Analyzes the correlation between engagement and topic popularity.

    Args:
        daily_engagement_df: DataFrame with daily engagement counts and a 'date' column.
        daily_topic_popularity_df: DataFrame with daily topic popularity and a 'date' column.

    Returns:
        A DataFrame representing the correlation matrix, or None if inputs are invalid.
    """
    if daily_engagement_df is None or daily_topic_popularity_df is None or 'date' not in daily_engagement_df.columns or 'date' not in daily_topic_popularity_df.columns:
        print("Cannot analyze correlation: Invalid input DataFrames.")
        return None

    daily_trends = pd.merge(daily_engagement_df, daily_topic_popularity_df, on='date')
    correlation_matrix = daily_trends.corr(numeric_only=True)

    print("Correlation matrix between engagement and topic popularity:")
    display(correlation_matrix[['engagement_count']])

    return correlation_matrix

# Update the notebook to use the new function
if daily_engagement is not None and daily_topic_popularity is not None:
    correlation_matrix = analyze_correlation(daily_engagement, daily_topic_popularity)

"""### Visualization

Define functions to handle the generation of engagement and topic popularity plots, as well as the correlation heatmap.
"""

import matplotlib.pyplot as plt
import seaborn as sns

def plot_engagement_trends(daily_df, weekly_df, monthly_df):
    """Plots daily, weekly, and monthly engagement trends."""
    if daily_df is not None:
        plt.figure(figsize=(15, 6))
        plt.plot(daily_df['date'], daily_df['engagement_count'])
        plt.title('Daily Engagement Trends')
        plt.xlabel('Date')
        plt.ylabel('Engagement Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    if weekly_df is not None:
        plt.figure(figsize=(15, 6))
        plt.plot(weekly_df['date'], weekly_df['weekly_engagement_count'])
        plt.title('Weekly Engagement Trends')
        plt.xlabel('Date')
        plt.ylabel('Weekly Engagement Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    if monthly_df is not None:
        plt.figure(figsize=(15, 6))
        plt.plot(monthly_df['date'], monthly_df['monthly_engagement_count'])
        plt.title('Monthly Engagement Trends')
        plt.xlabel('Date')
        plt.ylabel('Monthly Engagement Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def plot_topic_popularity_trends(daily_df, weekly_df, monthly_df, topic_keywords):
    """Plots daily, weekly, and monthly topic popularity trends."""
    if daily_df is not None:
        plt.figure(figsize=(15, 6))
        for topic in topic_keywords.keys():
            plt.plot(daily_df['date'], daily_df[topic], label=topic)
        plt.title('Daily Topic Popularity Trends')
        plt.xlabel('Date')
        plt.ylabel('Popularity (Mean Engagement)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    if weekly_df is not None:
        plt.figure(figsize=(15, 6))
        for topic in topic_keywords.keys():
            plt.plot(weekly_df['date'], weekly_df[topic], label=topic)
        plt.title('Weekly Topic Popularity Trends')
        plt.xlabel('Date')
        plt.ylabel('Popularity (Sum of Daily Means)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    if monthly_df is not None:
        plt.figure(figsize=(15, 6))
        for topic in topic_keywords.keys():
            plt.plot(monthly_df['date'], monthly_df[topic], label=topic)
        plt.title('Monthly Topic Popularity Trends')
        plt.xlabel('Date')
        plt.ylabel('Popularity (Sum of Daily Means)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_correlation_heatmap(correlation_matrix):
    """Generates a heatmap of the correlation matrix."""
    if correlation_matrix is not None:
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix: Engagement vs. Topic Popularity')
        plt.tight_layout()
        plt.show()

# Update the notebook to use the new functions
if daily_engagement is not None and weekly_engagement is not None and monthly_engagement is not None:
    plot_engagement_trends(daily_engagement, weekly_engagement, monthly_engagement)

if daily_topic_popularity is not None and weekly_topic_popularity is not None and monthly_topic_popularity is not None and topic_keywords is not None:
     plot_topic_popularity_trends(daily_topic_popularity, weekly_topic_popularity, monthly_topic_popularity, topic_keywords)

if correlation_matrix is not None:
    plot_correlation_heatmap(correlation_matrix)

"""# Metrics-Driven Report

### Project Description

This project analyzes engagement data to identify trends and topics of interest over time. It provides insights into how user engagement changes daily, weekly, and monthly, and explores the correlation between engagement and different topics discussed in media captions and comments.

### Setup and Usage

1.  **Clone the Repository:** If you are using Git, clone the repository to your local machine or open the notebook in Google Colab.
2.  **Install Dependencies:** The project requires the following libraries:
    *   pandas
    *   matplotlib
    *   seaborn
    *   nltk
    *   transformers
    *   torch

    You can install them using pip:
"""

import nltk
    nltk.download('stopwords')