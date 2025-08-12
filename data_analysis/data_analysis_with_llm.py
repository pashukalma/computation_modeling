''' 
Analysis of data with LLM
''' 

!pip install openai > /dev/null
!pip install pandas scikit-learn > /dev/null

os.environ['OPENAI_API_KEY'] = openai_api_key

''' https://platform.openai.com/docs/models for a list of models '''
from openai import OpenAI
import os

client = OpenAI(api_key=openai_api_key)

def set_environment():
  variable_dict = globals().items()
  for key, value in variable_dict:
    if 'API' in key or 'ID' in key:
      os.environ[key] = value
set_environment()

import argparse
import pandas as pd

def create_prompt(text):
  instructions = 'Is the review sentiment positive or negative?'
  formatting = '"Positive" or "Negative"'
  return f'Text:{text}\n{instructions}\nAnswer ({formatting}):'

def invoke_llm(prompt):
  messages = [
        {'content':prompt, 'role':'user'} ]
  response = client.chat.completions.create(
        messages=messages, model='gpt-4o')
  return response.choices[0].message.content

def classify(text):
  prompt = create_prompt(text)
  return invoke_llm(prompt)

from google.colab import files
files.upload()

df = pd.read_csv('engagements.csv')

df.head()

df['class_media_caption'] = df['media_caption'].head(100).apply(classify)

df['class_text'] = df['comment_text'].head(100).apply(classify)

statistics = df['class_media_caption'].value_counts()
print(statistics)
df.to_csv('class_media_caption.csv')

statistics = df['class_text'].value_counts()
print(statistics)
df.to_csv('class_text.csv')

!head class_media_caption.csv

!head class_text.csv

"""Clustering"""

!pip install scikit-learn==1.3.2 numpy==1.26.0 > /dev/null

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# Combine relevant text columns for clustering
df_subset = df.head(100).copy()
df_subset['combined_text'] = df_subset['media_caption'].fillna('') + ' ' + df_subset['comment_text'].fillna('')

# Vectorize the text data
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df_subset['combined_text'])

# Perform KMeans clustering
num_clusters = 5  # You can adjust the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df_subset['cluster'] = kmeans.fit_predict(X)

# Display the first few rows with cluster labels
display(df_subset[['combined_text', 'cluster']].head())