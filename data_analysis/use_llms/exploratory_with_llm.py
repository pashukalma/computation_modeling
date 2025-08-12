''' 

### Analysis of data with LLM with OpenAI
''' 

!pip install openai > /dev/null
!pip install pandas scikit-learn > /dev/null

from google.colab import files
files.upload()

import os
import argparse
import pandas as pd

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

"""### Clustering"""

!pip install scikit-learn numpy > /dev/null

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to the input file')
    args = parser.parse_args()

    df = pd.read_csv(args.path)
    df['class_media_caption'] = df['media_caption'].head(100).apply(classify)
    df.to_csv('class_media_caption.csv')
    print(df)
    statistics = df['class_text'].value_counts()
    print(statistics)
    df.to_csv('class_text.csv')

"""### Sentiment pipeline using langchain"""

import argparse
import pandas as pd

!pip install langchain_openai > /dev/null
!pip install langchain_core > /dev/null

from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables.passthrough import RunnablePassthrough

''' Creates chain for text classification'''
def create_chain():
    prompt = ChatPromptTemplate.from_template(
        '{text}\n'
        'Is the sentiment positive or negative?\n'
        'Answer ("Positive"/"Negative")\n')
    llm = ChatOpenAI(
        model='gpt-4o', temperature=0,
        max_tokens=1)
    parser = StrOutputParser()
    chain = ({'text':RunnablePassthrough()} | prompt | llm | parser)
    return chain

df = pd.read_csv('engagements.csv')
df,

chain = create_chain()

class_text_result = chain.batch(list(df['comment_text'[:10]]))
df['class_text'] = class_text_result
df.to_csv('class_text_result.csv')

''' process with batch_size '''
import time
chain = create_chain()
batch_size = 100  # Adjust batch size as needed
num_batches = (len(df) + batch_size - 1) // batch_size
all_results = []

for i in range(num_batches):
    start_index = i * batch_size
    end_index = min((i + 1) * batch_size, len(df))
    comment_batch = list(df['comment_text'][start_index:end_index])
    batch_results = chain.batch(comment_batch)
    all_results.extend(batch_results)
    time.sleep(1) # Add a delay between batches

df['class_text'] = all_results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help='Path to input .csv file')
    args = parser.parse_args()

    df = pd.read_csv(args.file_path)
    chain = create_chain()

    class_text_result = chain.batch(list(df['comment_text']))
    df['class_text'] = class_text_result
    df.to_csv('class_text_result.csv')

"""### Setiment pipeline with Vertex"""

!pip install google-cloud-aiplatform > /dev/null

from google.colab import auth
auth.authenticate_user()

import vertexai

PROJECT_ID = 'llm-test-428715'  # Replace with your Google Cloud project ID
LOCATION = 'us-central1'  # Replace with your desired Vertex AI location
vertexai.init(project=PROJECT_ID, location=LOCATION)
print(f"Vertex AI initialized for project '{PROJECT_ID}' in location '{LOCATION}'.")

from vertexai.language_models import TextGenerationModel
# Choose a model suitable for text classification/sentiment analysis
# 'text-bison@001' is a good general-purpose text model
model = TextGenerationModel.from_pretrained("text-bison@001")

import pandas as pd
df = pd.read_csv('engagements.csv')
display(df.describe(include='all'))

df['vertex_sentiment'].fillna('Unknown', inplace=True)

# 1. Define a prompt template string
sentiment_prompt_template = """
Analyze the sentiment of the following text and classify it as either "Positive" or "Negative".

Text: {text}
Sentiment:
"""

def get_vertex_sentiment(text):
    if pd.isna(text):
        return None  # Handle missing values
    try:
        prompt = sentiment_prompt_template.format(text=text)
        response = model.predict(prompt)
        sentiment = response.text.strip()
        if sentiment not in ["Positive", "Negative"]:
             return "Unknown" # Or re-try, or log an error
        return sentiment
    except Exception as e:
        print(f"Error processing text: {text[:50]}... Error: {e}")
        return "Error" # Handle potential errors during prediction

df['vertex_sentiment'] = df['comment_text'].head(100).apply(get_vertex_sentiment)
df['vertex_sentiment'].fillna('Unknown', inplace=True)
display(df[['comment_text', 'vertex_sentiment']].head())