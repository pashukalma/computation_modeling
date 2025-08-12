''' 

### **Text processing**
Apply LLM directly for analysis, transform a text into embedding vectors and create embedding 


- classification
- extracting key information from application
- clustering documents by their content
''' 

!pip install openai
!pip install pandas scikit-learn==1.3

''' https://platform.openai.com/docs/models for a list of models '''
from openai import OpenAI
import os

openai_api_key = ""
client = OpenAI(api_key=openai_api_key)
os.environ['OPENAI_API_KEY'] = openai_api_key

def set_environment():
  variable_dict = globals().items()
  for key, value in variable_dict:
    if 'API' in key or 'ID' in key:
      os.environ[key] = value
set_environment()

models = client.models.list()
for model in models.data:
  print(model.id) if model.id.startswith('gpt-4o-search') else None

'''
customize model behavior - configure output configuration - configure randomization
'''
result = client.chat.completions.create(
    model = 'gpt-4o',
    messages = [{
            'role': 'user',
            'content': 'Data Analysis with LLMs'}],
    max_tokens = 100, #512,
    stop = 'stopping word',
    temperature = 1.5,
    presence_penalty= 0.5,
    logit_bias= {'50256': -100}
)
result.choices[0].message.content

"""Example a classification problem for book reviews

"""

''' Example a classification problem for book reviews
Review --> Generate Prompt --> Language model --> Classification
'''
def create_prompt(text):

  task = 'Is the sentiment positive, negative or neutral'
  answer_format = 'Review ("Positive"/"Negative")'
  return f'{text} \n {task} \n {answer_format}'

def invoke_llm(prompt):
  ''' Query LLM with input prompt and return answer by language model '''
  for i in range(1, 3):
    try:
      response = client.chat.completions.create(
          model = 'gpt-4o',
          messages = [{
              'role': 'user', 'content': prompt}]
      )
      return response.choices[0].message.content
    except:
      continue
  raise Exception('Unable to query OpenAI at this time')

def classify_review(text):
  prompt = create_prompt(text)
  label = invoke_llm(prompt)
  return label

''' output the result '''
import argparse
import pandas as pd
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('filepath__', type=str)
  args = parser.parse_args()
  print(classify_review(args.text))

  df = pd.read_csv(args.text)
  df['Class'] = df['Review'].apply(classify_review)
  statistics = df['Class'].value_counts()
  print(statistics)
  df.to_csv('results.csv')


df = pd.read_csv('classification_data.txt')
df.head()

df['Class'] = df['Review'].apply(classify_review)
statistics = df['Class'].value_counts()
print(statistics)
df.to_csv('results.csv')

"""**Text Extraction**"""
import re, time, argparse

def create_prompt(text, attributes):
  parts = []
  parts += ['Extract these attributes into a table']
  #parts += [f'Attributes: {attributes}']
  parts = [','.join(attributes)]
  parts += [f'Text source: {text}']
  parts += [('Mark the beginning of the table with <BeginTable> and the end with <EndTable>.')]
  parts += [('Separate rows by newline symbols and separate fields by pipe symbols (|).')]
  parts += [('Omit the table header and insert values in the attribute order from above.')]
  parts += [('Use the placeholder <NA> if the value for an attribute that is not available')]
  return '\n'.join(parts)

def invoke_llm(prompt):
  for i in range(1, 3):
    try:
      response = client.chat.completions.create(
          model = 'gpt-4o',
          messages = [{
              'role': 'user', 'content': prompt}]
      )
      return response.choices[0].message.content
    except:
      continue
  raise Exception('Unable to query OpenAI at this time')

def post_process(raw_answer):
  results = []
  table = re.findall('<BeginTable>(.*)<EndTable>', raw_answer, re.DOTALL)[0]
  for raw_data in table.split('\n'):
    if raw_data:
      row = raw_data.split('|')
      row = [field.strip() for field in row]
      row = [field for field in row if field]
      results.append(row)
  return results

def extract_attributes(text, attributes):
  prompt = create_prompt(text, attributes)
  print(prompt)
  raw_answer = invoke_llm(prompt)
  return post_process(raw_answer)

# Commented out IPython magic to ensure Python compatibility.
# %%writefile student_data.txt
# Name,GPA,Degree

df = pd.read_csv('student_data.txt')
attributes = df.columns.tolist()
extractions = []
for text in df.values:
  extractions += extract_attributes(text, attributes)
result_df = pd.DataFrame(extractions, columns=attributes)
result_df.to_csv('results.csv')

"""**Clustering text documents using language models**"""

'''
Email 1           Email 2
  |                 |
  |                 |
Embedding vector 1  Embedding vector N

          Clustering emails
'''

def get_embedding(text):
  for i in range(1, 3):
    try:
      response = client.embeddings.create(
          model = 'text-embedding-ada-002',
          input = text
      )
      return response.data[0].embedding
    except:
      continue
  raise Exception('Unable to query OpenAI at this time')

import sklearn
from sklearn.cluster import KMeans

def get_kmeans(embeddings, k):
  kmeans = KMeans(n_clusters = k, init = 'k-means++')
  kmeans.fit(embeddings)
  return kmeans.labels_

text1 = """
The Project Gutenberg eBook of Anna Karenina, by Leo Tolstoy
This eBook is for the use of anyone anywhere in the United States and most other
parts of the world at no cost and with almost no restrictions whatsoever.
You may copy it, give it away or re-use it under the terms of the Project
Gutenberg License included with this eBook or online at www.gutenberg.org. """
text2 = """
Title: Anna Karenina  Author: Leo Tolstoy  Release Date: July 1, 1998
[eBook #1399] [Most recently updated: September 20, 2022]  Language: English
Character set encoding: UTF-8  Produced by: David Brannan, Andrew Sly and
David Widger  *** START OF THE PROJECT GUTENBERG EBOOK ANNA KARENINA ***
[Illustration] ANNA KARENINA by Leo Tolstoy Translated by Constance Garnett
"""
get_embedding(text)[:3]

embeddings = [ get_embedding(text) for text in [text1, text2] ]
embeddings_ = pd.DataFrame['text'].apply(embeddings)
embeddings_

"""### **Working with images**

**Extraction information from multimodal**
- building agent for data analysis

**Using Graphs for queries**
"""