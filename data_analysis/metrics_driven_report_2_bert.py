''' 
Apply BERT classification on the combined text of media_caption and comment_text to categorize the content.
'''

# Commented out IPython magic to ensure Python compatibility.
# %pip install transformers

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Ensure 'combined_text' is all strings and fill any potential NaNs
df['combined_text'] = df['combined_text'].astype(str).fillna('')


model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Preprocess the 'combined_text' column
encoded_data = tokenizer.batch_encode_plus(
    df['combined_text'].tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    padding='max_length',
    max_length=128,  # You might need to adjust this based on your text length
    truncation=True,
    return_tensors='pt'
)

input_ids = encoded_data['input_ids']
attention_masks = encoded_data['attention_mask']

print("Preprocessing complete. Sample input_ids and attention_masks:")
print(input_ids[:2])
print(attention_masks[:2])

"""Summarize the findings in a comprehensive report, including visualizations of trends and key insights."""

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Plotting daily engagement trends
plt.figure(figsize=(15, 6))
plt.plot(daily_engagement['date'], daily_engagement['engagement_count'])
plt.title('Daily Engagement Trends')
plt.xlabel('Date')
plt.ylabel('Engagement Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 1. Plotting weekly engagement trends
plt.figure(figsize=(15, 6))
plt.plot(weekly_engagement['date'], weekly_engagement['weekly_engagement_count'])
plt.title('Weekly Engagement Trends')
plt.xlabel('Date')
plt.ylabel('Weekly Engagement Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 1. Plotting monthly engagement trends
plt.figure(figsize=(15, 6))
plt.plot(monthly_engagement['date'], monthly_engagement['monthly_engagement_count'])
plt.title('Monthly Engagement Trends')
plt.xlabel('Date')
plt.ylabel('Monthly Engagement Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Plotting daily topic popularity trends
plt.figure(figsize=(15, 6))
for topic in topic_keywords.keys():
    plt.plot(daily_topic_popularity['date'], daily_topic_popularity[topic], label=topic)
plt.title('Daily Topic Popularity Trends')
plt.xlabel('Date')
plt.ylabel('Popularity (Mean Engagement)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 2. Plotting weekly topic popularity trends
plt.figure(figsize=(15, 6))
for topic in topic_keywords.keys():
    plt.plot(weekly_topic_popularity['date'], weekly_topic_popularity[topic], label=topic)
plt.title('Weekly Topic Popularity Trends')
plt.xlabel('Date')
plt.ylabel('Popularity (Sum of Daily Means)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 2. Plotting monthly topic popularity trends
plt.figure(figsize=(15, 6))
for topic in topic_keywords.keys():
    plt.plot(monthly_topic_popularity['date'], monthly_topic_popularity[topic], label=topic)
plt.title('Monthly Topic Popularity Trends')
plt.xlabel('Date')
plt.ylabel('Popularity (Sum of Daily Means)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 3. Generating heatmap of correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix: Engagement vs. Topic Popularity')
plt.tight_layout()
plt.show()

'''
The dataset was successfully preprocessed for BERT analysis,
including tokenization and the creation of attention masks. However, the BERT
classification task could not be completed due to the lack of labeled data for training.
'''

"""Identify topics of interest (k-means clustering)"""

df['date'] = pd.to_datetime(df['timestamp'], format='mixed').dt.date
daily_topic_cluster_popularity = df.groupby('date')['topic_cluster'].mean().reset_index(name='mean_topic_cluster')
daily_topic_cluster_popularity['date'] = pd.to_datetime(daily_topic_cluster_popularity['date'])
daily_topic_cluster_popularity = daily_topic_cluster_popularity.sort_values(by='date')
display(daily_topic_cluster_popularity.head())

weekly_topic_cluster_popularity = daily_topic_cluster_popularity.resample(
    'W', on='date')['mean_topic_cluster'].mean().reset_index(name='weekly_mean_topic_cluster')
monthly_topic_cluster_popularity = daily_topic_cluster_popularity.resample(
    'M', on='date')['mean_topic_cluster'].mean().reset_index(name='monthly_mean_topic_cluster')

print("Weekly Topic Cluster Popularity:")
display(weekly_topic_cluster_popularity.head())
print("\nMonthly Topic Cluster Popularity:")
display(monthly_topic_cluster_popularity.head())



