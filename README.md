# SMS Spam Collection Classification

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/) 
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24-orange)](https://scikit-learn.org/)

This project detects **spam SMS messages** using **NLP** and **Machine Learning**. Messages are labeled as `spam` or `ham`.

## Dataset

- **Source:** [SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)  
- **Format:** CSV (tab-separated)  
- **Columns:**
  - `label`: `spam` or `ham`
  - `text`: SMS message content

## Project Workflow

1. **Load Dataset** with `pandas`.
2. **Split Data** into training/testing sets (`train_test_split`).
3. **Vectorization** using TF-IDF (`TfidfVectorizer`) with English stop words removed.
4. **Train Model** with **Multinomial Naive Bayes**.
5. **Evaluate** using `classification_report`.
6. **Visualize** frequent words in `spam` and `ham` using **WordCloud**.

## Usage

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('SMSSpamCollection.csv', encoding='latin-1', sep='\t', header=None)
df.columns = ['label', 'text']

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# WordCloud for spam vs ham
spam_words = ' '.join(df[df['label']=='spam']['text'])
ham_words = ' '.join(df[df['label']=='ham']['text'])

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(WordCloud(width=600, height=400, background_color='white').generate(spam_words))
plt.title('Spam WordCloud')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(WordCloud(width=600, height=400, background_color='white').generate(ham_words))
plt.title('Ham WordCloud')
plt.axis('off')

plt.show()
