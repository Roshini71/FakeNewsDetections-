import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pandas as pd
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")
print(fake.head())
print(real.head())

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real])
data = data.sample(frac=1)
data.reset_index(drop=True, inplace=True)
print(data.head())

X = data["text"]
y = data["label"]

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

news = ["Government announces new economic reform"]
news_vector = vectorizer.transform(news)
prediction = model.predict(news_vector)
print(prediction)

import matplotlib.pyplot as plt
import seaborn as sns

labels = ['Fake News', 'Real News']
sizes = [len(fake), len(real)]
plt.figure()
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title("Fake vs Real News Distribution")
plt.show()

news_counts = data['label'].value_counts()
plt.figure()
sns.barplot(x=news_counts.index, y=news_counts.values)
plt.xlabel("News Type")
plt.ylabel("Count")
plt.title("Number of Fake vs Real News Articles")
plt.show()

data['word_count'] = data['text'].apply(lambda x: len(str(x).split()))
plt.figure()
sns.histplot(data['word_count'], bins=50)
plt.title("Word Count Distribution in News Articles")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()
