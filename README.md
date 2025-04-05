# CodeSoft
Absolutely! Let's walk through a detailed guide to building a machine learning model that predicts the genre of a movie based on its plot summary, using TF-IDF and classifiers like Naive Bayes, Logistic Regression, or Support Vector Machines (SVM).

# 🧠 Goal:
Build a model that takes text (plot summary) as input and outputs predicted genre(s). Since a movie can belong to multiple genres, this is a multi-label classification problem.
# ✅ Step-by-Step Guide
# 1. 📂 Load and Explore the Dataset
We assume a dataset with at least two columns:

plot: Text summary of the movie.

genres: List of genres (e.g. ['Action', 'Adventure'])

import pandas as pd

# Load dataset
df = pd.read_csv('movies.csv')  # Make sure your CSV is formatted correctly

# Check sample
print(df[['plot', 'genres']].head())
# 2. 🧹 Text Cleaning & Preprocessing
Clean the plot text: remove punctuation, lowercase everything, remove stopwords.
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-letters
    text = text.lower()                      # Lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['clean_plot'] = df['plot'].apply(clean_text)
# 3. 🔠 Feature Extraction (TF-IDF)
Convert text into numeric features using TF-IDF.
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_plot'])
# 4. 🎯 Encode Target Labels (Genres)
Use MultiLabelBinarizer because each movie can belong to multiple genres.
from sklearn.preprocessing import MultiLabelBinarizer
import ast

# Convert genres to list if they are string representations
df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genres'])

print("Classes (Genres):", mlb.classes_)
# 5. 🔀 Train/Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 6. ⚙️ Choose and Train Classifier
You can use any of the following:

# ✅ Option 1: Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

lr_model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
lr_model.fit(X_train, y_train)
# ✅ Option 2: Naive Bayes
from sklearn.naive_bayes import MultinomialNB

nb_model = OneVsRestClassifier(MultinomialNB())
nb_model.fit(X_train, y_train)
# ✅ Option 3: Support Vector Machine (SVM)
from sklearn.svm import LinearSVC

svm_model = OneVsRestClassifier(LinearSVC())
svm_model.fit(X_train, y_train)
# 7. 📊 Evaluation
from sklearn.metrics import classification_report, accuracy_score

y_pred = lr_model.predict(X_test)  # or nb_model / svm_model

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))
# 8. 🔍 Predict New Movie Genre
def predict_genre(plot_summary, model):
    cleaned = clean_text(plot_summary)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)
    return mlb.inverse_transform(prediction)

# Try an example
example_plot = "A group of heroes must save the world from an alien invasion."
predicted_genres = predict_genre(example_plot, lr_model)
print("Predicted Genres:", predicted_genres)
# 🔄 Switch Between Classifiers
Change this line depending on the model you want:
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model = OneVsRestClassifier(MultinomialNB())
model = OneVsRestClassifier(LinearSVC())
# 🧪 Optional: Use Word Embeddings (Instead of TF-IDF)
You can replace TF-IDF with Word2Vec, GloVe, or BERT embeddings using libraries like gensim or transformers. TF-IDF is faster and more interpretable, though.

Let me know if you want a version using Word2Vec or BERT.




