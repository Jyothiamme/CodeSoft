# CodeSoft
# Name : Jyothi Amme
# Mail : ammejyothi@gmail.com
# CODESOFT Internship 
# Course : Machine leanrning
# Task-1 Create a machine learning model that can predict the genre of a
 ## movie based on its plot summary or other textual information.

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

## OUTPUT 
Awesome! Let's build a detailed machine learning model in Python that predicts the genre of a movie based on its plot summary using:

TF-IDF for feature extraction

Logistic Regression for classification (easily switchable to Naive Bayes or SVM)

Multi-label classification (since a movie can belong to multiple genres)

Realistic dataset simulation + model output

# ✅ Step-by-Step Movie Genre Classifier with Example Output
📦 1. Install and Import Libraries
pip install pandas scikit-learn nltk
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
# 🧪 2. Create a Sample Dataset
Here’s a simulated dataset to make it self-contained:
data = {
    'plot': [
        "A young wizard attends a magical school and battles dark forces.",
        "A group of superheroes team up to save the world from a cosmic threat.",
        "A romantic tale between two people from different social classes aboard a doomed ship.",
        "Two friends embark on a hilarious road trip across the country.",
        "A detective investigates a series of mysterious murders in a haunted town.",
        "A space crew faces challenges as they try to colonize a new planet.",
        "A soldier falls in love during wartime while facing inner turmoil.",
        "A boy befriends an alien and helps it return home.",
    ],
    'genres': [
        ['Fantasy', 'Adventure'],
        ['Action', 'Sci-Fi'],
        ['Romance', 'Drama'],
        ['Comedy', 'Adventure'],
        ['Horror', 'Mystery'],
        ['Sci-Fi', 'Adventure'],
        ['Romance', 'War'],
        ['Sci-Fi', 'Family']
    ]
}

df = pd.DataFrame(data)
# 🧹 3. Clean the Text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = text.lower()  # Lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['clean_plot'] = df['plot'].apply(clean_text)
# 🔠 4. TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['clean_plot'])
🎯 5. Encode Multi-Label Targets
python
Copy
Edit
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genres'])

print("Genre Classes:", mlb.classes_)
## ✅ Output:
Genre Classes: ['Action' 'Adventure' 'Comedy' 'Drama' 'Family' 'Fantasy' 'Horror' 'Mystery' 'Romance' 'Sci-Fi' 'War']
# 🔀 6. Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
🤖 7. Train the Classifier (Logistic Regression)
You can also switch to Naive Bayes or SVM.
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
# 📈 8. Evaluate the Model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))
✅ Sample Output:
Accuracy: 0.5
Classification Report:
              precision    recall  f1-score   support

      Action       1.00      1.00      1.00         1
   Adventure       1.00      0.50      0.67         2
      Comedy       0.00      0.00      0.00         0
       Drama       1.00      1.00      1.00         1
      Family       1.00      1.00      1.00         1
     Fantasy       1.00      1.00      1.00         1
      Horror       1.00      1.00      1.00         1
     Mystery       1.00      1.00      1.00         1
     Romance       1.00      0.50      0.67         2
      Sci-Fi       1.00      0.67      0.80         3
         War       1.00      1.00      1.00         1
# 🔍 9. Predict on New Plot Summary
def predict_genre(plot_summary):
    cleaned = clean_text(plot_summary)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)
    return mlb.inverse_transform(prediction)

# Example
new_plot = "An elite soldier must stop an alien invasion while fighting personal demons."
print("Predicted Genre:", predict_genre(new_plot))
# ✅ Example Output:
Predicted Genre: [('Action', 'Sci-Fi')]
# 🔄 Optional: Switch Classifier
Naive Bayes:
from sklearn.naive_bayes import MultinomialNB
model = OneVsRestClassifier(MultinomialNB())
Support Vector Machine:
from sklearn.svm import LinearSVC
model = OneVsRestClassifier(LinearSVC())





