
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% [markdown]
# ## Importing the dataset

# %%
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# %% [markdown]
# ## Cleaning the texts

# %%
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

# %% [markdown]
# ## Creating the Bag of Words model

# %%
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# %% [markdown]
# ## Splitting the dataset into the Training set and Test set

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# %% [markdown]
# ## Training the SVC Linear model on the Training set

# %%
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# %% [markdown]
# ## Predicting the Test set results

# %%
y_pred = classifier.predict(X_test)

# %% [markdown]
# ## Confusion Matrix

# %%
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
score = accuracy_score(y_test, y_pred)
print("Accuracy Score will be :",score*100,"%")

# %% [markdown]
# ## Predicting if a single review is positive or negative

# %%
review = input("Write any Positive/Negative Feedback(For ex - I loved the ambiance of the place): ")


# %%
review = re.sub('[^a-zA-Z]', ' ', review)
review = review.lower()
review = review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
review = ' '.join(review)
new_corpus = [review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
if(new_y_pred == 1):
    print("It's a Positive Feedback")
else:
    print("It's a Negative Feedback")


