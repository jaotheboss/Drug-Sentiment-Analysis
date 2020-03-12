from os import chdir
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import xgboost as xgb
import numpy as np
import seaborn as sns

# Extracting the Dataset from Local Storage
## training set
train = pd.read_csv('drugsComTrain_raw.tsv', sep = '\t')
train_drugs = train.loc[:, 'drugName']
train = train.loc[:, ['review', 'rating', 'usefulCount']]
train = train.iloc[[x[0] or x[1] for x in list(zip((train['rating'] <= 4), (train['rating'] >= 7)))], :]
## test set
test = pd.read_csv('drugsComTest_raw.tsv', sep = '\t')
test = test.loc[:, ['review', 'rating', 'usefulCount']]
test = test.iloc[[x[0] or x[1] for x in list(zip((test['rating'] <= 4), (test['rating'] >= 7)))], :]

# Pre Processing
stop_words = stopwords.words('english')
wnl = WordNetLemmatizer()

def preprocess(text_column):
       # Remove link,user and special characters
       # And Lemmatize the words
       new_review = []
       for review in text_column:
              # this text is a list of tokens for the review
              text = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ', str(review).lower()).strip()
              text = [wnl.lemmatize(i) for i in text.split(' ') if i not in stop_words]
              new_review.append(' '.join(text))
       return new_review

train['review'] = preprocess(train['review'])
test['review'] = preprocess(test['review'])

# Data Exploration of Training Set
## proportion of positive and negative reviews
n_positive = 0
n_negative = 0
n_ratings = Counter(train['rating'])
for i in n_ratings.keys():
       if i >= 7:
              n_positive += n_ratings[i]
       elif i <= 4:
              n_negative += n_ratings[i]
plt.bar(['Positive', 'Negative'], [n_positive, n_negative])
plt.title('Proportion of Positive and Negative Reviews')
plt.ylabel('Count')
plt.show()

## top reviewed drugs
top_25 = train_drugs.value_counts()[:25] # looking at the top 25
top_25 = top_25.reset_index()
top_25.columns = ['drugName', 'Count']
sns.barplot(x = top_25.loc[:, 'Count'], y = top_25.loc[:, 'drugName'])

## counting the frequencies of words
train_pos = train.loc[train['rating'] >= 7, 'review']
train_neg = train.loc[train['rating'] <= 4, 'review']

total_freq = dict()
unique_freq = dict()
for review in train_pos:
       # this text is a list of tokens for the review
       text = review.split(' ')
       freq = nltk.FreqDist(text)
       for token, count in freq.most_common(20):
              if token in total_freq.keys():
                     total_freq[token] += count
                     unique_freq[token] += 1
              else:
                     total_freq[token] = count
                     unique_freq[token] = 1
## top 20 words for positive reviews
sorted(total_freq.items(), key = lambda x: x[1], reverse = True)[:20]
sorted(unique_freq.items(), key = lambda x: x[1], reverse = True)[:20]

total_freq = dict()
unique_freq = dict()
for review in train_neg:
       # this text is a list of tokens for the review
       text = review.split(' ')
       freq = nltk.FreqDist(text)
       for token, count in freq.most_common(20):
              if token in total_freq.keys():
                     total_freq[token] += count
                     unique_freq[token] += 1
              else:
                     total_freq[token] = count
                     unique_freq[token] = 1
## top 20 words for positive reviews
sorted(total_freq.items(), key = lambda x: x[1], reverse = True)[:20]
sorted(unique_freq.items(), key = lambda x: x[1], reverse = True)[:20]

# Further Pre Processing
## splitting into X and y
## creating labels
train_labels = []
for rating in train['rating']:
       if rating >= 7:
              train_labels.append(1)
       elif rating <= 4:
              train_labels.append(-1)
       else:
              train_labels.append(0)

test_labels = []
for rating in test['rating']:
       if rating >= 7:
              test_labels.append(1)
       elif rating <= 4:
              test_labels.append(-1)
       else:
              test_labels.append(0)
              
## creating train_x and test_x
train_x = train['review']
test_x = test['review']
"""
What CountVectorizer does:
It creates one very large matrix with one column for every unique word in your corpus
(where the corpus is all 50k reviews in our case). Then we transform each review into 
one row containing 0s and 1s, where 1 means that the word in the corpus corresponding 
to that column appears in that review.
"""
## creates a matrix where the columns are the unique words and the each row is a review
## where the corresponding element indicate the presence of that particular word
cv = CountVectorizer(binary = True)
cv.fit(train['review'])
train_x = cv.transform(train_x)
test_x = cv.transform(test_x)

# Looking at different ways to tokenize each review
## This is for Word Count
cv = CountVectorizer(binary = False)
cv.fit(train_x)
k_train_x = cv.transform(train_x)
k_test_x = cv.transform(test_x)

## Returns the TF-IDF for each Word
tv = TfidfVectorizer(analyzer = 'word')
tv.fit(train_x)
k_train_x = tv.transform(train_x)
k_test_x = tv.transform(test_x)

## Returns the TF-IDF for each 2-gram Words
tv = TfidfVectorizer(analyzer = 'word', ngram_range = (2, 2))
tv.fit(train_x)
k_train_x = tv.transform(train_x)
k_test_x = tv.transform(test_x)

## Returns the TF-IDF for each n-gram character within Words
tv = TfidfVectorizer(analyzer = 'char_wb')
tv.fit(train_x)
k_train_x = tv.transform(train_x)
k_test_x = tv.transform(test_x)

# Logistic Regression Model Implementation
model = LogisticRegression(max_iter = train_x.shape[1])
model.fit(train_x, train_labels)
y_pred = model.predict(test_x)

## Log Reg Evaluation
accuracy_score(test_labels, y_pred) # 0.8742464801683798
f1_score(test_labels, y_pred) # 0.9150633505396528

# Decision Tree Model Implementation
tree_model = DecisionTreeClassifier() 
tree_model.fit(train_x, train_labels)
y_pred_tree = tree_model.predict(test_x)

## Decision Tree Evaluation
accuracy_score(test_labels, y_pred_tree) # 0.8943335308662157
f1_score(test_labels, y_pred_tree) # 0.927198749806417

# XGBoost
xgb_train_labels = []
for rating in train['rating']:
       if rating >= 7:
              xgb_train_labels.append(1)
       elif rating <= 4:
              xgb_train_labels.append(0)
       else:
              xgb_train_labels.append(None)

xgb_test_labels = []
for rating in test['rating']:
       if rating >= 7:
              xgb_test_labels.append(1)
       elif rating <= 4:
              xgb_test_labels.append(0)
       else:
              xgb_test_labels.append(None)
xgb_train = xgb.DMatrix(train_x, xgb_train_labels)
xgb_test = xgb.DMatrix(test_x, xgb_test_labels)
param = {'eta': 0.75,
         'max_depth': 50,
         'objective': 'binary:logitraw'}
## Training and Predicting
xgb_model = xgb.train(param, xgb_train, num_boost_round = 30)
y_pred_xgb = xgb_model.predict(xgb_test)
y_pred_xgb = np.where(np.array(y_pred_xgb) > 0.5, 1, -1)

# xgb Evaluation
accuracy_score(test_labels, y_pred_xgb) # 0.9247195373643664
f1_score(test_labels, y_pred_xgb) # 0.9483063452417704
recall_score(k_test_labels, y_pred_k_xgb)
precision_score(k_test_labels, y_pred_k_xgb)
confusion_matrix(k_test_labels, y_pred_k_xgb)

# Exploratory Methods for XGBoost
## looking at feature importance
xgb_scores = xgb_model.get_score(importance_type = 'weight')
xgb_scores = list(xgb_scores.items())
xgb_scores.sort(key = lambda x: x[1], reverse = True)
x = [i[0] for i in xgb_scores[:20]]
y = [i[1] for i in xgb_scores[:20]]
# plot
plot = sns.barplot(y, x)
plot.set_title('Top 20 Most Significant Variables', x = 0.66, weight = 'bold')
plot.set_xlabel('No. of times feature is used to split the data across all trees')

xgb_scores = xgb_model.get_score(importance_type = 'gain')
xgb_scores = list(xgb_scores.items())
xgb_scores.sort(key = lambda x: x[1], reverse = True)
x = [i[0] for i in xgb_scores[:20]]
y = [i[1] for i in xgb_scores[:20]]
# plot
plot = sns.barplot(y, x)
plot.set_title('Top 20 Most Significant Variables', x = 0.66, weight = 'bold')
plot.set_xlabel('How effective a feature is when used to split the data across all trees')

## a more streamline way would be:
xgb_model.feature_importances_
