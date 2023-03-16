import pandas as pd, numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
train = pd.read_csv(r'D:\fakenews630\fakenews630\newrumcedtrain.csv',engine='python',encoding='utf-8',header=None)
test = pd.read_csv(r'D:\fakenews630\fakenews630\newrumcedtest.csv',engine='python',encoding='utf-8',header=None)

train = train.sample(frac=1).reset_index(drop=True)
test = test.sample(frac=1).reset_index(drop=True)
#TF-IDF  特征提取
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
vec = TfidfVectorizer(ngram_range=(1,3), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,max_features=500
              )

trn_term_doc = vec.fit_transform(train[1])
x = trn_term_doc
lr = LogisticRegression()
lr.fit(x, train[0])
# preds=lr.predict(test_x)
preds=[]
for i in range(len(test)):
    test_term_doc = vec.transform(pd.Series(test[1][i]))
    test_x = test_term_doc
    preds.append(lr.predict(test_x)[0])
print("Logistic RCED")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print("accuracy_score:",accuracy_score(test[0],preds))
print("F1_score:",f1_score(test[0], preds, average="macro"))
print("precision_score:",precision_score(test[0], preds, average="macro"))
print("recall_score:",recall_score(test[0], preds, average="macro"))
