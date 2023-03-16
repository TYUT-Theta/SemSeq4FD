import pandas as pd, numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
test_term_doc = vec.transform(test[1]) #TF-IDF文档矩阵
x = trn_term_doc
test_x = test_term_doc
m = SVC(C=4,kernel='linear',probability=True)
m=m.fit(x, train[0])
preds=m.predict(test_x)

print("SVM RCED")
# dict = {'all_actual':test[0],'all_predicted':preds}
# #
# # savecsv=pd.DataFrame(dict)
# # savecsv.to_csv('weibo_SVM.csv',index=None)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print("accuracy_score:",accuracy_score(test[0],preds))
print("F1_score:",f1_score(test[0], preds, average="macro"))
print("precision_score:",precision_score(test[0], preds, average="macro"))
print("recall_score:",recall_score(test[0], preds, average="macro"))