from gensim.models.doc2vec import Doc2Vec
import pandas as pd
import numpy as np
from sklearn.svm import SVC
raw=pd.read_csv("../train.csv")
content=raw["content"]
sentiment_value=raw["sentiment_value"]
model= Doc2Vec.load("d2v.model")
vectors=[]
for i in range(len(content)):
    vectors.append(model.docvecs[str(i)])
vectors=np.array(vectors)
clf = SVC(gamma='auto')
clf.fit(vectors,sentiment_value.tolist())
model2= Doc2Vec.load("d2v_test.model")
test=pd.read_csv("../commit/result_tfidf.csv")
content_test=test["content"]
vectors_test=[]
for i in range(len(content_test)):
    vectors_test.append(model.infer_vector(content_test.values[i]))
vectors_test=np.array(vectors_test)
labels=clf.predict(vectors_test)
test["sentiment_value"]=labels
test["sentiment_value"]=test["sentiment_value"].astype(int)
test=test.drop("content",axis=1)
test.to_csv("svm.csv",index=False,encoding="UTF-8")