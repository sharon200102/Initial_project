import numpy as np


def split_data(data,cl,ratio):
  msk = np.random.rand(len(data)) < ratio
  trainX = data[msk]
  trainY=np.asarray(cl)[msk]
  testX=data[~msk]
  testY=np.asarray(cl)[~msk]
  return trainX,trainY,testX,testY

def cross_validation(kf,data,labels,model):
  score_per_fold=[]
  for train_index, test_index in kf.split(data):
    X_train, X_test, y_train, y_test = data.iloc[train_index], data.iloc[test_index], labels.iloc[train_index], labels.iloc[test_index]
    model.fit(X_train, y_train)
    score_per_fold.append(model.score(X_test,y_test))
  return np.mean(score_per_fold)