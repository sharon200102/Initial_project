import numpy as np
from sklearn.metrics import roc_curve, auc


def split_data(data,cl,ratio):
  msk = np.random.rand(len(data)) < ratio
  trainX = data[msk]
  trainY=np.asarray(cl)[msk]
  testX=data[~msk]
  testY=np.asarray(cl)[~msk]
  return trainX,trainY,testX,testY

# Perform cross validation on the data using the kfold object inserted and evaluated by accuracy or auc.
def cross_validation(kf,data,labels,model,eval_fn_name="Accuracy"):
  score_per_fold=[]
  for train_index, test_index in kf.split(data):
    # Split the data according tho the kfold object.
    X_train, X_test, y_train, y_test = data.iloc[train_index], data.iloc[test_index], labels.iloc[train_index], labels.iloc[test_index]
    model.fit(X_train, y_train)
    # Evaluate the data according to the inserted function name.
    if eval_fn_name== "Accuracy":
      score_per_fold.append(model.score(X_test,y_test))
    elif eval_fn_name== "Auc":
      y_scores = model.predict_proba(X_test)
      fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
      score_per_fold.append(auc(fpr, tpr))
    elif eval_fn_name== "R2":
      score_per_fold.append((model.score(X_test,y_test)))
    else:
      raise NameError("Couldn't recognize the evaluation function inserted")


  return np.mean(score_per_fold)