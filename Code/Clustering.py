import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
import torch.nn as nn
import torch.nn.functional as F
def split_data(data,cl,ratio):
  msk = np.random.rand(len(data)) < ratio
  trainX = data[msk]
  trainY=np.asarray(cl)[msk]
  testX=data[~msk]
  testY=np.asarray(cl)[~msk]
  return trainX,trainY,testX,testY

# Perform cross validation on the data using the kfold object inserted and evaluated by accuracy or auc.
def cross_validation(kf,data,labels,model,eval_fn_name="Accuracy"):
  test_score_per_fold=[]
  train_score_per_fold=[]

  for train_index, test_index in kf.split(data):
    # Split the data according tho the kfold object.
    X_train, X_test, y_train, y_test = data.iloc[train_index], data.iloc[test_index], labels.iloc[train_index], labels.iloc[test_index]
    model.fit(X_train, y_train)
    # Evaluate the data according to the inserted function name.
    if eval_fn_name== "Accuracy":
      train_score_per_fold.append(model.score(X_train,y_train))
      test_score_per_fold.append(model.score(X_test,y_test))

    elif eval_fn_name== "Auc":
      y_train_scores = model.predict_proba(X_train)
      y_test_scores = model.predict_proba(X_test)
      fpr_train, tpr_train, threshold_train = roc_curve(y_train, y_train_scores[:, 1])
      fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_test_scores[:, 1])

      train_score_per_fold.append(auc(fpr_train, tpr_train))
      test_score_per_fold.append(auc(fpr_test, tpr_test))
    elif eval_fn_name== "R2":
      train_score_per_fold.append((model.score(X_train,y_train)))
      test_score_per_fold.append((model.score(X_test,y_test)))
    else:
      raise NameError("Couldn't recognize the evaluation function inserted")


  return np.mean(train_score_per_fold), np.mean(test_score_per_fold)


class learning_model(nn.Module):
  def __init__(self,structure_list,out_size):
    super(learning_model,self).__init__()
    self.linears=nn.ModuleList([nn.Linear(structure_list[i],structure_list[i+1]) for i in range(0,len(structure_list)-1)])
    self.out=nn.Linear(structure_list[-1], out_size)
  def forward(self,input):
    for layer in self.linears:
      input=F.tanh(layer(input))
    output=self.out(input)
    return output

  def predict(self, x, threshold=0.5):
    # Apply softmax to output.
    is_binary=self.out.out_features==2
    pred = F.softmax(x)
    ans = []
    # Pick the class with maximum weight
    for t in pred:
      if is_binary:
        if t[1] >= threshold:
          ans.append(1)
        else:
          ans.append(0)
      else:
        ans.append(t.argmax())
    return ans
  def predict_prob(self,input):
    return F.softmax(self(input))

def make_train_step(model, loss_fn, optimizer):
  # Builds function that performs a step in the train loop
  def train_step(x, y):
    # Sets model to TRAIN mode
    model.train()
    # Makes predictions
    yhat = model(x)
    # Computes loss
    loss = loss_fn(yhat,y)
    # Computes gradients
    loss.backward()
    # Updates parameters and zeroes gradients
    optimizer.step()
    optimizer.zero_grad()
    # Returns the loss
    return loss.item()

  # Returns the function that will be called inside the train loop
  return train_step

def weight_reset(m):
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    m.reset_parameters()
def calculate_f1_score(precision,recall):
  return 2 * (precision * recall) / (precision + recall)
def best_threshold(precision_arr, recall_arr, thresholds_arr, index_function=None, maximize=True):
  if index_function is None:
    index_function=calculate_f1_score
  index_values=[index_function(precision,recall) for precision,recall in zip(precision_arr,recall_arr)]
  if maximize:
    best_value=max(index_values)
  else:
    best_value = min(index_values)
  return thresholds_arr[index_values.index(best_value)],best_value

def early_stopping(history,patience=2,ascending=True):
  if len(history)<=patience:
    return False
  if ascending:
    return history[-patience-1]==max(history[-patience-1:])
  else:
    return history[-patience-1]==min(history[-patience-1:])





