import numpy as np
import pandas as pd
def data_rearrangement(data):
    data=data.T
    data.insert(loc=0,column='#SampleID',value=data.index)
    data.reset_index(drop=True,inplace=True)
    data=data.rename_axis([None], axis=1).rename_axis('index')
    return data

def one_hot(data,categorical_columns):
  for col in categorical_columns:

  # Get one hot encoding of columns
    one_hot = pd.get_dummies(data[col])
    # Drop column B as it is now encoded
    data.drop(col,axis = 1,inplace=True)
    # Join the encoded df
    data = data.join(one_hot)
  return data

def conditional_identification(df,dic):
  mask = pd.DataFrame([df[key] == val for key, val in dic.items()]).T.all(axis=1)
  return df[mask]

def removeZeroCols(df):
  for col in df.columns:
    if list(df[col]).count(0)==len(df[col]):
      df.drop(col,axis=1,inplace=True)

def dropHighCorr(data,threshold):
  corr = data.corr()
  df_not_correlated = ~(corr.mask(np.tril(np.ones([len(corr)]*2, dtype=bool))).abs() > threshold).any()
  un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
  df_out = data[un_corr_idx]
  return df_out

def decrease_level(taxon_col,level,**kwargs):
  return taxon_col.apply(lambda x:"".join(x.split(**kwargs)[0:level]))
