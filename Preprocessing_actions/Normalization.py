from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
def Robust_std(df):
    transformer = RobustScaler().fit(df)
    return pd.DataFrame(transformer.transform(df), columns=df.columns)


def zscore_norm(df):
    return (df - df.mean()) / df.std()


def minmax_norm(df):
    return (df - df.min()) / (df.max() - df.min())

def log_normalization(as_data_frame, eps_for_zeros=1):
    as_data_frame += eps_for_zeros
    as_data_frame = np.log10(as_data_frame)
    return as_data_frame
