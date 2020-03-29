from sklearn.preprocessing import RobustScaler
import pandas as pd
def Robust_std(df):
    transformer = RobustScaler().fit(df)
    return pd.DataFrame(transformer.transform(df), columns=df.columns)


def zscore_norm(df):
    return (df - df.mean()) / df.std()


def minmax_norm(df):
    return (df - df.min()) / (df.max() - df.min())
