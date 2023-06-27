import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense

from methods import drop_rows_by_timestamp

def print_number_of_high_anomalies(df,timestamps):
    df = drop_rows_by_timestamp(df, timestamps)
    df = df[df['Count'] > 6]
    return len(df)

def display_high_anomalies(df, timestamps):

    df = df[df['Count'] > 6]
    df_ver2 = drop_rows_by_timestamp(df, timestamps)
    df_ver2 = df_ver2.to_string(index=False)
    print(df)
    return df_ver2

def print_number_of_mid_anomalies(df, timestamps):
    df = drop_rows_by_timestamp(df, timestamps)
    df = df[df['Count'] < 7]
    df = df[df['Count'] > 3]
    return (len(df))

def display_mid_anomalies(df,timestamps):
    df = df[df['Count'] < 7]
    df = df[df['Count'] > 3]
    df_ver2 = drop_rows_by_timestamp(df, timestamps)
    df_ver2 = df_ver2.to_string(index=False)
    return df_ver2

def print_number_of_low_anomalies(df, timestamps):
    df = drop_rows_by_timestamp(df, timestamps)
    df = df[df['Count'] < 4]
    return (len(df))

def display_low_anomalies(df,timestamps):
    df = df[df['Count'] < 4]
    df_ver2 = drop_rows_by_timestamp(df, timestamps)
    df_ver2 = df_ver2.to_string(index=False)
    return df_ver2

def display_all_anomalies(df,timestamps):
    if df.empty or df.isnull().values.all():
        return print("Not detected")
    else:
        df_ver2 = drop_rows_by_timestamp(df, timestamps)
        df_ver2 = df_ver2.to_string(index=False)
        return df_ver2

