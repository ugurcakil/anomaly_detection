# METODLAR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
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
import matplotlib.backends.backend_qt5agg as mpl_backend
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA



def zscore_method(df):

    def calculate_zscore(column):
        zscore = (column - column.mean()) / column.std()
        return zscore

    # Calculate Z-Score and mark anomalies
    anomalies = []
    for column in df.columns[1:]:
        zscore = calculate_zscore(df[column])
        threshold = 2.5  # Set the threshold value, you can adjust it as needed

        column_anomalies = np.where(np.abs(zscore) > threshold)[0]  # Flatten the indices
        for anomaly in column_anomalies:
            anomaly_data = {
                'Column': column,
                'Index': anomaly,
                'Timestamp': df['timestamp'].iloc[anomaly],
                'Value': df[column].iloc[anomaly]
            }
            anomalies.append(anomaly_data)

    # Save anomalies to a CSV file
    anomalies_df = pd.DataFrame(anomalies)
    return anomalies_df

def lof_method(df):
    # Anomalileri depolamak için boş bir liste oluştur
    anomalies = []

    # Her bir sütun için LOF uygula
    for column in df.columns[1:]:
        X = np.array(df[column]).reshape(-1, 1)

        # LOF modelini oluştur
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)

        # LOF'yu veriye uygula ve anomali skorlarını al
        anomaly_scores = lof.fit_predict(X)

        # Anomalileri belirle
        column_anomalies = np.where(anomaly_scores == -1)[0]

        # Anomalileri listeye ekle
        for anomaly in column_anomalies:
            anomaly_data = {
                'Column': column,
                'Index': anomaly,
                'Timestamp': df['timestamp'].iloc[anomaly],
                'Value': df[column].iloc[anomaly]
            }
            anomalies.append(anomaly_data)

    # Anomalileri DataFrame'e dönüştür
    anomalies_df = pd.DataFrame(anomalies)
    return anomalies_df

def autoencoder_method(df):
    # Anomalileri depolamak için boş bir liste oluştur
    anomalies = []

    # Her bir sütun için Autoencoder uygula
    for column in df.columns[1:]:
        X = np.array(df[column]).reshape(-1, 1)

        # Veriyi normalleştir
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Autoencoder modelini oluştur
        input_dim = X_scaled.shape[1]
        encoding_dim = int(input_dim / 2)

        autoencoder = Sequential()
        autoencoder.add(Dense(encoding_dim, activation='relu', input_shape=(input_dim,)))
        autoencoder.add(Dense(input_dim, activation='linear'))

        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        # Autoencoder'ı veriye uygula ve çıktıları al
        autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, shuffle=True, verbose=0)
        encoded_data = autoencoder.predict(X_scaled)

        # Hata hesapla
        errors = np.mean(np.square(X_scaled - encoded_data), axis=1)

        # Anomalileri belirle
        column_anomalies = np.where(errors > np.mean(errors) + 2 * np.std(errors))[0]

        # Anomalileri listeye ekle
        for anomaly in column_anomalies:
            anomaly_data = {
                'Column': column,
                'Index': anomaly,
                'Timestamp': df['timestamp'].iloc[anomaly],
                'Value': df[column].iloc[anomaly]
            }
            anomalies.append(anomaly_data)

    # Anomalileri DataFrame'e dönüştür
    anomalies_df = pd.DataFrame(anomalies)
    return anomalies_df

def isolation_forest_method(df):
    # Anomalileri depolamak için boş bir liste oluştur
    anomalies = []

    # Her bir sütun için Isolation Forest uygula
    for column in df.columns[1:]:
        X = np.array(df[column]).reshape(-1, 1)

        # Isolation Forest modelini oluştur
        isolation_forest = IsolationForest(contamination=0.05)

        # Isolation Forest'ı veriye uygula ve anomali skorlarını al
        anomaly_scores = isolation_forest.fit_predict(X)

        # Anomalileri belirle
        column_anomalies = np.where(anomaly_scores == -1)[0]

        # Anomalileri listeye ekle
        for anomaly in column_anomalies:
            anomaly_data = {
                'Column': column,
                'Index': anomaly,
                'Timestamp': df['timestamp'].iloc[anomaly],
                'Value': df[column].iloc[anomaly]
            }
            anomalies.append(anomaly_data)

    # Anomalileri DataFrame'e dönüştür
    anomalies_df = pd.DataFrame(anomalies)
    return anomalies_df

def one_class_svm_method(df):
    # Anomalileri depolamak için boş bir liste oluştur
    anomalies = []

    # Her bir sütun için One-Class SVM uygula
    for column in df.columns[1:]:
        X = np.array(df[column]).reshape(-1, 1)

        # One-Class SVM modelini oluştur
        one_class_svm = OneClassSVM(nu=0.05)

        # One-Class SVM'yi veriye uygula ve anomali skorlarını al
        anomaly_scores = one_class_svm.fit_predict(X)

        # Anomalileri belirle
        column_anomalies = np.where(anomaly_scores == -1)[0]

        # Anomalileri listeye ekle
        for anomaly in column_anomalies:
            anomaly_data = {
                'Column': column,
                'Index': anomaly,
                'Timestamp': df['timestamp'].iloc[anomaly],
                'Value': df[column].iloc[anomaly]
            }
            anomalies.append(anomaly_data)

    # Anomalileri DataFrame'e dönüştür
    anomalies_df = pd.DataFrame(anomalies)
    return anomalies_df

def robust_covariance_method(df):
    # Anomalileri depolamak için boş bir liste oluştur
    anomalies = []

    # Her bir sütun için Robust Covariance uygula
    for column in df.columns[1:]:
        X = np.array(df[column]).reshape(-1, 1)

        # Robust Covariance modelini oluştur
        robust_covariance = EllipticEnvelope(contamination=0.05)

        # Robust Covariance'ı veriye uygula ve anomali skorlarını al
        anomaly_scores = robust_covariance.fit_predict(X)

        # Anomalileri belirle
        column_anomalies = np.where(anomaly_scores == -1)[0]

        # Anomalileri listeye ekle
        for anomaly in column_anomalies:
            anomaly_data = {
                'Column': column,
                'Index': anomaly,
                'Timestamp': df['timestamp'].iloc[anomaly],
                'Value': df[column].iloc[anomaly]
            }
            anomalies.append(anomaly_data)

    # Anomalileri DataFrame'e dönüştür
    anomalies_df = pd.DataFrame(anomalies)
    return anomalies_df

def pca_method(df):
    # Anomalies list to store the detected anomalies
    anomalies = []

    # Apply PCA for each column
    for column in df.columns[1:]:
        X = np.array(df[column]).reshape(-1, 1)

        # Perform PCA
        pca = PCA(n_components=1)
        transformed = pca.fit_transform(X)

        # Calculate the reconstruction error
        reconstruction_error = np.abs(X - pca.inverse_transform(transformed))

        # Set a threshold for anomaly detection (example: 3 standard deviations)
        threshold = np.mean(reconstruction_error) + 3 * np.std(reconstruction_error)

        # Identify anomalies based on the threshold
        column_anomalies = np.where(reconstruction_error > threshold)[0]

        # Append anomalies to the list
        for anomaly in column_anomalies:
            anomaly_data = {
                'Column': column,
                'Index': anomaly,
                'Timestamp': df['timestamp'].iloc[anomaly],
                'Value': df[column].iloc[anomaly]
            }
            anomalies.append(anomaly_data)

    # Convert anomalies to a DataFrame
    anomalies_df = pd.DataFrame(anomalies)
    return anomalies_df



def exponential_smoothing_method(df):
    # Ignore all warnings
    warnings.filterwarnings("ignore")
    alpha = 0.3
    def detect_anomalies(series):
        model = ExponentialSmoothing(series, initialization_method="heuristic").fit(smoothing_level=alpha)
        smoothed = model.fittedvalues
        residuals = series - smoothed
        zscore = (residuals - residuals.mean()) / residuals.std()
        threshold = 2.5  # Set the threshold value, you can adjust it as needed

        anomalies = np.where(np.abs(zscore) > threshold)[0]  # Flatten the indices
        return anomalies

    anomalies = []
    for column in df.columns[1:]:
        column_anomalies = detect_anomalies(df[column])
        for anomaly in column_anomalies:
            anomaly_data = {
                'Column': column,
                'Index': anomaly,
                'Timestamp': df['timestamp'].iloc[anomaly],
                'Value': df[column].iloc[anomaly]
            }
            anomalies.append(anomaly_data)

    anomalies_df = pd.DataFrame(anomalies)
    return anomalies_df


def detect_anomalies_MA(df):
    # Anomalies list to store the detected anomalies
    anomalies = []
    window_size = 10
    threshold = 3

    # Iterate over each column
    for column in df.columns[1:]:
        # Prepare the data for moving average
        data = pd.DataFrame()
        data['timestamp'] = df['timestamp']
        data['value'] = df[column]



        # Compute the moving average
        moving_avg = data['value'].rolling(window=window_size).mean()

        # Calculate the residuals
        residuals = np.abs(data['value'] - moving_avg)

        # Set a threshold for anomaly detection
        # (example: threshold = mean + 3 * standard deviation of residuals)
        anomaly_threshold = np.mean(residuals) + threshold * np.std(residuals)

        # Identify anomalies based on the threshold
        column_anomalies = np.where(residuals > anomaly_threshold)[0]

        # Append anomalies to the list
        for anomaly in column_anomalies:
            anomaly_data = {
                'Column': column,
                'Index': anomaly,
                'Timestamp': data['timestamp'].iloc[anomaly],
                'Value': data['value'].iloc[anomaly]
            }
            anomalies.append(anomaly_data)

    # Convert anomalies to a DataFrame
    anomalies_df = pd.DataFrame(anomalies)
    return anomalies_df



def prophet_method(df):
    # Anomalies list to store the detected anomalies
    anomalies = []

    # Iterate over each column
    for column in df.columns[1:]:
        # Prepare the data for Prophet
        data = pd.DataFrame()
        data['ds'] = df['timestamp']
        data['y'] = df[column]

        # Create a Prophet model
        model = Prophet()

        # Fit the model to the data
        model.fit(data)

        # Make predictions
        predictions = model.predict(data)

        # Calculate the residuals
        residuals = np.abs(data['y'] - predictions['yhat'])

        # Set a threshold for anomaly detection (example: 3 standard deviations)
        threshold = np.mean(residuals) + 3 * np.std(residuals)

        # Identify anomalies based on the threshold
        column_anomalies = np.where(residuals > threshold)[0]

        # Append anomalies to the list
        for anomaly in column_anomalies:
            anomaly_data = {
                'Column': column,
                'Index': anomaly,
                'Timestamp': df['timestamp'].iloc[anomaly],
                'Value': df[column].iloc[anomaly]
            }
            anomalies.append(anomaly_data)

    # Convert anomalies to a DataFrame
    anomalies_df = pd.DataFrame(anomalies)
    return anomalies_df


def detect_anomalies_ARIMA(df):
    # Anomalies list to store the detected anomalies
    anomalies = []

    # Iterate over each column
    for column in df.columns[1:]:
        # Prepare the data for ARIMA
        data = pd.DataFrame()
        data['timestamp'] = df['timestamp']
        data['value'] = df[column]

        # Fit the ARIMA model
        model = ARIMA(data['value'], order=(1, 0, 0))  # ARIMA(p, d, q) model, burada p=1, d=0, q=0
        model_fit = model.fit()

        # Make predictions
        predictions = model_fit.predict()

        # Calculate the residuals
        residuals = np.abs(data['value'] - predictions)

        # Set a threshold for anomaly detection (example: 3 standard deviations)
        threshold = np.mean(residuals) + 3 * np.std(residuals)

        # Identify anomalies based on the threshold
        column_anomalies = np.where(residuals > threshold)[0]

        # Append anomalies to the list
        for anomaly in column_anomalies:
            anomaly_data = {
                'Column': column,
                'Index': anomaly,
                'Timestamp': data['timestamp'].iloc[anomaly],
                'Value': data['value'].iloc[anomaly]
            }
            anomalies.append(anomaly_data)

    # Convert anomalies to a DataFrame
    anomalies_df = pd.DataFrame(anomalies)
    return anomalies_df



# DF BİRLEŞTİRME

def unify_dataframes(*args):
    # Combine the input dataframes into a single dataframe
    combined_df = pd.concat(args)

    # Remove the 'Index' column
    combined_df = combined_df.drop('Index', axis=1)

    # Check for duplicate rows and update the 'Count' column
    combined_df['Count'] = combined_df.groupby(['Column', 'Timestamp'])['Column'].transform('count')

    # Remove duplicate rows
    df_unique = combined_df.drop_duplicates(subset=['Column', 'Timestamp'])

    # Sort the DataFrame based on the 'Count' column
    df_sorted = df_unique.sort_values('Count', ascending=False)

    # Convert 'Count' column to integer
    df_sorted['Count'] = df_sorted['Count'].astype(int)

    return df_sorted

def drop_rows_by_timestamp(dataframe, timestamp_list):
    #dataframe['Timestamp'] = pd.to_datetime(
    #    dataframe['Timestamp'])  # Convert Timestamp column to datetime if not already in that format
    #timestamp_list = pd.to_datetime(timestamp_list)  # Convert timestamp_list to datetime format

    dataframe = dataframe[~dataframe['Timestamp'].isin(timestamp_list)]  # Drop rows with timestamps in timestamp_list
    anomalies_df = pd.DataFrame(dataframe)
    return anomalies_df

# GRAFİK BASTIRMA FONKSİYONU

def print_graph(anomalies_df, df, timestamps_ver):
    if anomalies_df.empty or anomalies_df.isnull().values.all():
        columns = df.columns[1:]
        timestamps = pd.to_datetime(df['timestamp'])
        graphs = []

        for column in set(columns):

            fig, ax = plt.subplots(figsize=(12, 6))
            canvas = mpl_backend.FigureCanvas(fig)

            ax.plot(timestamps, df[column], label=column)


            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Value')
            ax.set_title('NO ANOMALY DETECTED')
            ax.legend()
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.grid(True)
            plt.tight_layout()

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            graphs.append(canvas)
        return graphs

    else:
        anomalies_df = drop_rows_by_timestamp(anomalies_df, timestamps_ver)
        columns = anomalies_df['Column']
        indices = anomalies_df['Index']

        timestamps = pd.to_datetime(df['timestamp'])

        anomaly_color = ['#FF0000']
        line_colors = ['#0000FF']

        graphs = []

        for column in set(columns):
            column_indices = indices[columns == column]
            line_color = random.choice(line_colors)

            fig, ax = plt.subplots(figsize=(12, 6))
            canvas = mpl_backend.FigureCanvas(fig)

            ax.plot(timestamps, df[column], label=column, color=line_color)
            ax.scatter(timestamps.iloc[column_indices], df[column].iloc[column_indices], color=anomaly_color,
                       marker='o', label='Anomaly')

            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Value')
            ax.set_title('Anomaly Detection - {}'.format(column))
            ax.legend()
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.grid(True)
            plt.tight_layout()

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            graphs.append(canvas)

        return graphs

#df = pd.read_csv("C:/Users/ugrck/PycharmProjects/bitirme/second_data_set/start_trial_count.csv")
# z_score_df = zscore_method(df)
# lof_df = lof_method(df)
#autoencoder_df = autoencoder_method(df)
# isolation_forest_df = isolation_forest_method(df)
# one_class_svm_df = one_class_svm_method(df)
# robust_covariance_df = robust_covariance_method(df)
# pca_df = pca_method(df)
# exponential_smoothing_df = exponential_smoothing_method(df)
# detect_anomalies_MA_df = detect_anomalies_MA(df)
# prophet_df = prophet_method(df)
# detect_anomalies_ARIMA_df = detect_anomalies_ARIMA(df)
#timestamps_ver = []
# print(len(z_score_df))
# print(len(lof_df))
#print(len(autoencoder_df))
#print_graph(autoencoder_df, df, timestamps_ver)
# print(len(isolation_forest_df))
# print(len(one_class_svm_df))
# print(len(robust_covariance_df))
# print(len(pca_df))
# print(len(exponential_smoothing_df))
# print(len(detect_anomalies_MA_df))
# print(len(prophet_df))
# print(len(detect_anomalies_ARIMA_df))
