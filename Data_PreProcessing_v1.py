import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

"""
Directory: C:/Users/anazeri/Downloads/ML/V2G/ML_Opt_RTDS_v3
file name: ML_Opt_RTDS_v3
Note: 
A multi-step LSTM model is designed. A chunck of load data will be the 
output of LSTM per ONE prediction. Multi-input-multi-output Model. 

Update:
06/16/2022    Replaced "scaler.fit_transform" with my "Normalizer" function

"""
##########################################################################################

# Read and plot the entire dataset
df_ini = pd.read_csv("C:/Users/anazeri/Downloads/ML/V2G/load estimation/Data preparation/A_year_load_data/Cleaned data/cleaned_year_2020pal_modif.csv").dropna()
df_ini['Time Stamp'] = pd.to_datetime(df_ini['Time Stamp'], infer_datetime_format=True)


# split the last 400 samples and assign them as forecast ground truth 

df_ini['Minute'] = df_ini['Time Stamp'].dt.minute
df_ini['hours'] = df_ini['Time Stamp'].dt.hour
df_ini['daylight'] = ((df_ini['hours'] >= 7) & (df_ini['hours'] <= 22)).astype(int)
df_ini['DayOfWeek'] = df_ini['Time Stamp'].dt.dayofweek
df_ini['WeekDay'] = (df_ini['DayOfWeek'] < 5).astype(int)

df = df_ini[["Load", "Minute",'hours', 'daylight', 'DayOfWeek', 'WeekDay']]

shift = 400 + 130
df_size = 100000
# df_unscaled = df.iloc[shift:shift+df_size+400]
df_train_unscaled = df.iloc[shift : shift+df_size]
# df_forecast_unscaled = df.iloc[     shift+df_size : shift+df_size+400]
# shift+df_size+testsize : shift+df_size+testsize+600

def Normalizer(unscaled_file, xmin, xmax, min, max):
    Norm_list = []
    """
    Output: Normalized batch of M-data : (M, N) where N is # of feauters, M = # of New_time_stamp
    """
    def MinMaxCal(X):

        """
        input: a single data point : 1D array : ["Load", "Minute",'hours', 'daylight', 'DayOfWeek', 'WeekDay'] for time t0
        output: Normalized signle data point: 1D array : (1, N)- N is # of feauters : 'Minute', 'hours', 'daylight', 'DayOfWeek', 'WeekDay'
        min = 0, max = 1, xmin = np.array([0, 0 , 0, 0, 0]), xmax = np.array([55, 23, 1, 6, 1])
        xmin, xmax are N-element arrayes for N features:  1D array
        """

        return ((X-xmin)/(xmax-xmin))*((max-min)+min)

    for i in range(len(unscaled_file)):
        Norm_list.append(MinMaxCal(unscaled_file[i,:]))
    Normalized_data = np.asarray(Norm_list)

    return Normalized_data

def DeNormalizer(scaled_file, xmin, xmax):

    DeNorm_list = []
    """
    Output: Normalized batch of M-data : (M, N) where N is # of feauters, M = # of New_time_stamp
    """
    def DeMinMaxCal(X):

        """
        input: a single data point : 1D array : ["Load", "Minute",'hours', 'daylight', 'DayOfWeek', 'WeekDay'] for time t0
        output: Normalized signle data point: 1D array : (1, N)- N is # of feauters : 'Minute', 'hours', 'daylight', 'DayOfWeek', 'WeekDay'
        min = 0, max = 1, xmin = np.array([0, 0 , 0, 0, 0]), xmax = np.array([55, 23, 1, 6, 1])
        xmin, xmax are N-element arrayes for N features:  1D array
        """

        return X*(xmax-xmin) +xmin

    for i in range(len(scaled_file)):
        DeNorm_list.append(DeMinMaxCal(scaled_file[i,:]))
    DeNormalized_data = np.asarray(DeNorm_list)

    return DeNormalized_data


 
min = 0
max = 1
XMIN = df["Load"].min()
XMAX = df["Load"].max()
x_min = np.array([XMIN, 0, 0, 0, 0, 0])
x_max = np.array([XMAX, 55, 23, 1, 6, 1])

data_scaled = Normalizer(df.to_numpy(), x_min, x_max, min, max)
data_scaled = pd.DataFrame(data_scaled)

forecast_len_tot = 2880

df_train = data_scaled.iloc[shift : shift + df_size]
df_forecast = data_scaled.iloc[     shift + df_size : shift+df_size + forecast_len_tot]
x_forecast_scaled = df_forecast.to_numpy()


#training/test split - sliding window
testsize = 0
predict_time = 1 
unroll_length = 2*288

#Training data
x_train0 = df_train.values

def unroll(data, label ,sequence_length,target_length):
    dataset = []
    labels = []
    for index in range(len(data) - sequence_length-target_length):
        dataset.append(data[index: index + sequence_length])
        labels.append(label[index + sequence_length: index + sequence_length + target_length])

    dataset_np = np.asarray(dataset)
    labels_np = np.asarray(labels)

    return dataset_np, labels_np

#Modified as LSTM standard input shape
future_len = 60


################################################

generate_RTDS_data = False

if generate_RTDS_data: 

    """
        Generate load data batches as RTDS outputs. 
    """
    df_forecast_unscaled = df.iloc[     shift+df_size : shift+df_size + forecast_len_tot]

    df_forecast_unscaled = df_forecast_unscaled.to_numpy()
    j = 0
    # create/save
    for j in range( int(forecast_len_tot/future_len) ):
        np.savetxt("C:/Users/anazeri/Downloads/ML/V2G/ML_Opt_RTDS_v3/Measdata_5hrs_future_len60/RTDS_PL_unscaled"+str(j+1)+".csv", df_forecast_unscaled[ j*future_len : (j+1)*future_len, 0])



