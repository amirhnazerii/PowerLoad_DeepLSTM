import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 

import Data_PreProcessing_v1
from Data_PreProcessing_v1 import *

"""
Directory: C:/Users/anazeri/Downloads/ML/V2G/ML_Opt_RTDS_v3
file name: ML_Opt_RTDS_v3
Note: 
A multi-step LSTM model is designed. A chunck of load data will be the 
output of LSTM per ONE prediction. Multi-input-multi-output Model. 
06/19/2022    "forecast", "XinputUpdatedRTDS", and "func_plot" are updated.  
              
"""

LSTM_model = tf.keras.models.load_model('C:/Users/anazeri/Downloads/ML/V2G/LSTM_saved_model/lstm_model_multiOutput_07062022_future60')
LSTM_model.summary()

"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 576, 64)           18176     
                                                                 
 dropout (Dropout)           (None, 576, 64)           0         
                                                                 
 lstm_1 (LSTM)               (None, 64)                33024     
                                                                 
 dropout_1 (Dropout)         (None, 64)                0         
                                                                 
 dense (Dense)               (None, 12)                780       
                                                                 
=================================================================
Total params: 51,980
Trainable params: 51,980
Non-trainable params: 0
_________________________________________________________________
"""

##########################################################################################


def GenNEXTtimestamps(LASTtimestamps, num_timestamps, First_attampt ):


        # get the LAST time stamp 
        if First_attampt == True:
            dataset_original= LASTtimestamps
            start_date = dataset_original.iloc[shift+df_size-1]["Time Stamp"] 
        else:
            start_date =LASTtimestamps.iloc[num_timestamps-1]["Time Stamp Generated"] 

        # generate the NEXT time stamps
        Generated_timestamps= pd.date_range(start_date, periods=num_timestamps+1, freq="5min")
        Generated_timestamps= Generated_timestamps.to_frame(index = False, 
                                                            name = "Time Stamp Generated").iloc[1:].reset_index()  # dropped the first date, bc it was the last date of train_test dataset which we just used to generate the next 12(1hr) time stamps in the future
        Generated_timestamps= Generated_timestamps.drop(["index"], axis=1)
        return Generated_timestamps, start_date


def ExtractTimeIndex(New_time_stamp):
    
        '''
            Parameters
            ----------
            New_time_stamp : 1D Dataframe, size = (future_len, 1)

            Returns
            ------
            Extracted_Indices: np 1darray, size = (future_len ,1)
        '''

        New_time_stamp['Minute'] = New_time_stamp['Time Stamp Generated'].dt.minute
        New_time_stamp['hours'] = New_time_stamp['Time Stamp Generated'].dt.hour
        New_time_stamp['daylight'] = ((New_time_stamp['hours'] >= 7) & (New_time_stamp['hours'] <= 22)).astype(int)
        New_time_stamp['DayOfWeek'] = New_time_stamp['Time Stamp Generated'].dt.dayofweek
        New_time_stamp['WeekDay'] = (New_time_stamp['DayOfWeek'] < 5).astype(int)
        Extracted_Indices = New_time_stamp[['Minute' ,'hours', 'daylight', 'DayOfWeek', 'WeekDay']].to_numpy()
        return Extracted_Indices


def Normalizer(ExtractTimeIndices, xmin, xmax, min, max):
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

        for i in range(len(ExtractTimeIndices)):
            Norm_list.append(MinMaxCal(ExtractTimeIndices[i,:]))
        Normalized_timestamp = np.asarray(Norm_list)

        return Normalized_timestamp
    


def forecast(
            x_input, Normalized_new_timestamp):
   
        '''
            Parameters
            ----------
            x_input : 3darray
                standart input for LSTM, size = (1, unroll_length, num_features)
            Normalized_new_timestamp: 1d Dataframe, size = (future_len, 1)

            Returns
            ------
            y_forecasted: 1darray, size = (future_len ,1)
            x_input_update: 3darray, 
                standart input for LSTM, size = (1, unroll_length, num_features)
            
            Does forecast for t1 and update LSTM input for t2. 
            # x_input_temp = (1,unroll_length,num_features), y_pred_temp0 = (1, future_len)
            # y_pred_temp = (future_len, 1)
            #aa = (future_len, 1)(future_len, num_time_features=5) -> (future_len, num_features=6), Note: 1) future_len=num_timestamps
            # (1,unroll_length,num_features) = (1,unroll_length-num_timestamps,num_features)(1,future_len, num_features)

        '''
        x_input_temp = x_input

        aa = []
        y_pred_temp0 = LSTM_model.predict(x_input_temp) ;       
        y_pred_temp = y_pred_temp0.reshape(y_pred_temp0.shape[1], 1)        
        y_forecasted = Data_PreProcessing_v1.DeNormalizer(y_pred_temp, xmin = XMIN, 
                                                          xmax= XMAX )
        aa= np.append(y_pred_temp, Normalized_new_timestamp, axis =1)            
        x_input_update = np.append( x_input_temp[:, len(Normalized_new_timestamp):, :],[aa] , axis = 1)         
                                                
        return y_forecasted , x_input_update



def XinputUpdatedRTDS( 
            num_timestamps, original_data, replace_data):

        """
        replace the forecasted load data with sim/measured load data every 1hr.     
        Parameters
        ----------
        num_timestamps : int value (=future_len)
        original_data: 3darray
            standart input for LSTM, size = (1, unroll_length, num_features)
        replace_data: 1darray
            Recieved data from RTDS/Simulink, size = (1, future_len)

        Returns
        ------
        original_data: 3darray, (1, unroll_length, num_features)

        """
        original_data[0,-num_timestamps:, 0] =  replace_data  

        return original_data


def func_plot(
              past_data, future_len_sofar, predicted_future, true_future):
        """
        Parameters
        ----------
        past_data : 1darray, (100K, 1)
        future_len_sofar: int
        predicted_future: 1darray, (future_len, 1)
        true_future: 1darray, (future_len, 1)
       
        """
        plt.figure(figsize=(10,5))
        plt.plot(np.arange( len(past_data)), past_data, 
                    label = 'past data', color = 'blue' )
        plt.plot(np.arange( len(past_data)+future_len_sofar,  len(past_data)+future_len_sofar+future_len ), 
                    predicted_future, color = 'red', label = 'predicted future')
        plt.plot(np.arange( len(past_data)+future_len_sofar,  len(past_data)+future_len_sofar+future_len ), 
                    true_future, color = 'green', label ='true future')
        # plt.plot(np.arange( len(past_data), len(past_data)+len(true_future)), true_future["Load"], color = 'green', label ='true future')
        plt.xlim([ len(past_data)-700, len(past_data)+500])   #x_forecast_scaled
        plt.xlabel('time(5 mins)')
        plt.ylabel('load power (MWatt)')
        plt.legend()
        plt.show()





