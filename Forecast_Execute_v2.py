import numpy as np

from Data_PreProcessing_v1 import *
import Forecast_Funcs
from Forecast_Funcs import *
 
"""
Directory: C:/Users/anazeri/Downloads/ML/V2G/ML_Opt_RTDS_v3
file name: ML_Opt_RTDS_v3
Note: 

Update:
06/17/2022      "forecast_load" is modified. 

"""

## normalizing parameters for Time features. 
min = 0
max = 1
xmin = np.array([0, 0 , 0, 0, 0])
xmax = np.array([55, 23, 1, 6, 1])
num_features = 6
num_timestamps = future_len 
First_attampt = False



def forecast_load(New_timestamp, x_input_new):


    Extracted_Time_Indices= ExtractTimeIndex(New_timestamp)
    Normalized_new_timestamp = Forecast_Funcs.Normalizer(Extracted_Time_Indices, 
                                                            xmin, xmax, min, max)

    y_forecasted, y_forecasted_concat_X = forecast(x_input_new, Normalized_new_timestamp)

    return y_forecasted, y_forecasted_concat_X