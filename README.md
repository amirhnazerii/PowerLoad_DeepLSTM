# Load Time-series Forecasting via Multivariate Deep LSTM Neural Network

> This repository is the source code for the Accepted paper named *Machine Learning-assisted Energy Management System in an Islanded Microgrid with Resiliency Investigation against Cyber-Physical Attacks* in the 54th annual North American Power Symposium (NAPS 2022), Salt Lake City, Utah. 

## A. Data Preparation and Preprocessing
The load dataset used for the training is one year of load demand from Jan-1st 2020 to Jan-1st 2021 borrowed from The New York Independent System Operator that is cleaned and uploaded as a CSV file on this repository. The load datapoints are recorded with a time interval of 5 minutes. In the first phase of the data preparation the dataset it cleaned by removing the null values, outliers, duplicated and missing timestamps. Fig.1 shows the dataset load demand for a year.

![historical_load](https://github.com/amirhnazerii/PowerLoad_DeepLSTM/blob/main/Images_onGit/LoadData.png | width =100)

