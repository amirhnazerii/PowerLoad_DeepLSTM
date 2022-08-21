# Load Time-series Forecasting via Multivariate Deep LSTM Neural Network

> This repository is the source code for the Accepted paper named *Machine Learning-assisted Energy Management System in an Islanded Microgrid with Resiliency Investigation against Cyber-Physical Attacks* in the 54th annual North American Power Symposium (NAPS 2022), Salt Lake City, Utah. 

### Requirements
* Python 3.9.x
* TensorFlow 2.9.0
* Keras 2.9.0

## A. Data Preparation and Preprocessing
The load dataset used for the training is one year of load demand from Jan-1st 2020 to Jan-1st 2021 borrowed from The New York Independent System Operator that is cleaned and uploaded as a CSV file on this repository. The load datapoints are recorded with a time interval of 5 minutes. In the first phase of the data preparation the dataset it cleaned by removing the null values, outliers, duplicated and missing timestamps. Figure below shows the dataset load demand for a year.

<p align="center">
  <img src="/Images_onGit/LoadData.png" width=500>
</p>

The time features are extracted and concatenated with the load data as the inputs to the machine learning model. The extracted time features in this study include Minutes, Hours, Daylight, Day-of-Week, and Weekdays. In the next step, a Min-Max scaler function normalizes the inputs in the range [0, 1] so the algorithm will converge faster.   
Then, keeping the temporal order of timestamps, the dataset is divided into the train, validation, and test sets. Firstly, the dataset is split into train-validation and test sets by 99/1 ratio. Then the train-validation set is split into train and validation subsets by ratio of 80/20.


## B. Deep LSTM Machine Learning Model
In this section a deep LSTM model is proposed to predict electric load demand that is formulated as a multivariate multi-step time-series forecasting problem. The proposed deep LSTM model forecasts one hour (12 data points) in the future by looking at one day in the past (288 data points).
The extracted time features in the section II-A are normalized and concatenated with the normalized load data and set as input to LSTM model. The model architecture consists of input layer, two hidden LSTM layers along with two Dropout layers at the end of each LSTM layer. Finally, a fully connected feed-forward dense layer is placed at the end of the second hidden layer. Validation loss is selected as the metric to evaluate the performance of the trained LSTM model. python 3.9.12 and TensorFlow 2.9.0 are utilized to implement the deep LSTM model. The model’s hyperparameters are optimized through the Random Search technique using the Keras-tuner from Keras module. The hyperparameters after optimization are as below:  two hidden layers of LSTM with 128 neurons each, a fully connected feed-forward dense layer with 12 neurons, two dropout regularization layers after each hidden layer with value of 0.2 each, and LeakyReLU with alpha = 0.5 is considered as activation function for LSTM hidden layer. Adam optimization algorithm with learning rate of 0.0003 is selected for the stochastic gradient descent for model training. Figure below demonstrates the architecture of the proposed deep LSTM model.

<p align="center">
  <img src="/Images_onGit/DeepLSTM_model.png" width=500>
</p>


## C. Load Forecasting Results
The dataset is split into three subsets, training, testing and validation sets. The date starting 1-Jan-2020 to 5-Oct-2020 (80000 datapoints) is assigned as training set. 6-Oct-2020 to 14-Dec-2020 (20000 datapoints), and 15-Dec-2020 to 26-Dec-2020 (1500 datapoints) are set as validation and test subsets, respectively. The model is trained by the computational help of an NVIDIA GPU GeForce RTX(TM) 3070 8GB GDDR6. The batch size and number of training epochs are 256 and 50 after fine tuning, respectively. The Mean Squared Error (MSE) for the forecasted data on the training set is 4e-4 while it is 1.5e-4 on the validation set. Figure shows the evaluation of the trained deep LSTM model on the test set compared to the actual data. The time interval between two records is 5 minutes.

<p align="center">
  <img src="/Images_onGit/Test_data_evaluation.png" width=600>
</p>


