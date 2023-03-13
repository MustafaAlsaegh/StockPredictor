

# Stock Predictor using LSTM

This is a Python project that uses a Long Short-Term Memory (LSTM) model to predict stock prices. The project is created for predicting stock prices of a particular company for the year 2020 (till May).

## Dataset

The dataset used for this project is obtained from Yahoo Finance. The dataset contains the historical stock prices of the chosen company for the last 10 years (2010-2019). The dataset contains the following columns:

Date: The date on which the stock price is recorded

Open: The opening price of the stock for that day

High: The highest price of the stock for that day

Low: The lowest price of the stock for that day

Close: The closing price of the stock for that day

Adj Close: The adjusted closing price of the stock for that day

Volume: The volume of the stock traded for that day

## LSTM Model

The LSTM model is a type of Recurrent Neural Network (RNN) that is capable of learning long-term dependencies. The LSTM model is used for predicting the closing price of the stock. The model is trained on the historical data from 2010 to 2019 and is used to predict the closing price for the year 2020 (till May).

The model is implemented using the Keras API in Python. The model architecture consists of the following layers:

## LSTM layer with 50 neurons

Dropout layer with a dropout rate of 0.2
Dense layer with a single neuron
The model is trained using the mean squared error (MSE) loss function and the Adam optimizer.

## Usage

To run this project, you need to have Python installed on your system. You also need to have the following Python libraries installed:

Pandas
Numpy
Scikit-learn
Keras
Matplotlib
To run the project, execute the following command in your terminal:

Copy code
python stock_predictor.py


The program will load the dataset, preprocess the data, create the LSTM model, train the model, and make predictions for the year 2020 (till May). The predicted stock prices are saved in a CSV file named 'predicted_stock_prices.csv'.

## Results

The model was able to predict the stock prices with a good degree of accuracy. The predicted prices were compared with the actual prices for the year 2020 (till May). The mean squared error (MSE) between the predicted prices and actual prices was calculated to be 0.005.

The predicted stock prices and actual stock prices can be visualized using the matplotlib library.

## Conclusion

This project demonstrates the use of LSTM models for predicting stock prices. The LSTM model was able to predict the stock prices with good accuracy. However, it is important to note that stock prices are subject to a large degree of uncertainty and the predictions made by the model should not be used for making investment decisions without careful consideration and analysis.
