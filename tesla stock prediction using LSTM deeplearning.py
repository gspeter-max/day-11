import pandas as pd

# Load the dataset, skipping the first row (which contains "TSLA" in all columns)
file_path = "/content/drive/My Drive/tesla_stock_data.csv"
df = pd.read_csv(file_path, skiprows=1)

# Display the first few rows to confirm the fix
print(df.head())

df.columns = ["Date", "Adj_Close", "Close", "High", "Low", "Open", "Volume"]
df = df.iloc[1:,:]


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df['Close']  = scaler.fit_transform(df[['Close']] )
df = df.set_index('Date')

def make_lstm_data(flod = 60,df = None):
    x,y = [], [] 
    for i in range(len(df)-flod-1):
        x.append(df.iloc[i:i + flod].values)
        y.append(df.iloc[i + flod].values)
    
    return  np.array(x), np.array(y)

main_data = df[['Close']]
x , y = make_lstm_data(df = main_data) 
from sklearn.model_selection import train_test_split 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout 
import  tensorflow as tf


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = Sequential([
        LSTM(100, return_sequences = True, input_shape = (60,1)), 
        Dropout(0.2), 
        LSTM(50, return_sequences = False),
        Dropout(0.2), 
        Dense(25), 
        Dense(1)
])

model.compile(
    optimizer = 'adam', loss = 'mse'
)

history = model.fit(
    x_train, y_train, 
    validation_data = (x_test, y_test), 
    verbose = 1, 
    epochs = 50, 
    batch_size = 32
)
model.save("/content/drive/My Drive/tesla_lstm_model.h5")
model.summary() 

model_predict = model.predict(x_test)

predict_data = scaler.inverse_transform(model_predict)
actual_data = scaler.inverse_transform(y_test.reshape(-1,1))

import matplotlib.pyplot as plt 

plt.figure(figsize = (10,6))
plt.plot(predict_data, label = 'predicted_data', color = 'blue')
plt.plot(actual_data , label = 'acutal_data', color = 'red')
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Tesla Stock Price Prediction using LSTM")
plt.legend()
plt.show()
