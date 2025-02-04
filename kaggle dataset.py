import os
import zipfile

# Move Kaggle API key to the right location
os.environ['KAGGLE_CONFIG_DIR'] = "/content/drive/My Drive/"

# Check if kaggle.json exists
if not os.path.exists("/content/drive/My Drive/kaggle.json"):
    print("⚠ kaggle.json not found! Upload it to Google Drive and try again.")
else:
    print("✅ Kaggle API key found!")

!kaggle competitions download -c rossmann-store-sales -p "/content/drive/My Drive/Rossmann_Dataset"
import os

file_path = "/content/drive/My Drive/Rossmann_Dataset/rossmann-store-sales.zip"

if os.path.exists(file_path):
    print("✅ Dataset downloaded successfully!")
else:
    print("❌ Dataset not found! Try downloading again.")
import zipfile

# Define file paths
dataset_zip = "/content/drive/My Drive/Rossmann_Dataset/rossmann-store-sales.zip"
extract_path = "/content/rossmann"

# Unzip dataset
with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Check extracted files
import os
print(os.listdir(extract_path))
import pandas as pd

# Load train dataset
train_df = pd.read_csv(f"{extract_path}/train.csv")
print(train_df.head())

# Load test dataset
test_df = pd.read_csv(f"{extract_path}/test.csv")
print(test_df.head())
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Display column names
print("Columns in Train:", train_df.columns)
print("Columns in Test:", test_df.columns)
train_df = train_df.set_index('Date')
test_df = test_df.set_index('Date')

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer , KNNImputer
from sklearn.ensemble import RandomForestRegressor


imputer = IterativeImputer(estimator = RandomForestRegressor(n_estimators=400), max_iter = 10, random_state = 42)

train_df[['Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo', 'SchoolHoliday']] = imputer.fit_transform(train_df[['Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo', 'SchoolHoliday']])

''' (optional)'''
# knn_imputer = KNNImputer(n_neighbors = 5)
# train_df[['Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo', 'SchoolHoliday']] = knn_imputer.fit_transform(train_df[['Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo', 'SchoolHoliday']])

import numpy as np
from scipy import stats

z_scores = np.abs(stats.zscore(train_df['Sales']))
df_zscore_cleaned = train_df[(z_scores < 3)]
# IQR
Q1 = train_df['Sales'].quantile(0.25)
Q3 = train_df['Sales'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

train_df = train_df[(train_df['Sales'] >= lower_bound) & (train_df['Sales'] <= upper_bound)]

print(train_df.memory_usage(deep = True))
train_df['StateHoliday'] = train_df['StateHoliday'].astype('category')
print(train_df.memory_usage(deep = True))

train_df = train_df.reset_index() 
train_df['Date'] = pd.to_datetime(train_df['Date'])

train_df['Year'] = train_df['Date'].dt.year
train_df['Month'] = train_df['Date'].dt.month
train_df['Weekday'] = train_df['Date'].dt.weekday  # Monday=0, Sunday=6
train_df['Week_of_Year'] = train_df['Date'].dt.isocalendar().week

# Display the DataFrame after extracting time-based features
print("Time-based Features:\n", train_df.head())

train_df['7_day_MA'] = train_df['Sales'].rolling(window=7).mean()
train_df['14_day_MA'] = train_df['Sales'].rolling(window=14).mean()
train_df['30_day_MA'] = train_df['Sales'].rolling(window=30).mean()

import numpy as np 
from numpy.fft import fft 

fft_values = fft(train_df['Sales'])
fft_abs = np.abs(fft_values)

import matplotlib.pyplot as plt 
plt.figure(figsize = (10,6))
plt.plot(fft_abs)
plt.title('Magnitude of FFT for Sales Data')
plt.xlabel('Frequency Index')
plt.ylabel('Magnitude')
plt.show()

from sklearn.preprocessing import TargetEncoder, OneHotEncoder
import category_encoders  as ce

target_encoder = TargetEncoder()
target_encoders = ce.TargetEncoder(cols = ['StateHoliday '])
one_hot_encoder = OneHotEncoder()

# train_df['StateHoliday'] = target_encoder.fit_transform(train_df[['StateHoliday']],train_df['Open']) 
one_hot_sparas = one_hot_encoder.fit_transform(train_df[['StateHoliday']]) 

monthly_avg_sales = train_df.groupby(['Store', 'Month'])['Sales'].mean().reset_index()# Example 2: Calculate average sales per store for each week
weekly_avg_sales = train_df.groupby(['Store', 'Weekday'])['Sales'].mean().reset_index()
weekly_avg_sales = train_df.groupby(['Store', 'Year'])['Sales'].mean().reset_index()

# Output results
print("Average Sales per Store (Monthly):")
print(monthly_avg_sales)

print("\nAverage Sales per Store (Weekly):")
print(weekly_avg_sales)

from pytrends.request import TrendReq 
import pandas as pd 

pytread = TrendReq(hl = 'en-US',tz = 360)
key_wards = ['market crash','economic growth', 'inflation']
pytread.build_payload(key_wards,timeframe = 'today 5-y',geo ='US')
pytread = pytread.interest_over_time()
trends_data = pytread.drop(columns = ['isPartial'])
trends_data = trends_data.reset_index()
print(trends_data)

import pandas_datareader as pdr 
from datetime import datetime 

start = datetime(2020,1,1)
end = datetime.today()

inflation_data = pdr.get_data_fred('CPIAUCSL',start, end)

inflation_data.rename(columns = {'CPIAUCSL':'inflation_rate'}, inplace = True)
inflation_data  = inflation_data.pct_change().dropna() 
print(inflation_data)
train_df = train_df.merge(trends_data,left_on = 'Date' ,right_on = 'date', how = "left")
train_df = train_df.merge(inflation_data, left_on = 'Date', right_on = 'DATE', how = "left")

''' after that inflation table columns and trends columns is nan becuase the date in train_df is around 2015 but in trends_data have dates in 2020 - 2025 so when you
use left than show nan becuase no same columns is exists ''' 
