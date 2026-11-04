import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import datetime
import time
import logging

# Replace with your API key details
API_KEY = 'PKS83XYYEBQDAQUMC11P'
API_SECRET = 'phV8PZYN9xZsubzhH3xBAjaqKAC4hW54VwdXKHq6'
APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'  # Paper trading base URL

# Initialize API connection
api = tradeapi.REST(API_KEY, API_SECRET, APCA_API_BASE_URL, api_version='v2')

SEQ_LENGTH = 4
MAX_POSITION_SIZE = 1000  # Maximum number of shares to buy/sell at once
TRANSACTION_COST = 0.001  # 0.1% transaction cost
MARGIN_REQUIREMENT = 0.5  # 50% margin requirement for short selling

# Setup logging
logging.basicConfig(filename='trading_log.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load model and scaler
model = load_model('model.keras')
scaler = joblib.load('scaler.pkl')

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(prices, slow=26, fast=12, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def fetch_data(symbol, interval='30Min', start='2023-06-23T09:30:00-04:00', end='2023-06-30T16:00:00-04:00'):
    df = api.get_bars(symbol, interval, start=start, end=end).df
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Calculate RSI and MACD
    df['RSI'] = compute_rsi(df['Close'])
    macd, macd_signal = compute_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal

    return df

def preprocess_data_for_prediction(data, scaler):
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']
    data_scaled = scaler.transform(data[required_columns])
    return data_scaled

def predict_next_closing_price(data_for_prediction):
    data_for_prediction = np.array([data_for_prediction[-SEQ_LENGTH:]])  # Reshape for the model
    predicted_price_scaled = model.predict(data_for_prediction)
    features_dummy = np.zeros((1, 7))
    features_dummy[:, 3] = predicted_price_scaled[0, 0]  # Assuming 'Close' is the 4th column
    predicted_price = scaler.inverse_transform(features_dummy)[:, 3]
    return predicted_price[0]

def execute_trade(symbol, prediction, current_price):
    if prediction > current_price * (1 + TRANSACTION_COST):
        qty_to_trade = min(int(10000 / current_price), MAX_POSITION_SIZE)
        print(f"Buying {qty_to_trade} shares at {current_price}")
        api.submit_order(symbol=symbol, qty=qty_to_trade, side='buy', type='market', time_in_force='gtc')
        logging.info(f"Bought {qty_to_trade} shares of {symbol} at {current_price}")
    elif prediction < current_price * (1 - TRANSACTION_COST):
        qty_to_trade = min(int(10000 / current_price), MAX_POSITION_SIZE)
        print(f"Selling {qty_to_trade} shares at {current_price}")
        api.submit_order(symbol=symbol, qty=qty_to_trade, side='sell', type='market', time_in_force='gtc')
        logging.info(f"Sold {qty_to_trade} shares of {symbol} at {current_price}")

def align_to_interval():
    now = datetime.datetime.now()
    wait_time = 1800 - (now.minute * 60 + now.second) % 1800  # Wait time until the next half-hour mark
    time.sleep(wait_time)

current_time = datetime.datetime.now()
logging.info(f"Script was triggered {current_time}")

try:
    
    while True:
        align_to_interval()  # Align execution to 30-minute intervals
        current_time = datetime.datetime.now()
        start_time = current_time - datetime.timedelta(minutes=30)
        
        # Fetch and process the data
        data = fetch_data('AAPL', start=start_time.isoformat(), end=current_time.isoformat())
        preprocessed_data = preprocess_data_for_prediction(data, scaler)
        
        # Predict and execute trades
        prediction = predict_next_closing_price(preprocessed_data)
        execute_trade('AAPL', prediction, data['Close'].iloc[-1])

except Exception as e:
    print(f"An error occurred: {e}")
    logging.error(f"An error occurred {e}", exc_info=True)