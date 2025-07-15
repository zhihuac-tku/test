  # py code beginning

#import yfinance as yf
import pandas as pd
import streamlit as st
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.graph_objects as go

import requests

ALPHA_VANTAGE_API_KEY = "31UZ3H0NFLPVJPKL"


# --- App Config ---
st.set_page_config(page_title="LSTM Stock Predictor", layout="wide")
st.title("📈 Stock Direction Prediction with LSTM and Market Index Context")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"Using device: {device}")

# --- Sidebar: Inputs ---
st.sidebar.header("1️⃣ Select Stock")
popular_tickers = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.", "AMZN": "Amazon.com, Inc.",
    "TSLA": "Tesla Inc.",  "META": "Meta Platforms, Inc.",
    "NVDA": "NVIDIA Corporation", "NFLX": "Netflix, Inc.",
    "BRK-B": "Berkshire Hathaway Inc.", "JPM": "JPMorgan Chase & Co."
}
selected_symbol = st.sidebar.selectbox(
    "Choose a stock:", options=list(popular_tickers.keys()),
    format_func=lambda x: f"{x} - {popular_tickers[x]}"
)
manual_override = st.sidebar.text_input("Or enter a custom symbol (e.g., ^GSPC):", "").strip().upper()
ticker = manual_override if manual_override else selected_symbol

st.sidebar.header("2️⃣ Set Dates & Hyperparameters")
today = pd.Timestamp.today().normalize()
yesterday = today - pd.Timedelta(days=1)

# Date inputs return datetime.date; convert to pd.Timestamp immediately for consistency
def to_timestamp(date_obj):
    return pd.Timestamp(date_obj)

start_date = to_timestamp(st.sidebar.date_input("Data Start Date", pd.to_datetime("2015-01-01")))
end_date = to_timestamp(st.sidebar.date_input("Data End Date", today))
train_start_date = to_timestamp(st.sidebar.date_input("Train Start Date", pd.to_datetime("2018-01-01")))
train_end_date = to_timestamp(st.sidebar.date_input("Train End Date", pd.to_datetime("2023-12-31")))
forecast_target_date = to_timestamp(st.sidebar.date_input("Forecast Target Date", yesterday))

st.sidebar.markdown(
    """
    **📅 Date Selection Reminder**
    Please ensure:
    `start_date < train_start_date < train_end_date < forecast_target_date <= end_date`
    """
)

sequence_length = st.sidebar.number_input("Sequence Length (days)", 3, 60, 10)
batch_size = st.sidebar.number_input("Batch Size", 8, 128, 32)
epochs = st.sidebar.number_input("Epochs", 1, 50, 10)
learning_rate = st.sidebar.number_input("Learning Rate", 1e-5, 1e-1, 0.001, format="%.5f")

# --- Dataset Class ---
class StockDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_len):
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len

    def __len__(self):
        return len(self.df) - self.seq_len

    def __getitem__(self, idx):
        seq_x = self.df.iloc[idx:idx+self.seq_len][self.feature_cols].values.astype('float32')
        seq_y = self.df.iloc[idx+self.seq_len][self.target_col].astype('float32')
        return torch.tensor(seq_x), torch.tensor(seq_y)

# --- LSTM Classifier ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = hn[-1]
        out = self.fc(out)
        return self.sigmoid(out).squeeze()

# --- Training Function ---
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(dataloader.dataset)

# --- Evaluation Function ---
def evaluate(model, dataloader):
    model.eval()
    preds_all, y_true = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            preds = model(xb).cpu()
            preds_all.extend(preds.numpy())
            y_true.extend(yb.numpy())
    preds_label = [1 if p > 0.5 else 0 for p in preds_all]
    acc = accuracy_score(y_true, preds_label)
    cm = confusion_matrix(y_true, preds_label)
    return acc, cm, preds_all, y_true

def download_alpha_vantage_daily(symbol, start, end):
    """
    Download daily OHLCV data from Alpha Vantage API (free function).
    Returns a DataFrame indexed by date.
    """
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",  # ✅ 免費版 function
        "symbol": symbol,
        "outputsize": "full",
        "apikey": ALPHA_VANTAGE_API_KEY,
    }

    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        st.error(f"Alpha Vantage API request failed with status {response.status_code}")
        return pd.DataFrame()

    data = response.json()

    if "Error Message" in data:
        st.error(f"Alpha Vantage error: {data['Error Message']}")
        return pd.DataFrame()
    if "Note" in data:
        st.warning(f"Alpha Vantage API note: {data['Note']}")
        return pd.DataFrame()

    ts_key = "Time Series (Daily)"
    if ts_key not in data:
        st.error("Alpha Vantage response missing 'Time Series (Daily)' data.")
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(data[ts_key], orient="index")
    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. volume": "Volume",
    })

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.loc[(df.index >= start) & (df.index <= end)]
    df = df.apply(pd.to_numeric, errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def load_data_alpha_vantage(symbol, start, end):
    try:
        start = pd.to_datetime(start)
        end = min(pd.to_datetime(end), pd.Timestamp.today())

#        st.write(f"🔍 Downloading {symbol} data from Alpha Vantage from {start.date()} to {end.date()}...")


        st.write(f"⏳ Waiting for 15 seconds to respect API rate limits...")

        stock = download_alpha_vantage_daily(symbol, start, end)

        time.sleep(15)


        if stock.empty:
            st.error(f"❗ Stock data for {symbol} is empty or failed to load.")
            return pd.DataFrame()

#       st.write("📄 Raw stock data preview:")
#        st.dataframe(stock.head())

        # Use Adjusted Close as Close
        # stock["Close"] = stock["Adj Close"]

        # Alpha Vantage does not provide QQQ or VIX, so skip or implement separate download if you want.

        # Calculate technical indicators
        stock["SMA_10"] = SMAIndicator(stock["Close"], window=10).sma_indicator()
        stock["SMA_50"] = SMAIndicator(stock["Close"], window=50).sma_indicator()
        stock["RSI"] = RSIIndicator(stock["Close"]).rsi()
        macd = MACD(stock["Close"])
        stock["MACD"] = macd.macd()
        stock["MACD_signal"] = macd.macd_signal()

        # Targets and percent changes
        stock["Tomorrow_Close"] = stock["Close"].shift(-1)
        stock["Target"] = (stock["Tomorrow_Close"] > stock["Close"]).astype(int)

        stock.dropna(inplace=True)

#        st.write("✅ Final columns:")
#        st.write(stock.columns.tolist())

        return stock

    except Exception as e:
        st.error(f"❗ Error loading data: {e}")
        return pd.DataFrame()


# --- UI: Layout ---
left, right = st.columns([1, 2])

with left:
    if st.button("📊 Load & Show Price Chart"):
        safe_end = min(end_date + pd.Timedelta(days=1), pd.Timestamp.today())
        df = load_data_alpha_vantage(ticker, start_date, safe_end)
        if df.empty:
            st.error("Data is empty or failed to load.")
        else:
            st.session_state.loaded_data = df
#            st.success(f"Successfully loaded {len(df)} rows")

with right:
    if "loaded_data" in st.session_state:
        df = st.session_state.loaded_data
        st.subheader(f"{ticker} OHLC Candlestick Chart")
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=f"{ticker} OHLC"
        )])
        fig.update_layout(title=f"{ticker} OHLC Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)



# --- Training and Prediction ---
if st.button("🚀 Train and Predict"):
    # 日期驗證...
    if not (start_date < train_start_date < train_end_date < forecast_target_date <= end_date):
        st.warning("❗ Invalid date order. Please correct your dates.")
        st.stop()

    safe_end_for_download = min(end_date, forecast_target_date + pd.Timedelta(days=1), pd.Timestamp.today())
    st.write(f"Loading data from {start_date.date()} to {safe_end_for_download.date()} for training and testing...")
    # 同樣改成 Alpha Vantage 下載函數

    if "loaded_data" in st.session_state:
        data = st.session_state.loaded_data
    else:
        data = load_data_alpha_vantage(ticker, start_date, safe_end_for_download)

    if data.empty:
        st.error("❗ No data returned or data is empty.")
        st.stop()


# Filter data by date range
    features = ["SMA_10", "SMA_50", "RSI", "MACD", "MACD_signal"]

    target_col = "Target"
    train_df = data[(data.index >= train_start_date) & (data.index <= train_end_date)]
    test_df = data[data.index > train_end_date]

    st.write(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")

    if len(train_df) < sequence_length + 1 or len(test_df) < sequence_length + 1:
        st.warning("❗ Not enough data for training or testing. Please adjust date ranges.")
        st.stop()

    # Prepare Dataset and DataLoader
    train_ds = StockDataset(train_df, features, target_col, sequence_length)
    test_ds = StockDataset(test_df, features, target_col, sequence_length)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ld = DataLoader(test_ds, batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    model = LSTMClassifier(input_size=len(features)).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    progress = st.progress(0)
    status = st.empty()
    for ep in range(epochs):
        loss = train_epoch(model, train_ld, criterion, optimizer)
        acc, cm, _, _ = evaluate(model, test_ld)
        status.text(f"Epoch {ep+1}/{epochs} | Loss: {loss:.4f} | Test Acc: {acc:.2%}")
        progress.progress((ep + 1) / epochs)

    st.subheader("✅ Confusion Matrix")
    st.write(pd.DataFrame(cm, index=["Actual Down", "Actual Up"], columns=["Pred Down", "Pred Up"]))
    st.write(f"Test Accuracy: **{acc:.2%}**")


    # 將 model 與 data 存入 session_state
    st.session_state['model'] = model
    st.session_state['loaded_data'] = data
    st.session_state['features'] = features

# --- Optional: Multi-step forecast using predicted label feedback ---
def simulate_n_day_direction_forecast(model, df, features, sequence_length, n_days):
    simulated_df = df.copy()
    last_seq = simulated_df[-sequence_length:][features].values.astype("float32")
    preds = []
    synthetic_dates = []
    simulated_close = []
    up_pct=0.01
    down_pct=0.01


    current_close = simulated_df.iloc[-1]["Close"]
    last_date = simulated_df.index[-1]

    for i in range(n_days):
        input_tensor = torch.tensor(last_seq).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_prob = model(input_tensor).item()
        pred_label = 1 if pred_prob > 0.5 else 0
        preds.append((pred_prob, pred_label))

        # Simulate next close
        change_pct = up_pct if pred_label == 1 else -down_pct
        new_close = current_close * (1 + change_pct)
        simulated_close.append(new_close)
        current_close = new_close

        # Create synthetic row
        new_row = simulated_df.iloc[-1].copy()
        new_row["Close"] = new_close
        new_row["Open"] = new_close
        new_row["High"] = new_close * (1 + 0.005)
        new_row["Low"] = new_close * (1 - 0.005)

        # Add synthetic date (skip weekends)
        new_date = last_date + pd.Timedelta(days=1)
        while new_date.weekday() >= 5:
            new_date += pd.Timedelta(days=1)
        last_date = new_date
        synthetic_dates.append(new_date)

        # Append and recompute indicators
        temp_df = pd.concat([simulated_df, pd.DataFrame([new_row], index=[new_date])])
        temp_df["SMA_10"] = SMAIndicator(temp_df["Close"], window=10).sma_indicator()
        temp_df["SMA_50"] = SMAIndicator(temp_df["Close"], window=50).sma_indicator()
        temp_df["RSI"] = RSIIndicator(temp_df["Close"]).rsi()
        macd = MACD(temp_df["Close"])
        temp_df["MACD"] = macd.macd()
        temp_df["MACD_signal"] = macd.macd_signal()

        simulated_df = temp_df.dropna()
        last_seq = simulated_df[-sequence_length:][features].values.astype("float32")

    forecast_df = pd.DataFrame(preds, columns=["Pred_Prob", "Pred_Label"])
    forecast_df.index = synthetic_dates
    forecast_df["Simulated_Close"] = simulated_close
    return forecast_df


    # Prediction Results
    test_preds, test_true = [], []
    with torch.no_grad():
        for xb, yb in test_ld:
            out = model(xb.to(device))
            test_preds.extend(out.cpu().numpy())
            test_true.extend(yb.numpy())

    idx_offset = sequence_length
    pred_df = test_df.iloc[idx_offset:].copy()
    pred_df["Pred_Prob"] = test_preds
    pred_df["Pred_Label"] = (pred_df["Pred_Prob"] > 0.5).astype(int)

    st.subheader("📉 Prediction vs Close Price")
    st.line_chart(pred_df[["Close"]])
    st.line_chart(pred_df[["Target", "Pred_Label"]])

    # Forecast for specific target date with auto-adjustment
    if forecast_target_date not in data.index:
        forecast_target_date = data.index[-2]  # Use second-to-last date to have tomorrow's target
        st.info(f"🔄 Auto-adjusted forecast target date to **{forecast_target_date.date()}** (latest available).")

    seq_end_idx = data.index.get_loc(forecast_target_date) - 1  # use previous day as last seq day
    seq_start_idx = seq_end_idx - sequence_length + 1

    if seq_start_idx < 0:
        st.warning("⚠️ Not enough prior data for the selected sequence length.")
    else:
        last_seq = torch.tensor(
            data.iloc[seq_start_idx:seq_end_idx + 1][features].values,
            dtype=torch.float32
        ).unsqueeze(0).to(device)
    with torch.no_grad():
        prob_up = model(last_seq).item()
    st.subheader("🔮 Forecast Result")
    st.write(f"Probability that **{ticker}** increases on {forecast_target_date.date()}: **{prob_up:.2%}**")

# --- 多日預測功能，需確認 model 已訓練並存入 session_state ---
if 'model' in st.session_state and 'loaded_data' in st.session_state:
    model = st.session_state['model']
    df = st.session_state['loaded_data']
    features = st.session_state['features']

    st.subheader("🔁 Multi-Day Forecast (Simulated)")

    n_forecast_days = st.slider("How many future days to simulate?", 1, 15, 7)

    if st.button("🔮 Forecast N Future Days"):
        forecast_df = simulate_n_day_direction_forecast(
            model, df, features, sequence_length, n_days=n_forecast_days
        )

        st.subheader("📄 Multi-Day Forecast Table")
        st.dataframe(forecast_df)

        st.line_chart(forecast_df["Pred_Prob"], use_container_width=True)
        st.bar_chart(forecast_df["Pred_Label"])

        import plotly.graph_objects as go
        trend_chart = go.Figure()
        trend_chart.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df["Simulated_Close"],
            mode="lines+markers",
            marker=dict(
                color=["green" if label == 1 else "red" for label in forecast_df["Pred_Label"]],
                size=8
            ),
            line=dict(color="gray"),
            name="Simulated Price"
        ))
        trend_chart.update_layout(
            title="📈 Simulated Price Trend with Direction Labels",
            xaxis_title="Date",
            yaxis_title="Simulated Close Price",
            showlegend=True
        )
        st.plotly_chart(trend_chart, use_container_width=True)

else:
    st.info("Please complete the model training before using the multi-day prediction function.")










