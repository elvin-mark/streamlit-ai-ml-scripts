import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="AI Stock Forecasting", layout="wide", initial_sidebar_state="expanded"
)

st.title("🚀 Advanced Time Series Forecasting with Deep Learning")
st.markdown("*Predict stock prices using Recurrent Neural Networks (RNN/LSTM)*")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Configuration")

    # Stock selection
    ticker = st.text_input(
        "📊 Stock Ticker",
        value="AAPL",
        help="Enter stock symbol (e.g., AAPL, GOOGL, MSFT)",
    )

    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime.now() - timedelta(days=365 * 2),
            max_value=datetime.now() - timedelta(days=30),
        )
    with col2:
        end_date = st.date_input(
            "End Date", datetime.now(), min_value=start_date + timedelta(days=30)
        )

    st.divider()

    # Model configuration
    st.subheader("🧠 Model Settings")
    model_type = st.selectbox(
        "Model Type", ["LSTM", "RNN", "GRU"], help="Choose neural network architecture"
    )
    hidden_size = st.slider("Hidden Units", 16, 256, 64, step=16)
    num_layers = st.slider("Number of Layers", 1, 4, 2)
    window_size = st.slider(
        "Sequence Length", 10, 100, 30, help="Number of past days to use for prediction"
    )

    st.divider()

    # Training configuration
    st.subheader("🏋️ Training Settings")
    epochs = st.slider("Training Epochs", 50, 500, 150, step=25)
    learning_rate = st.select_slider(
        "Learning Rate", options=[0.001, 0.005, 0.01, 0.05], value=0.01
    )
    train_split = st.slider("Training Data %", 70, 90, 80) / 100

    st.divider()

    # Forecasting
    st.subheader("🔮 Forecasting")
    forecast_days = st.slider("Days to Forecast", 1, 60, 14)


# Data loading function
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_stock_data(ticker, start_date, end_date):
    """Load stock data with error handling"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None, "No data found for this ticker"

        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        return data[["Close"]].dropna(), None
    except Exception as e:
        return None, f"Error loading data: {str(e)}"


# Model class
class AdvancedRNNModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        rnn_type="LSTM",
        dropout=0.2,
    ):
        super().__init__()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        else:  # RNN
            self.rnn = nn.RNN(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])  # Take last output
        return self.fc(out)


def create_sequences(data, window_size):
    """Create sequences for time series prediction"""
    sequences, targets = [], []
    for i in range(len(data) - window_size):
        sequences.append(data[i : i + window_size])
        targets.append(data[i + window_size])
    return np.array(sequences), np.array(targets)


def train_model_with_progress(
    model, X_train, y_train, X_val, y_val, epochs, learning_rate
):
    """Train model with progress tracking"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    train_losses = []
    val_losses = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    best_val_loss = float("inf")
    patience_counter = 0
    early_stopping_patience = 20

    for epoch in range(epochs):
        # Training
        model.train()
        train_output = model(X_train)
        train_loss = criterion(train_output, y_train)

        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)

        scheduler.step(val_loss)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            st.info(f"Early stopping at epoch {epoch+1}")
            break

        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
        )

    progress_bar.empty()
    status_text.empty()

    return model, train_losses, val_losses


# Main app logic
if st.button("🚀 Run Forecasting", type="primary"):
    # Load data
    with st.spinner("Loading stock data..."):
        data, error = load_stock_data(ticker, start_date, end_date)

    if error:
        st.error(error)
        st.stop()

    if len(data) < window_size + 30:
        st.error(
            f"Insufficient data. Need at least {window_size + 30} days, got {len(data)} days."
        )
        st.stop()

    # Display basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Points", len(data))
    with col2:
        current_price = float(data["Close"].iloc[-1])
        st.metric("Current Price", f"${current_price:.2f}")
    with col3:
        price_change = float(data["Close"].iloc[-1] - data["Close"].iloc[-2])
        st.metric("Daily Change", f"${price_change:.2f}", f"{price_change:.2f}")
    with col4:
        volatility = float(data["Close"].pct_change().std() * np.sqrt(252) * 100)
        st.metric("Volatility (Annual)", f"{volatility:.1f}%")

    # Show raw data chart
    st.subheader(f"📈 {ticker} Stock Price History")
    fig_raw = go.Figure()
    fig_raw.add_trace(
        go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close Price")
    )
    fig_raw.update_layout(
        title=f"{ticker} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=400,
    )
    st.plotly_chart(fig_raw, use_container_width=True)

    # Prepare data
    with st.spinner("Preparing data and training model..."):
        # Normalize data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values)

        # Create sequences
        X, y = create_sequences(scaled_data, window_size)

        # Train/validation split
        split_idx = int(len(X) * train_split)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)

        # Create and train model
        model = AdvancedRNNModel(1, hidden_size, 1, num_layers, model_type)

        st.info("🏋️ Training model... This may take a moment.")
        model, train_losses, val_losses = train_model_with_progress(
            model,
            X_train_tensor,
            y_train_tensor,
            X_val_tensor,
            y_val_tensor,
            epochs,
            learning_rate,
        )

    # Training results
    st.success("✅ Model training completed!")

    # Show training curves
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=train_losses, mode="lines", name="Training Loss"))
    fig_loss.add_trace(go.Scatter(y=val_losses, mode="lines", name="Validation Loss"))
    fig_loss.update_layout(
        title="📉 Training Progress",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=300,
    )
    st.plotly_chart(fig_loss, use_container_width=True)

    # Make predictions
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_tensor).numpy()
        val_pred = model(X_val_tensor).numpy()

    # Inverse transform predictions
    train_pred_prices = scaler.inverse_transform(train_pred)
    val_pred_prices = scaler.inverse_transform(val_pred)

    # Calculate metrics
    train_actual = scaler.inverse_transform(y_train)
    val_actual = scaler.inverse_transform(y_val)

    train_mse = mean_squared_error(train_actual, train_pred_prices)
    val_mse = mean_squared_error(val_actual, val_pred_prices)
    train_mae = mean_absolute_error(train_actual, train_pred_prices)
    val_mae = mean_absolute_error(val_actual, val_pred_prices)

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Train RMSE", f"{np.sqrt(train_mse):.2f}")
    with col2:
        st.metric("Val RMSE", f"{np.sqrt(val_mse):.2f}")
    with col3:
        st.metric("Train MAE", f"{train_mae:.2f}")
    with col4:
        st.metric("Val MAE", f"{val_mae:.2f}")

    # Generate forecasts
    with st.spinner("Generating forecasts..."):
        last_sequence = scaled_data[-window_size:]
        forecasts = []
        input_seq = torch.FloatTensor(last_sequence).unsqueeze(0)

        model.eval()
        for _ in range(forecast_days):
            with torch.no_grad():
                pred = model(input_seq).numpy()
            forecasts.append(pred[0])

            # Update input sequence
            new_input = np.append(input_seq.squeeze()[1:], pred[0])
            input_seq = torch.FloatTensor(new_input).unsqueeze(0).unsqueeze(-1)

        forecast_prices = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))

    # Create comprehensive visualization
    st.subheader("🔮 Forecasting Results")

    # Prepare data for plotting
    train_dates = data.index[window_size : window_size + len(train_pred)]
    val_dates = data.index[
        window_size + len(train_pred) : window_size + len(train_pred) + len(val_pred)
    ]
    forecast_dates = pd.date_range(
        data.index[-1] + timedelta(days=1), periods=forecast_days
    )

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Price Prediction & Forecast", "Prediction Error"),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
    )

    # Main plot
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Actual Price",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=train_dates,
            y=train_pred_prices.flatten(),
            mode="lines",
            name="Training Predictions",
            line=dict(color="green"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=val_dates,
            y=val_pred_prices.flatten(),
            mode="lines",
            name="Validation Predictions",
            line=dict(color="orange"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_prices.flatten(),
            mode="lines+markers",
            name="Forecast",
            line=dict(color="red", dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Error plot
    train_errors = train_actual.flatten() - train_pred_prices.flatten()
    val_errors = val_actual.flatten() - val_pred_prices.flatten()

    fig.add_trace(
        go.Scatter(
            x=train_dates,
            y=train_errors,
            mode="lines",
            name="Training Error",
            line=dict(color="lightgreen"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=val_dates,
            y=val_errors,
            mode="lines",
            name="Validation Error",
            line=dict(color="lightcoral"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=800, title_text=f"{ticker} Stock Price Prediction & Forecast"
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Error ($)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    st.subheader("📊 Detailed Forecast")
    forecast_df = pd.DataFrame(
        {
            "Date": forecast_dates,
            "Predicted Price": forecast_prices.flatten(),
            "Days Ahead": range(1, forecast_days + 1),
        }
    )

    # Add confidence intervals (simple approach)
    recent_volatility = data["Close"].pct_change().tail(30).std()
    forecast_df["Lower Bound"] = forecast_df["Predicted Price"] * (
        1 - recent_volatility * np.sqrt(forecast_df["Days Ahead"])
    )
    forecast_df["Upper Bound"] = forecast_df["Predicted Price"] * (
        1 + recent_volatility * np.sqrt(forecast_df["Days Ahead"])
    )

    st.dataframe(forecast_df.set_index("Date").round(2), use_container_width=True)

    # Summary insights
    st.subheader("📝 Key Insights")
    current_price = float(data["Close"].iloc[-1])
    avg_forecast = float(forecast_prices.mean())
    price_change_pct = ((avg_forecast - current_price) / current_price) * 100

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Average Forecast Price**: ${avg_forecast:.2f}")
        st.info(f"**Expected Change**: {price_change_pct:+.1f}%")
    with col2:
        trend = "📈 Upward" if price_change_pct > 0 else "📉 Downward"
        st.info(f"**Trend Direction**: {trend}")
        st.info(f"**Model Validation RMSE**: ${np.sqrt(val_mse):.2f}")

else:
    # Show demo information
    st.info(
        "👆 Configure your settings in the sidebar and click 'Run Forecasting' to start!"
    )

    st.markdown(
        """
    ## 🎯 Features
    
    - **Multiple Architectures**: Choose between LSTM, RNN, and GRU models
    - **Advanced Training**: Early stopping, learning rate scheduling, gradient clipping
    - **Comprehensive Metrics**: RMSE, MAE for both training and validation
    - **Interactive Visualizations**: Training curves, predictions, and forecasts
    - **Confidence Intervals**: Uncertainty estimation for forecasts
    - **Real-time Data**: Fetches latest stock data from Yahoo Finance
    
    ## 📊 How It Works
    
    1. **Data Collection**: Fetches historical stock data for your chosen ticker
    2. **Preprocessing**: Normalizes data and creates sequential windows
    3. **Model Training**: Trains deep learning model with your chosen architecture
    4. **Validation**: Evaluates model performance on unseen data
    5. **Forecasting**: Generates future price predictions
    
    ## ⚠️ Disclaimer
    
    This tool is for educational and research purposes only. Stock predictions are inherently uncertain and should not be used as the sole basis for investment decisions.
    """
    )
