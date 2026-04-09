import yfinance as yf
import pandas as pd
import numpy_financial as npf
from langchain_core.tools import tool
from statsmodels.tsa.arima.model import ARIMA
import warnings
import requests  # 新增
import os        # 新增


warnings.filterwarnings("ignore")

@tool
def get_stock_data(ticker: str) -> str:
    """
    Retrieves historical market data and computes key risk metrics.
    Parameter ticker: The stock symbol (e.g., 'D05.SI' for DBS).
    """
    print(f"🔧 [Tool Execution] Ingesting market data for {ticker} via yfinance...")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        
        # Clean dirty data
        hist = hist.dropna(subset=['Close']) 
        if hist.empty or len(hist) < 25:
            return f"❌ Error: Insufficient valid market data for {ticker}."

        current_price = hist['Close'].iloc[-1]
        high_52w = hist['High'].max()
        low_52w = hist['Low'].min()
        
        daily_returns = hist['Close'].pct_change().dropna()
        annual_volatility = daily_returns.std() * (252 ** 0.5) * 100 
        
        price_20d_ago = hist['Close'].iloc[-20]
        momentum_20d = ((current_price - price_20d_ago) / price_20d_ago) * 100

        report = (
            f"[Market Data Summary for {ticker} - Trailing 12 Months]\n"
            f"- Last Close Price: {current_price:.2f} SGD\n"
            f"- 52-Week High/Low: {high_52w:.2f} SGD / {low_52w:.2f} SGD\n"
            f"- Annualized Volatility: {annual_volatility:.2f}%\n"
            f"- 20-Day Price Momentum: {momentum_20d:.2f}%\n"
        )
        return report
    except Exception as e:
        return f"❌ Data Ingestion Error: {str(e)}"

@tool
def predict_future_price(ticker: str) -> str:
    """
    Executes an ARIMA time-series model to forecast the asset's price trajectory.
    """
    print(f"📈 [Tool Execution] Initializing ARIMA forecasting engine for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")
        
        hist = hist.dropna(subset=['Close'])
        if hist.empty or len(hist) < 50:
            return f"❌ Error: Insufficient historical data for model training."

        close_prices = hist['Close'].values
        model = ARIMA(close_prices, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=7)
        
        current_price = close_prices[-1]
        future_price = forecast[-1]
        trend = "Bullish" if future_price > current_price else "Bearish"
        expected_variance = ((future_price - current_price) / current_price) * 100
        
        report = (
            f"[ARIMA Quantitative Forecast ({ticker})]\n"
            f"- Baseline Price: {current_price:.2f} SGD\n"
            f"- T+7 Forecasted Price: {future_price:.2f} SGD\n"
            f"- Algorithmic Trend: {trend} (Expected Variance: {expected_variance:.2f}%)\n"
            f"(*Disclaimer: ARIMA(5,1,0) relies strictly on historical stationarity and does not account for tail-risk events*)"
        )
        return report
    except Exception as e:
        return f"❌ ARIMA Execution Error: {str(e)}"



from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

# Initialize a lightweight model specifically for sentiment classification
sentiment_llm = ChatGroq(model="llama-3.1-8b-instant") 

@tool
def analyze_provided_news(news_text: str) -> str:
    """
    Performs semantic and sentiment analysis on user-provided market intelligence using a Large Language Model.
    """
    print("📰 [Tool Execution] Executing LLM-based sentiment analysis...")
    
    if not news_text or len(news_text.strip()) == 0:
        return "⚠️ No external market intelligence provided. Default sentiment: Neutral ⚪"

    prompt = f"""
    You are a professional financial sentiment analysis engine. Please review the following market intelligence and output your analysis strictly in JSON or structured text format.
    
    Intelligence Content: "{news_text}"
    
    Please analyze the following:
    1. Overall Sentiment: Must select exactly one from [Bullish 🟢, Bearish 🔴, Neutral ⚪].
    2. Key Catalyst: Summarize the core event driving this sentiment in a single sentence.
    3. Confidence Score: A scale from 0 to 100 (100 indicates a highly certain major catalyst, 50 indicates neutral or ambiguous data).
    """
    
    try:
        response = sentiment_llm.invoke([
            SystemMessage(content="You are a precise financial sentiment classifier."),
            HumanMessage(content=prompt)
        ])
        return f"[Sentiment Analysis Report]\n{response.content}"
    except Exception as e:
         return f"❌ Sentiment Analysis Engine Error: {str(e)}"

@tool
def tvm_calculator(target: str, rate: float = 0.0, nper: int = 0, pmt: float = 0.0, pv: float = 0.0, fv: float = 0.0) -> str:
    """Professional Time Value of Money (TVM) Computation Engine."""
    print(f"🧮 [Tool Execution] Engaging TVM Calculator to solve for: {target.upper()}")
    try:
        if target.lower() == 'fv': result = npf.fv(rate, nper, pmt, pv)
        elif target.lower() == 'pv': result = npf.pv(rate, nper, pmt, fv)
        elif target.lower() == 'pmt': result = npf.pmt(rate, nper, pv, fv)
        elif target.lower() == 'rate': result = npf.rate(nper, pmt, pv, fv)
        elif target.lower() == 'nper': result = npf.nper(rate, pmt, pv, fv)
        else: return f"❌ Invalid target variable '{target}'."
        return f"[TVM Engine Output] The precise calculation for {target.upper()} yields: {result:.4f}"
    except Exception as e:
        return f"❌ TVM Computation Error: {str(e)}"