from flask import Flask, render_template, request, jsonify
from agent.graph import finance_agent
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy_financial as npf
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    payload = request.json
    ticker = payload.get('ticker', 'D05.SI')
    news_text = payload.get('news_text', '')
    
    try:
        print(f"\n[API Request] Initiating Multi-Agent Workflow for {ticker}...")
        initial_state = {"ticker": ticker, "user_news_input": news_text} 
        
        # 核心改动：注入 config 参数，启用记忆线程 (thread_id)
        config = {"configurable": {"thread_id": ticker}}
        
        # 传入 config，让 Agent 记住这个 ticker 的历史状态
        result = finance_agent.invoke(initial_state, config=config)
        
        return jsonify({"status": "success", "report": result["final_report"]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/chart_data', methods=['POST'])
def get_chart_data():
    ticker = request.json.get('ticker', 'D05.SI')
    try:
        stock = yf.Ticker(ticker)
        # 稍微多抓一点数据，防止清洗后数据不够
        hist = stock.history(period="80d") 
        
        # 🛡️ 核心修复：清理脏数据，剔除收盘价为空的行
        hist = hist.dropna(subset=['Close'])
        
        # 截取最近的 60 个有效交易日用于展示
        hist = hist.tail(60)
        hist.index = hist.index.strftime('%Y-%m-%d') 
        
        close_prices = hist['Close'].values
        
        if len(close_prices) < 10:
             return jsonify({"status": "error", "message": "Not enough data points."})
        
        # 运行轻量级 ARIMA 用于前端可视化
        model = ARIMA(close_prices, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=7)
        
        # 生成未来 7 天的日期标签 (排除周末)
        last_date = pd.to_datetime(hist.index[-1])
        future_dates = []
        days_added = 0
        current_date = last_date
        while days_added < 7:
            current_date += pd.Timedelta(days=1)
            if current_date.weekday() < 5: # 0-4 代表周一到周五
                future_dates.append(current_date.strftime('%Y-%m-%d'))
                days_added += 1
        
        return jsonify({
            "status": "success",
            "history_dates": list(hist.index),
            "history_prices": list(close_prices),
            "future_dates": future_dates,
            "future_prices": list(forecast)
        })
    except Exception as e:
        print(f"Chart Data Error: {e}") # 在终端打印具体错误方便排查
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/tvm', methods=['POST'])
def calculate_tvm():
    data = request.json
    target = data.get('target', 'pv')
    
    try:
        rate = float(data.get('rate', 0))
        nper = float(data.get('nper', 0))
        pmt = float(data.get('pmt', 0))
        pv = float(data.get('pv', 0))
        fv = float(data.get('fv', 0))
        
        if target == 'fv': result = npf.fv(rate, nper, pmt, pv)
        elif target == 'pv': result = npf.pv(rate, nper, pmt, fv)
        elif target == 'pmt': result = npf.pmt(rate, nper, pv, fv)
        elif target == 'rate': result = npf.rate(nper, pmt, pv, fv)
        elif target == 'nper': result = npf.nper(rate, pmt, pv, fv)
        else: return jsonify({"status": "error", "message": "Unknown target variable."})
        
        return jsonify({"status": "success", "result": round(float(result), 4)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
