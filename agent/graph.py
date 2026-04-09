import os
from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver # 新增：引入持久化记忆模块
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from agent.tools import get_stock_data, predict_future_price, analyze_provided_news, tvm_calculator

# 1. 扩充状态字典 (加入量化分析师的中间报告)
class AgentState(TypedDict):
    ticker: str          
    user_news_input: str 
    raw_data: str        
    prediction_data: str 
    news_data: str       
    quant_report: str    # 新增：智能体A的输出
    final_report: str    # 智能体B的最终输出

llm = ChatGroq(model="llama-3.3-70b-versatile")

# --- 工具调用节点 ---
def gather_data_node(state: AgentState):
    print(f"🧠 [Agent 1 准备] 获取底层基本面数据...")
    return {"raw_data": get_stock_data.invoke({"ticker": state["ticker"]})}

def run_model_node(state: AgentState):
    print(f"🧠 [Agent 1 准备] 运行 ARIMA 模型...")
    return {"prediction_data": predict_future_price.invoke({"ticker": state["ticker"]})}

def gather_news_node(state: AgentState):
    print(f"🧠 [Agent 2 准备] 提取非结构化情报...")
    return {"news_data": analyze_provided_news.invoke({"news_text": state["user_news_input"]})}

# --- 智能体 A：量化分析师节点 ---
def quant_analyst_node(state: AgentState):
    print(f"👨‍💻 [Multi-Agent: Quant] 量化分析师正在起草数学模型报告...")
    
    prompt = f"""
    You are a Quantitative Analyst. Based ONLY on the mathematical data below, draft a strict quantitative trend report for {state['ticker']}.
    Ignore any external news or human sentiment. Focus strictly on numbers.
    
    [Market Fundamentals]: {state['raw_data']}
    [ARIMA Forecast]: {state['prediction_data']}
    """
    response = llm.invoke([
        SystemMessage(content="You are a strict, math-driven Quant Analyst."),
        HumanMessage(content=prompt)
    ])
    return {"quant_report": response.content}

# --- 智能体 B：首席风控官节点 (核心制衡逻辑) ---
def risk_controller_node(state: AgentState):
    print(f"🕵️‍♂️ [Multi-Agent: Risk Controller] 风控官正在审查量化报告并结合情绪进行最终裁决...")
    
    demo_tvm = tvm_calculator.invoke({"target": "pv", "rate": 0.05/365, "nper": 7, "pmt": 0, "fv": 100})
    
    prompt = f"""
    You are the Chief Risk Officer (CRO). You must review the Quantitative Analyst's mathematical report and cross-validate it against human-provided intelligence.
    
    [Quant Analyst's Mathematical Report]: 
    {state['quant_report']}
    
    [Human-in-the-Loop Intelligence]: 
    {state['news_data']}
    
    Output Requirements for Final Executive Report:
    1. Quant Summary: Briefly summarize the Quant Analyst's mathematical findings.
    2. Multi-Agent Cross-Validation (Crucial): Does the human intelligence contradict the math? (e.g., ARIMA is bullish, but news indicates a lawsuit). If there is a contradiction, you MUST override the Quant Analyst and issue a downgrade.
    3. TVM Context: Mention theoretical PV ({demo_tvm}) to illustrate time-value risk over a 7-day hold.
    4. Final Verdict: Issue the final Buy/Hold/Sell decision on behalf of the risk committee.
    """
    response = llm.invoke([
        SystemMessage(content="You are a conservative, risk-averse Chief Risk Officer."),
        HumanMessage(content=prompt)
    ])
    return {"final_report": response.content}

# --- 构建图与记忆机制 ---
workflow = StateGraph(AgentState)

workflow.add_node("gather_data", gather_data_node)
workflow.add_node("run_model", run_model_node)
workflow.add_node("gather_news", gather_news_node)
workflow.add_node("quant_analyst", quant_analyst_node)       # 加入智能体 A
workflow.add_node("risk_controller", risk_controller_node)   # 加入智能体 B

# 定义执行流
workflow.add_edge(START, "gather_data")
workflow.add_edge("gather_data", "run_model")
workflow.add_edge("run_model", "gather_news")
workflow.add_edge("gather_news", "quant_analyst")
workflow.add_edge("quant_analyst", "risk_controller") # Quant 写完报告后，直接交给 CRO 审查
workflow.add_edge("risk_controller", END)

# 初始化长期记忆数据库 (保存在内存中)
memory = MemorySaver()

# 编译时注入记忆机制
finance_agent = workflow.compile(checkpointer=memory)