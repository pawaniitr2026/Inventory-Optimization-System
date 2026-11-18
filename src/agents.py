"""LangGraph multi-agent workflow with optional LangChain/OpenRouter narrative generation."""
import os
import numpy as np
from typing import TypedDict, Any

class WorkflowState(TypedDict, total=False):
    daily: Any
    forecast: Any
    metrics: Any
    lead_time: int
    service_level: float
    shock_mode: str
    forecast_agent: str
    uncertainty: dict
    plan: dict
    trace: list

def _llm_summary(prompt, fallback):
    """Optional LLM narrative; numeric planning remains deterministic and auditable."""
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return fallback
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        kwargs = {"model": os.getenv("LLM_MODEL", "openai/gpt-4o-mini"), "temperature": 0}
        if os.getenv("OPENROUTER_API_KEY"):
            kwargs["base_url"] = "https://openrouter.ai/api/v1"
            kwargs["api_key"] = os.getenv("OPENROUTER_API_KEY")
        else:
            kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(**kwargs)
        response = llm.invoke([HumanMessage(content=prompt)])
        return str(response.content)
    except Exception as exc:
        return fallback + f" (LLM narrative unavailable; deterministic fallback used: {type(exc).__name__})"

def forecast_agent(state: WorkflowState):
    forecast = state["forecast"]
    metrics = state["metrics"]
    model = str(forecast["selected_model"].iloc[0])
    best = metrics.iloc[0]
    fallback = (
        f"The forecast uses **{model}**, selected from a chronological holdout comparison. "
        f"Best baseline MAE is {best['MAE']:.2f}; forecast values are non-negative. "
        "Treat this as a baseline forecast, not a guarantee during abrupt regime changes."
    )
    prompt = (
        "You are a demand forecasting analyst. Explain briefly and cautiously. "
        f"Selected model: {model}. Holdout MAE: {best['MAE']:.2f}. "
        f"Future horizon: {len(forecast)} days. Avoid claiming the model is shock-proof."
    )
    summary = _llm_summary(prompt, fallback)
    return {"forecast_agent": summary, "trace": state.get("trace", []) + ["Forecast Agent"]}

def uncertainty_agent(state: WorkflowState):
    daily = state["daily"]
    y = daily["sales"].astype(float).to_numpy()
    recent_n = min(28, len(y))
    recent = y[-recent_n:]
    baseline = y[-min(90, len(y)):-recent_n] if len(y) > recent_n else y
    baseline_mean = float(np.mean(baseline)) if len(baseline) else float(np.mean(y))
    recent_mean = float(np.mean(recent))
    recent_std = float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0
    ratio = recent_mean / max(baseline_mean, 1.0)
    cv = recent_std / max(recent_mean, 1.0)
    mode = state.get("shock_mode", "Auto-detect")
    if mode == "No shock":
        shock = "No shock forced"
    elif mode == "Promotion spike":
        shock = "Promotion spike scenario"
    elif mode == "Supply disruption / demand surge":
        shock = "Demand surge scenario"
    elif ratio >= 1.25:
        shock = "Potential upward demand shock"
    elif ratio <= 0.75:
        shock = "Potential downward demand shock"
    elif cv >= 0.35:
        shock = "High demand volatility"
    else:
        shock = "No strong shock signal"
    summary = (
        f"Recent {recent_n}-day mean is {recent_mean:.2f}; comparison mean is {baseline_mean:.2f}; "
        f"recent coefficient of variation is {cv:.2f}. Status: {shock}. "
        "This is a simple statistical alert, not causal attribution."
    )
    return {
        "uncertainty": {
            "shock_status": shock,
            "recent_mean": round(recent_mean, 2),
            "comparison_mean": round(baseline_mean, 2),
            "recent_cv": round(cv, 3),
            "mean_ratio": round(ratio, 3),
            "summary": summary,
        },
        "trace": state.get("trace", []) + ["Uncertainty Agent"],
    }

def inventory_planner_agent(state: WorkflowState):
    import statistics
    forecast = state["forecast"]["forecast"].astype(float).to_numpy()
    lead = max(1, int(state["lead_time"]))
    service = float(state["service_level"])
    # Normal approximation for a one-sided service level.
    z_table = [(0.80, .842), (0.85, 1.036), (0.90, 1.282), (0.95, 1.645), (0.97, 1.881), (0.98, 2.054), (0.99, 2.326)]
    z = min(z_table, key=lambda p: abs(p[0] - service))[1]
    daily_sigma = max(float(state["uncertainty"]["recent_cv"]) * max(float(state["uncertainty"]["recent_mean"]), 1.0), 1.0)
    mean_demand = float(np.mean(forecast[:min(lead, len(forecast))])) if len(forecast) else float(state["uncertainty"]["recent_mean"])
    lead_demand = mean_demand * lead
    safety_stock = z * daily_sigma * np.sqrt(lead)
    if "shock" in state["uncertainty"]["shock_status"].lower() or "surge" in state["uncertainty"]["shock_status"].lower() or "spike" in state["uncertainty"]["shock_status"].lower():
        safety_stock *= 1.25
    reorder_point = lead_demand + safety_stock
    avg_daily = float(np.mean(forecast)) if len(forecast) else mean_demand
    order_up_to = avg_daily * (lead + 7) + safety_stock
    fallback = (
        f"Use a provisional reorder point of {reorder_point:.1f} units and safety stock of {safety_stock:.1f} units. "
        "Review daily while a shock alert is active, confirm supplier lead time and on-hand inventory, and avoid "
        "automatic purchase orders until the recommendation is validated."
    )
    prompt = (
        "You are an inventory planning analyst. Give a concise, cautious recommendation based on these deterministic calculations. "
        f"Lead time={lead} days; service level={service:.0%}; reorder point={reorder_point:.1f}; "
        f"safety stock={safety_stock:.1f}; order-up-to={order_up_to:.1f}; "
        f"shock={state['uncertainty']['shock_status']}. Do not invent costs or stock levels."
    )
    recommendation = _llm_summary(prompt, fallback)
    return {
        "plan": {
            "lead_time_days": lead,
            "target_service_level": service,
            "z_score_approx": z,
            "estimated_mean_daily_demand": round(mean_demand, 2),
            "lead_time_demand": round(lead_demand, 2),
            "safety_stock": round(float(safety_stock), 2),
            "reorder_point": round(float(reorder_point), 2),
            "order_up_to_level": round(float(order_up_to), 2),
            "recommendation": recommendation,
            "caveat": "Does not account for current inventory, open purchase orders, pack size, shelf life, costs, or supplier constraints.",
        },
        "trace": state.get("trace", []) + ["Inventory Planner Agent"],
    }

def run_agent_workflow(daily, forecast, metrics, lead_time, service_level, shock_mode):
    try:
        from langgraph.graph import StateGraph, START, END
        graph = StateGraph(WorkflowState)
        graph.add_node("forecast_agent", forecast_agent)
        graph.add_node("uncertainty_agent", uncertainty_agent)
        graph.add_node("inventory_planner", inventory_planner_agent)
        graph.add_edge(START, "forecast_agent")
        graph.add_edge("forecast_agent", "uncertainty_agent")
        graph.add_edge("uncertainty_agent", "inventory_planner")
        graph.add_edge("inventory_planner", END)
        app = graph.compile()
        initial = {
            "daily": daily, "forecast": forecast, "metrics": metrics,
            "lead_time": lead_time, "service_level": service_level,
            "shock_mode": shock_mode, "trace": [],
        }
        return app.invoke(initial)
    except ImportError:
        # Allows the app to run even before LangGraph is installed.
        state = {
            "daily": daily, "forecast": forecast, "metrics": metrics,
            "lead_time": lead_time, "service_level": service_level,
            "shock_mode": shock_mode, "trace": [],
        }
        for fn in (forecast_agent, uncertainty_agent, inventory_planner_agent):
            state.update(fn(state))
        return state
