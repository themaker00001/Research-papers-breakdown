"""
The LLM-based financial agent (Perception / Profile / Memory / Action).

Given the day's market context and a set of retrieved news items, the agent
returns one of {Buy, Sell, Hold}. An optional bias system-prompt (Phi_Bias)
can be injected into its context, and an optional time-series-grounding block
(the defense) can be prepended to its observation.
"""

from typing import List, Optional, Dict

import numpy as np

import prompts
from config import CONFIG
from data import NewsItem
from ollama_client import OllamaClient


VALID_ACTIONS = {"Buy", "Sell", "Hold"}


class FinAgent:
    def __init__(self, client: OllamaClient, cfg=CONFIG):
        self.client = client
        self.cfg = cfg
        self.memory: List[str] = []  # short summaries of past decisions

    def decide(self, asset: str, date: str, market_context: str,
               news: List[NewsItem], bias_prompt: Optional[str] = None,
               temporal_block: str = "") -> Dict:
        system = prompts.AGENT_SYSTEM
        if bias_prompt:
            system = system + "\n" + bias_prompt

        news_block = "\n".join(
            f"- {n.text}" for n in news) or "- (no news retrieved)"
        mem_block = ""
        if self.memory:
            mem_block = "\nYour recent notes:\n" + "\n".join(
                f"- {m}" for m in self.memory[-3:])

        user = prompts.AGENT_USER_TEMPLATE.format(
            asset=asset, date=date,
            market_context=market_context + mem_block,
            temporal_block=("\n" + temporal_block + "\n") if temporal_block else "\n",
            news_block=news_block,
        )
        out = self.client.chat_json(system, user, temperature=0.4)
        action = "Hold"
        conf, reason = 0.5, ""
        if out:
            a = str(out.get("action", "Hold")).capitalize()
            action = a if a in VALID_ACTIONS else "Hold"
            try:
                conf = float(out.get("confidence", 0.5))
            except Exception:  # noqa: BLE001
                conf = 0.5
            reason = str(out.get("reason", ""))[:200]
        return {"action": action, "confidence": conf, "reason": reason}

    def remember(self, date: str, decision: Dict):
        self.memory.append(
            f"{date}: {decision['action']} ({decision['reason']})")


# --------------------------------------------------------------------------
# Time-series-informed grounding (the defense). Builds a compact textual
# summary of recent market dynamics so the agent can sanity-check the news.
# --------------------------------------------------------------------------

def temporal_grounding(close: np.ndarray, t: int, window: int = 30) -> str:
    """Construct temporal evidence using ONLY data available before day t."""
    if t < 2:
        return ""
    hist = close[max(0, t - window):t]
    if len(hist) < 3:
        return ""
    rets = hist[1:] / hist[:-1] - 1.0
    price = hist[-1]
    last_ret = rets[-1]
    vol5 = float(np.std(rets[-5:])) if len(rets) >= 5 else float(np.std(rets))
    # 20-day max drawdown
    win = hist[-20:] if len(hist) >= 20 else hist
    peak = np.maximum.accumulate(win)
    drawdown = float((win[-1] - peak.max()) / peak.max())
    # trend divergence: short vs long moving-average gap
    short = float(np.mean(hist[-5:]))
    long = float(np.mean(hist[-min(len(hist), 20):]))
    trend_div = (short - long) / long if long else 0.0
    # anomaly: z-score of the latest return
    anomaly = float((last_ret - np.mean(rets)) / (np.std(rets) + 1e-9))

    return (
        "Market evidence (from price history only):\n"
        f"  last close={price:.1f}, last return={last_ret*100:+.2f}%, "
        f"5d volatility={vol5*100:.2f}%, 20d drawdown={drawdown*100:.2f}%, "
        f"trend divergence(short-long)={trend_div*100:+.2f}%, "
        f"return anomaly z={anomaly:+.2f}.\n"
        "  If a news item conflicts with this recent price behavior, treat it "
        "with caution."
    )
