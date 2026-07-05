"""
Data loading.

- Prices: BTC-USD daily bars via yfinance.
- News: the paper uses the Alpaca News API. To keep this runnable without API
  keys, we SYNTHESIZE a plausible daily headline+body from the day's price
  move. Synthetic news is clearly flagged. To use real news instead, drop a
  CSV at data_news.csv with columns [date, text] and set
  CONFIG.use_synthetic_news = False.
"""

import os
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from config import CONFIG


@dataclass
class NewsItem:
    id: str
    date: str
    text: str
    is_injected: bool = False
    strategies: tuple = ()   # (bias, minor, rewrite) if injected


@dataclass
class MarketData:
    dates: List[str]
    close: np.ndarray          # adjusted close, aligned to dates
    ret: np.ndarray            # daily simple return, aligned to dates
    real_news: List[NewsItem]  # one real-ish news item per date


_UP_TEMPLATES = [
    "{asset} climbed {pct:.1f}% today as buyers stepped in and momentum "
    "strengthened, with traders pointing to firming demand across the market.",
    "{asset} rallied {pct:.1f}% amid renewed optimism, as inflows picked up "
    "and sentiment turned more constructive on the session.",
]
_DOWN_TEMPLATES = [
    "{asset} fell {pct:.1f}% today as selling pressure mounted and risk "
    "appetite cooled, with participants citing profit-taking after recent gains.",
    "{asset} slid {pct:.1f}% amid weaker sentiment, as outflows rose and "
    "traders grew more cautious about near-term direction.",
]
_FLAT_TEMPLATES = [
    "{asset} was little changed, moving {pct:.1f}% as the market consolidated "
    "and participants awaited fresh catalysts.",
]


def _synth_news(asset: str, date: str, r: float, rng: random.Random) -> str:
    pct = abs(r) * 100.0
    if r > 0.01:
        tmpl = rng.choice(_UP_TEMPLATES)
    elif r < -0.01:
        tmpl = rng.choice(_DOWN_TEMPLATES)
    else:
        tmpl = rng.choice(_FLAT_TEMPLATES)
    return tmpl.format(asset=asset, pct=pct)


def load_market_data(cfg=CONFIG) -> MarketData:
    rng = random.Random(cfg.seed)
    df = _fetch_prices(cfg)
    df = df.head(cfg.n_days + 1)  # +1 so every day has a next-day return

    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    close = df["close"].to_numpy(dtype=float)
    ret = np.zeros_like(close)
    ret[1:] = close[1:] / close[:-1] - 1.0

    # Optional external real news
    ext_news = None
    if not cfg.use_synthetic_news:
        ext_news = _load_external_news()

    real_news: List[NewsItem] = []
    for i, d in enumerate(dates):
        if ext_news is not None and d in ext_news:
            text = ext_news[d]
        else:
            text = _synth_news(cfg.asset, d, ret[i], rng)
        real_news.append(NewsItem(id=f"real-{i}", date=d, text=text))

    return MarketData(dates=dates, close=close, ret=ret, real_news=real_news)


def _fetch_prices(cfg=CONFIG) -> pd.DataFrame:
    try:
        import yfinance as yf
        raw = yf.download(
            cfg.asset, start=cfg.start_date, end=cfg.end_date,
            progress=False, auto_adjust=True,
        )
        if raw is not None and len(raw) > 0:
            # handle possible multiindex columns
            col = "Close" if "Close" in raw.columns else raw.columns[0]
            out = pd.DataFrame({"close": raw[col].to_numpy().ravel()},
                               index=raw.index)
            out.index = pd.to_datetime(out.index)
            return out
    except Exception as e:  # noqa: BLE001
        print(f"[warn] yfinance fetch failed ({e}); using synthetic prices.")
    return _synth_prices(cfg)


def _synth_prices(cfg=CONFIG) -> pd.DataFrame:
    """Deterministic geometric-random-walk fallback if yfinance is unavailable."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_days + 5
    dates = pd.bdate_range(cfg.start_date, periods=n)
    rets = rng.normal(0.001, 0.03, size=n)
    price = 20000.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"close": price}, index=dates)


def _load_external_news(path: str = "data_news.csv") -> Optional[dict]:
    if not os.path.exists(path):
        print(f"[warn] use_synthetic_news=False but {path} not found; "
              f"falling back to synthetic news.")
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return dict(zip(df["date"], df["text"]))
