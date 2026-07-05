"""
AutoRedTrader closed-loop runner.

For each trading day t:
  1. clean decision   -> agent retrieves from REAL news, decides
  2. MisGen           -> generate + filter misinformation (feedback-aware)
  3. injected decision-> agent retrieves from REAL + MISINFO, decides
  4. record MER / flip; simulate the trade for CR using the next-day return
  5. update HistoryEffect from which strategies caused flips (closed loop)

Runs the attack twice: once against the plain agent, once against the
temporally-grounded agent (the defense), and prints a comparison table
in the style of the paper's Table 1.

Usage:
    python run.py            # full run (attack + defense)
    python run.py --attack   # attack only
    python run.py --defense  # defense only
"""

import argparse
import copy
from typing import Optional

import numpy as np

from config import CONFIG
from data import load_market_data, MarketData
from ollama_client import OllamaClient, check_ollama
from agent import FinAgent, temporal_grounding
from misgen import MisGen
from metrics import Metrics
from retrieval import retrieve


def _market_context(md: MarketData, t: int) -> str:
    lo = max(0, t - 5)
    recent = md.close[lo:t + 1]
    rets = md.ret[lo:t + 1]
    lines = [f"  {md.dates[i]}: close={md.close[i]:.1f} "
             f"({md.ret[i]*100:+.2f}%)" for i in range(lo, t + 1)]
    return ("Last sessions:\n" + "\n".join(lines) +
            f"\n  5-day avg return={np.mean(rets)*100:+.2f}%")


def run_experiment(md: MarketData, client: OllamaClient,
                   use_defense: bool, cfg=CONFIG) -> Metrics:
    label = "DEFENSE (temporal grounding)" if use_defense else "ATTACK (plain agent)"
    print(f"\n{'='*70}\n{label}\n{'='*70}")

    agent = FinAgent(client, cfg)
    misgen = MisGen(client, cfg)
    metrics = Metrics()
    running_cr = 0.0  # feedback signal for bias nudging

    n = min(cfg.n_days, len(md.dates) - 1)
    for t in range(n):
        date = md.dates[t]
        ctx = _market_context(md, t)
        real_pool = md.real_news[:t + 1]  # news available up to today
        query = f"{cfg.asset} outlook {date} price momentum risk"
        tblock = temporal_grounding(md.close, t) if use_defense else ""

        if cfg.verbose:
            print(f"\n[day {t+1}/{n}] {date}")

        # 1) clean decision
        clean_ret = retrieve(query, real_pool, cfg.k_retrieve)
        clean_dec = agent.decide(cfg.asset, date, ctx, clean_ret,
                                 temporal_block=tblock)

        # 2) MisGen (feedback-aware)
        injected = misgen.generate(
            md.real_news[t], running_cr, t, cfg.n_misinfo_per_day)
        inj_items = [it for it, _ in injected]
        bias_prompt: Optional[str] = None
        for _, bp in injected:
            if bp:
                bias_prompt = bp  # use the last non-null bias context
                break

        # 3) injected decision (retrieve from real + misinfo)
        inj_pool = real_pool + inj_items
        inj_ret = retrieve(query, inj_pool, cfg.k_retrieve)
        inj_dec = agent.decide(cfg.asset, date, ctx, inj_ret,
                               bias_prompt=bias_prompt, temporal_block=tblock)

        # 4) record + simulate trade with next-day return
        next_ret = md.ret[t + 1]
        metrics.record_day(inj_ret, clean_dec["action"],
                           inj_dec["action"], next_ret)

        # 5) closed-loop feedback: attribute the flip to retrieved injected items
        flipped = inj_dec["action"] != clean_dec["action"]
        for it in inj_ret:
            if getattr(it, "is_injected", False):
                misgen.update_history(it, flipped)

        agent.remember(date, inj_dec)
        running_cr = metrics.CR_inj / 100.0

        if cfg.verbose:
            exp = metrics.exposures[-1] * 100
            print(f"  clean={clean_dec['action']}  inj={inj_dec['action']}  "
                  f"flip={'Y' if flipped else 'N'}  "
                  f"exposure={exp:.0f}%  next_ret={next_ret*100:+.2f}%")

    s = metrics.summary()
    print(f"\n--- {label} results ---")
    print(f"  MER={s['MER']:.2f}%  ASR={s['ASR']:.2f}%  "
          f"CR_inj={s['CR_inj']:+.2f}%  CR_clean={s['CR_clean']:+.2f}%")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", action="store_true", help="attack only")
    ap.add_argument("--defense", action="store_true", help="defense only")
    ap.add_argument("--days", type=int, default=None, help="override n_days")
    ap.add_argument("--quiet", action="store_true", help="less logging")
    args = ap.parse_args()

    cfg = copy.copy(CONFIG)
    if args.days:
        cfg.n_days = args.days
    if args.quiet:
        cfg.verbose = False

    print("AutoRedTrader (local qwen3 reproduction)")
    print(f"model={cfg.model}  asset={cfg.asset}  days={cfg.n_days}")
    if not check_ollama(cfg):
        print("[fatal] Ollama not reachable or model missing. "
              "Start Ollama and `ollama pull qwen3:14b`.")
        return

    client = OllamaClient(cfg)
    md = load_market_data(cfg)
    print(f"loaded {len(md.dates)} sessions, "
          f"{len(md.real_news)} real news items.")

    do_attack = args.attack or not args.defense
    do_defense = args.defense or not args.attack

    results = {}
    if do_attack:
        results["attack"] = run_experiment(md, client, use_defense=False, cfg=cfg)
    if do_defense:
        results["defense"] = run_experiment(md, client, use_defense=True, cfg=cfg)

    # Paper-style summary table
    print(f"\n{'='*70}\nSUMMARY (paper Table 1 style)\n{'='*70}")
    print(f"{'Setting':<28}{'MER':>8}{'ASR':>8}{'CR':>10}")
    print("-" * 54)
    if "attack" in results:
        m = results["attack"]
        print(f"{'Base (clean)':<28}{'-':>8}{'-':>8}"
              f"{m.CR_clean:>+9.2f}%")
        print(f"{'AutoRedTrader':<28}{m.MER:>7.2f}%{m.ASR:>7.2f}%"
              f"{m.CR_inj:>+9.2f}%")
    if "defense" in results:
        m = results["defense"]
        print(f"{'AutoRedTrader+Time':<28}{m.MER:>7.2f}%{m.ASR:>7.2f}%"
              f"{m.CR_inj:>+9.2f}%")
    print("\nHigher MER/ASR = stronger attack. The +Time (defense) row should "
          "show lower MER/ASR, i.e. improved robustness.")


if __name__ == "__main__":
    main()
