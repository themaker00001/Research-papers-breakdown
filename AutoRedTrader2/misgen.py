"""
Misinformation generation (MisGen) with the paper's three composable
operators, quality filtering (A_sim / A_det), and history-aware strategy
selection (Polya-urn sampler with return-dependent bias nudging).

Pipeline per candidate (Algorithm 1):
  1. sample (bias, minor, rewrite) from current distribution pi_t
  2. optionally inject bias into the agent context (Phi_Bias)
  3. apply Minor perturbation to the news text (Phi_Minor)
  4. check A_sim >= tau_sim and A_det <= tau_det; if it passes, keep it
  5. else apply Rewrite (Phi_Rewrite) to launder style and re-check A_det
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import prompts
from config import CONFIG
from data import NewsItem
from ollama_client import OllamaClient
from retrieval import a_sim


@dataclass
class HistoryEffect:
    """Counts of how often each strategy type flipped the agent's decision."""
    bias: Dict[str, int] = field(default_factory=dict)
    minor: Dict[str, int] = field(default_factory=dict)
    rewrite: Dict[str, int] = field(default_factory=dict)

    def bump(self, bias: Optional[str], minor: str, rewrite: Optional[str]):
        if bias:
            self.bias[bias] = self.bias.get(bias, 0) + 1
        self.minor[minor] = self.minor.get(minor, 0) + 1
        if rewrite:
            self.rewrite[rewrite] = self.rewrite.get(rewrite, 0) + 1


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _polya_sample(options: List[str], history: Dict[str, int],
                  alpha: float, rng: random.Random) -> str:
    """Weighted choice: w_s = 1 + alpha * h_s, then normalize."""
    weights = [1.0 + alpha * history.get(o, 0) for o in options]
    total = sum(weights)
    probs = [w / total for w in weights]
    return rng.choices(options, weights=probs, k=1)[0]


class MisGen:
    def __init__(self, client: OllamaClient, cfg=CONFIG):
        self.client = client
        self.cfg = cfg
        self.rng = random.Random(cfg.seed + 1)
        self.history = HistoryEffect()

    # ---- strategy selection (approximates pi_t) ----
    def select_strategies(self, cumulative_return: float
                          ) -> Tuple[Optional[str], str, str]:
        cfg = self.cfg
        # bias applied with probability p
        bias = None
        if self.rng.random() < cfg.bias_prob:
            bias = _polya_sample(cfg.biases, self.history.bias,
                                 cfg.alpha, self.rng)
            # return-dependent nudge: winners -> overconfidence, losers -> loss aversion
            if cumulative_return > 0 and self.rng.random() < cfg.delta:
                bias = "overconfidence"
            elif cumulative_return < 0 and self.rng.random() < cfg.delta:
                bias = "loss_aversion"
        minor = _polya_sample(cfg.minor_types, self.history.minor,
                              cfg.alpha, self.rng)
        rewrite = _polya_sample(cfg.rewrite_styles, self.history.rewrite,
                                cfg.alpha, self.rng)
        return bias, minor, rewrite

    # ---- operators ----
    def _apply_minor(self, news: str, mtype: str) -> str:
        out = self.client.chat_json(
            prompts.MINOR_SYSTEM,
            prompts.MINOR_USER_TEMPLATE.format(
                mtype=mtype, instruction=prompts.minor_instruction(mtype),
                news=news),
        )
        return (out or {}).get("response", "").strip() or news

    def _apply_rewrite(self, text: str, style: str) -> str:
        out = self.client.chat_json(
            prompts.REWRITE_SYSTEM,
            prompts.REWRITE_USER_TEMPLATE.format(
                style=style, guidance=prompts.rewrite_guidance(style),
                text=text),
        )
        return (out or {}).get("response", "").strip() or text

    def _detectability(self, text: str, original: Optional[str] = None) -> float:
        # base (lenient) judge ignores the reference; strict subclass uses it
        out = self.client.chat_json(
            prompts.DETECT_SYSTEM,
            prompts.DETECT_USER_TEMPLATE.format(text=text),
            temperature=0.0,
        )
        try:
            return _clamp01(float((out or {}).get("misinformation_prob", 1.0)))
        except Exception:  # noqa: BLE001
            return 1.0

    # ---- main generation (Algorithm 1) ----
    def generate(self, real_news: NewsItem, cumulative_return: float,
                 day_idx: int, n_candidates: int
                 ) -> List[Tuple[NewsItem, Optional[str]]]:
        """
        Returns a list of (misinfo NewsItem, bias_prompt) pairs that passed
        filtering. bias_prompt is the system-prompt injection to feed the agent.
        """
        cfg = self.cfg
        kept: List[Tuple[NewsItem, Optional[str]]] = []
        for c in range(n_candidates):
            bias, minor, rewrite = self.select_strategies(cumulative_return)
            bias_prompt = prompts.BIAS_PROMPTS.get(bias) if bias else None

            text = self._apply_minor(real_news.text, minor)
            sim = a_sim(text, real_news.text)
            det = self._detectability(text, original=real_news.text)

            accepted = sim >= cfg.tau_sim and det <= cfg.tau_det
            used_rewrite: Optional[str] = None

            if not accepted:
                # constraint-repair: launder style and re-check detectability
                for _ in range(cfg.max_retries):
                    laundered = self._apply_rewrite(text, rewrite)
                    det2 = self._detectability(laundered, original=real_news.text)
                    sim2 = a_sim(laundered, real_news.text)
                    if det2 <= cfg.tau_det and sim2 >= cfg.tau_sim:
                        text, sim, det = laundered, sim2, det2
                        used_rewrite = rewrite
                        accepted = True
                        break

            if accepted:
                item = NewsItem(
                    id=f"inj-{day_idx}-{c}", date=real_news.date, text=text,
                    is_injected=True, strategies=(bias, minor, used_rewrite),
                )
                kept.append((item, bias_prompt))
                if cfg.verbose:
                    tag = f"bias={bias},minor={minor},rewrite={used_rewrite}"
                    print(f"    [misgen] kept ({tag}) "
                          f"sim={sim:.2f} det={det:.2f}")
        return kept

    def update_history(self, item: NewsItem, flipped: bool):
        if flipped and item.strategies:
            bias, minor, rewrite = item.strategies
            self.history.bump(bias, minor, rewrite)
