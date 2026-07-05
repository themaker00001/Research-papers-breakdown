"""
Lightweight retrieval over the news pool (stand-in for the paper's FAISS
semantic search). TF-IDF + cosine is deterministic, dependency-light, and
sufficient for a demo: injected misinformation is textually close to real
news, so it competes for retrieval slots as intended.

Also provides the similarity evaluator A_sim used by the MisGen filter.
"""

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data import NewsItem


def retrieve(query: str, pool: List[NewsItem], k: int) -> List[NewsItem]:
    """Return top-k news items most similar to the query."""
    if not pool:
        return []
    texts = [query] + [n.text for n in pool]
    vec = TfidfVectorizer(stop_words="english").fit_transform(texts)
    sims = cosine_similarity(vec[0:1], vec[1:]).ravel()
    order = sims.argsort()[::-1][:k]
    return [pool[i] for i in order]


def a_sim(candidate: str, original: str) -> float:
    """Semantic-ish similarity in [0,1] between a candidate and the original."""
    if not candidate or not original:
        return 0.0
    vec = TfidfVectorizer(stop_words="english").fit_transform(
        [candidate, original]
    )
    return float(cosine_similarity(vec[0:1], vec[1:2]).ravel()[0])
