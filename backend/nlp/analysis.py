"""
Policy analysis module for GovScheme Advisor.

Provides four NLP capabilities on the scheme corpus:
  1. LDA topic modeling             (sklearn LatentDirichletAllocation)
  2. Semantic topics ("BERTopic"-style) via sentence-transformer embeddings + KMeans
     (falls back to TF-IDF + KMeans if sentence-transformers is unavailable)
  3. Keyword extraction             (TF-IDF, top-N per scheme)
  4. Lexicon-based sentiment/tone   (no external corpus download)
  5. Cross-state and cross-category policy comparison

Designed to run cheaply at startup and serve pre-computed results to the frontend.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Optional: sentence-transformers for semantic embeddings. Falls back to TF-IDF.
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception:
    _HAS_ST = False


# ──────────────────────────────────────────────────────────────────────
# Sentiment lexicon (lightweight, no corpus download required)
# ──────────────────────────────────────────────────────────────────────

_POSITIVE_WORDS = {
    "benefit", "benefits", "support", "supports", "help", "helps", "free",
    "subsidy", "subsidised", "subsidized", "grant", "grants", "scholarship",
    "assistance", "empower", "empowered", "empowerment", "welfare", "improve",
    "improved", "improvement", "boost", "encourage", "encouraged", "incentive",
    "incentives", "promote", "promoted", "promotion", "reward", "rewards",
    "uplift", "develop", "development", "growth", "opportunity", "opportunities",
    "secure", "security", "safe", "safety", "protect", "protected", "protection",
    "relief", "easy", "simple", "fast", "quick", "comprehensive", "successful",
    "guaranteed", "ensure", "ensures", "ensured", "enable", "enables",
}

_NEGATIVE_WORDS = {
    "restricted", "restriction", "restrictions", "limit", "limits", "limited",
    "deny", "denied", "denial", "reject", "rejected", "rejection", "fine",
    "fines", "penalty", "penalties", "punish", "punishment", "exclude",
    "excluded", "exclusion", "barred", "prohibited", "prohibition", "fraud",
    "fraudulent", "violation", "violate", "ineligible", "disqualify",
    "disqualified", "delay", "delayed", "fail", "failed", "failure", "loss",
    "losses", "decline", "declined", "declined", "shortfall", "deficit",
    "below", "poor", "poverty", "hardship", "distress",
}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

_TEXT_FIELDS = ("name", "description", "eligibility_text", "benefits", "tags")


def _scheme_text(scheme: Dict) -> str:
    parts: List[str] = []
    for f in _TEXT_FIELDS:
        v = scheme.get(f)
        if isinstance(v, list):
            parts.append(" ".join(str(x) for x in v))
        elif v:
            parts.append(str(v))
    return " ".join(parts)


def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _label_topic(top_words: List[str]) -> str:
    """Heuristic human-readable label from topic top words."""
    s = " ".join(top_words).lower()
    rules = [
        ("Health & Insurance", ("health", "medical", "hospital", "insurance", "ayushman", "disease")),
        ("Education & Scholarship", ("scholarship", "student", "education", "school", "college", "exam")),
        ("Agriculture & Farmer", ("farmer", "crop", "agriculture", "kisan", "land", "irrigation")),
        ("Pension & Senior", ("pension", "senior", "elderly", "old", "vrid")),
        ("Women & Child", ("women", "girl", "mother", "widow", "child", "balika", "mahila")),
        ("Housing & Shelter", ("house", "housing", "awas", "shelter", "home", "pradhan mantri awas")),
        ("Employment & Skill", ("employment", "job", "skill", "training", "kaushal", "rozgar", "mgnrega")),
        ("Finance & Loan", ("loan", "credit", "bank", "mudra", "startup", "msme", "finance")),
        ("Social Security", ("social", "security", "disability", "minority", "sc", "st", "obc")),
        ("Food & Nutrition", ("food", "ration", "nutrition", "midday", "pds")),
    ]
    for label, kws in rules:
        if any(kw in s for kw in kws):
            return label
    return "General Welfare"


# ──────────────────────────────────────────────────────────────────────
# PolicyAnalyzer
# ──────────────────────────────────────────────────────────────────────


class PolicyAnalyzer:
    """Stateful analyzer that fits once on the scheme corpus and serves results."""

    def __init__(self, n_topics: int = 8, embed_model: str = "all-MiniLM-L6-v2"):
        self.n_topics = n_topics
        self.embed_model_name = embed_model
        self._embed_model: Optional[object] = None  # lazy

        # State after fit()
        self.schemes: List[Dict] = []
        self.cleaned: List[str] = []

        # LDA artifacts
        self.lda: Optional[LatentDirichletAllocation] = None
        self.count_vec: Optional[CountVectorizer] = None
        self.lda_topic_words: List[List[str]] = []
        self.lda_topic_labels: List[str] = []
        self.lda_doc_topics: Optional[np.ndarray] = None  # (n_docs, n_topics)

        # Semantic (BERTopic-style) artifacts
        self.semantic_topic_words: List[List[str]] = []
        self.semantic_topic_labels: List[str] = []
        self.semantic_doc_topics: Optional[np.ndarray] = None  # (n_docs,)
        self.semantic_backend: str = "none"  # "sbert" or "tfidf"

        # Keywords
        self.tfidf_vec: Optional[TfidfVectorizer] = None
        self.keywords_per_scheme: Dict[str, List[str]] = {}

        # Sentiment
        self.sentiment_per_scheme: Dict[str, Dict] = {}

    # ─────────────────────────── Fit ───────────────────────────

    def fit(self, schemes: List[Dict]) -> Dict:
        """Run all analyses on the scheme corpus. Returns a summary."""
        self.schemes = schemes
        self.cleaned = [_clean(_scheme_text(s)) for s in schemes]

        self._fit_lda()
        self._fit_keywords()
        self._fit_semantic_topics()
        self._fit_sentiment()

        return self.summary()

    # ─────────────────────────── LDA ───────────────────────────

    def _fit_lda(self) -> None:
        if not self.cleaned:
            return
        self.count_vec = CountVectorizer(
            max_features=600, stop_words="english", min_df=2, max_df=0.85
        )
        try:
            X = self.count_vec.fit_transform(self.cleaned)
        except ValueError:
            # Tiny corpora: relax filters
            self.count_vec = CountVectorizer(stop_words="english")
            X = self.count_vec.fit_transform(self.cleaned)

        n_topics = min(self.n_topics, max(2, X.shape[0] // 4))
        self.lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=42, max_iter=15, learning_method="batch"
        )
        self.lda.fit(X)
        vocab = self.count_vec.get_feature_names_out()

        self.lda_topic_words = []
        self.lda_topic_labels = []
        for k in range(n_topics):
            top_idx = self.lda.components_[k].argsort()[-12:][::-1]
            words = [vocab[i] for i in top_idx]
            self.lda_topic_words.append(words)
            self.lda_topic_labels.append(_label_topic(words))

        self.lda_doc_topics = self.lda.transform(X)

    # ─────────────────────────── Keywords ───────────────────────────

    def _fit_keywords(self) -> None:
        if not self.cleaned:
            return
        self.tfidf_vec = TfidfVectorizer(
            max_features=2000, stop_words="english", ngram_range=(1, 2), min_df=1
        )
        try:
            X = self.tfidf_vec.fit_transform(self.cleaned)
        except ValueError:
            return
        vocab = self.tfidf_vec.get_feature_names_out()
        for i, scheme in enumerate(self.schemes):
            row = X[i].toarray().ravel()
            top_idx = row.argsort()[-10:][::-1]
            kws = [vocab[j] for j in top_idx if row[j] > 0]
            sid = scheme.get("scheme_id") or scheme.get("name", f"scheme_{i}")
            self.keywords_per_scheme[sid] = kws

    # ─────────────────────────── Semantic topics ───────────────────────────

    def _fit_semantic_topics(self) -> None:
        if not self.cleaned:
            return

        embeddings: Optional[np.ndarray] = None
        if _HAS_ST:
            try:
                if self._embed_model is None:
                    self._embed_model = SentenceTransformer(self.embed_model_name)
                embeddings = np.array(
                    self._embed_model.encode(  # type: ignore[union-attr]
                        [_scheme_text(s) for s in self.schemes],
                        show_progress_bar=False,
                        normalize_embeddings=True,
                    )
                )
                self.semantic_backend = "sbert"
            except Exception:
                embeddings = None

        if embeddings is None and self.tfidf_vec is not None:
            X = self.tfidf_vec.transform(self.cleaned)
            embeddings = X.toarray()
            self.semantic_backend = "tfidf"

        if embeddings is None:
            return

        n_clusters = min(self.n_topics, max(2, embeddings.shape[0] // 4))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        self.semantic_doc_topics = labels

        # Top words per cluster from TF-IDF (interpretable)
        self.semantic_topic_words = []
        self.semantic_topic_labels = []
        if self.tfidf_vec is None:
            return
        vocab = self.tfidf_vec.get_feature_names_out()
        tfidf_X = self.tfidf_vec.transform(self.cleaned).toarray()
        for k in range(n_clusters):
            mask = labels == k
            if not mask.any():
                self.semantic_topic_words.append([])
                self.semantic_topic_labels.append(f"Cluster {k}")
                continue
            mean_tfidf = tfidf_X[mask].mean(axis=0)
            top_idx = mean_tfidf.argsort()[-12:][::-1]
            words = [vocab[i] for i in top_idx if mean_tfidf[i] > 0]
            self.semantic_topic_words.append(words)
            self.semantic_topic_labels.append(_label_topic(words))

    # ─────────────────────────── Sentiment ───────────────────────────

    def _score_sentiment(self, text: str) -> Dict:
        toks = _clean(text).split()
        if not toks:
            return {"polarity": 0.0, "label": "neutral", "pos": 0, "neg": 0}
        pos = sum(1 for t in toks if t in _POSITIVE_WORDS)
        neg = sum(1 for t in toks if t in _NEGATIVE_WORDS)
        denom = pos + neg
        polarity = (pos - neg) / denom if denom else 0.0
        if polarity > 0.2:
            label = "positive"
        elif polarity < -0.2:
            label = "negative"
        else:
            label = "neutral"
        return {
            "polarity": round(float(polarity), 3),
            "label": label,
            "pos": pos,
            "neg": neg,
        }

    def _fit_sentiment(self) -> None:
        for s in self.schemes:
            sid = s.get("scheme_id") or s.get("name", "")
            self.sentiment_per_scheme[sid] = self._score_sentiment(_scheme_text(s))

    # ─────────────────────────── Outputs ───────────────────────────

    def lda_topics(self, n_words: int = 10) -> List[Dict]:
        out = []
        for i, words in enumerate(self.lda_topic_words):
            out.append(
                {
                    "topic_id": i,
                    "label": self.lda_topic_labels[i],
                    "top_words": words[:n_words],
                    "scheme_count": int((self.lda_doc_topics.argmax(axis=1) == i).sum())
                    if self.lda_doc_topics is not None
                    else 0,
                }
            )
        return out

    def semantic_topics(self, n_words: int = 10) -> List[Dict]:
        out = []
        for i, words in enumerate(self.semantic_topic_words):
            count = (
                int((self.semantic_doc_topics == i).sum())
                if self.semantic_doc_topics is not None
                else 0
            )
            out.append(
                {
                    "topic_id": i,
                    "label": self.semantic_topic_labels[i] if i < len(self.semantic_topic_labels) else f"Cluster {i}",
                    "top_words": words[:n_words],
                    "scheme_count": count,
                }
            )
        return out

    def schemes_in_topic(self, topic_id: int, kind: str = "lda", limit: int = 20) -> List[Dict]:
        if kind == "lda":
            if self.lda_doc_topics is None:
                return []
            order = (-self.lda_doc_topics[:, topic_id]).argsort()
            indices = [i for i in order if self.lda_doc_topics[i].argmax() == topic_id][:limit]
        else:
            if self.semantic_doc_topics is None:
                return []
            indices = [i for i, t in enumerate(self.semantic_doc_topics) if t == topic_id][:limit]

        results = []
        for i in indices:
            s = self.schemes[i]
            sid = s.get("scheme_id") or s.get("name", "")
            results.append(
                {
                    "scheme_id": sid,
                    "name": s.get("name", ""),
                    "category": s.get("category", ""),
                    "state": s.get("state", "All India"),
                    "description": (s.get("description") or "")[:240],
                    "keywords": self.keywords_per_scheme.get(sid, [])[:6],
                    "sentiment": self.sentiment_per_scheme.get(sid, {}).get("label", "neutral"),
                }
            )
        return results

    def sentiment_summary(self) -> Dict:
        labels = [v["label"] for v in self.sentiment_per_scheme.values()]
        polarities = [v["polarity"] for v in self.sentiment_per_scheme.values()]
        c = Counter(labels)
        return {
            "total": len(labels),
            "positive": c.get("positive", 0),
            "neutral": c.get("neutral", 0),
            "negative": c.get("negative", 0),
            "avg_polarity": round(float(np.mean(polarities)) if polarities else 0.0, 3),
        }

    def keyword_cloud(self, top_n: int = 60) -> List[Dict]:
        all_kw: Counter = Counter()
        for kws in self.keywords_per_scheme.values():
            all_kw.update(kws)
        return [{"word": w, "count": c} for w, c in all_kw.most_common(top_n)]

    def category_distribution(self) -> List[Dict]:
        c: Counter = Counter()
        for s in self.schemes:
            c[s.get("category", "other") or "other"] += 1
        return [{"category": k, "count": v} for k, v in c.most_common()]

    def state_distribution(self) -> List[Dict]:
        c: Counter = Counter()
        for s in self.schemes:
            c[s.get("state") or "All India"] += 1
        return [{"state": k, "count": v} for k, v in c.most_common()]

    def state_sentiment_matrix(self) -> List[Dict]:
        agg: Dict[str, List[float]] = defaultdict(list)
        for s in self.schemes:
            sid = s.get("scheme_id") or s.get("name", "")
            polarity = self.sentiment_per_scheme.get(sid, {}).get("polarity", 0.0)
            agg[s.get("state") or "All India"].append(polarity)
        rows = [
            {
                "state": st,
                "schemes": len(vals),
                "avg_polarity": round(float(np.mean(vals)), 3),
            }
            for st, vals in agg.items()
        ]
        rows.sort(key=lambda r: r["schemes"], reverse=True)
        return rows

    def category_sentiment_matrix(self) -> List[Dict]:
        agg: Dict[str, List[float]] = defaultdict(list)
        for s in self.schemes:
            sid = s.get("scheme_id") or s.get("name", "")
            polarity = self.sentiment_per_scheme.get(sid, {}).get("polarity", 0.0)
            agg[s.get("category") or "other"].append(polarity)
        return [
            {
                "category": cat,
                "schemes": len(vals),
                "avg_polarity": round(float(np.mean(vals)), 3),
            }
            for cat, vals in sorted(agg.items(), key=lambda kv: -len(kv[1]))
        ]

    def scheme_analysis(self, scheme_id: str) -> Optional[Dict]:
        for i, s in enumerate(self.schemes):
            sid = s.get("scheme_id") or s.get("name", "")
            if sid != scheme_id:
                continue
            lda_topic = (
                int(self.lda_doc_topics[i].argmax())
                if self.lda_doc_topics is not None
                else None
            )
            sem_topic = (
                int(self.semantic_doc_topics[i])
                if self.semantic_doc_topics is not None
                else None
            )
            return {
                "scheme_id": sid,
                "name": s.get("name", ""),
                "category": s.get("category", ""),
                "state": s.get("state", "All India"),
                "keywords": self.keywords_per_scheme.get(sid, []),
                "sentiment": self.sentiment_per_scheme.get(sid, {}),
                "lda_topic": {
                    "id": lda_topic,
                    "label": self.lda_topic_labels[lda_topic] if lda_topic is not None else None,
                }
                if lda_topic is not None
                else None,
                "semantic_topic": {
                    "id": sem_topic,
                    "label": self.semantic_topic_labels[sem_topic]
                    if sem_topic is not None and sem_topic < len(self.semantic_topic_labels)
                    else None,
                }
                if sem_topic is not None
                else None,
            }
        return None

    # ─────────────────────────── Bundle ───────────────────────────

    def summary(self) -> Dict:
        return {
            "fitted": bool(self.schemes),
            "n_schemes": len(self.schemes),
            "n_topics_lda": len(self.lda_topic_words),
            "n_topics_semantic": len(self.semantic_topic_words),
            "semantic_backend": self.semantic_backend,
            "sentiment_summary": self.sentiment_summary() if self.schemes else {},
        }

    def full_bundle(self) -> Dict:
        """All pre-computed analysis in one payload (for static export & overview UI)."""
        return {
            "summary": self.summary(),
            "lda_topics": self.lda_topics(),
            "semantic_topics": self.semantic_topics(),
            "sentiment_summary": self.sentiment_summary(),
            "keyword_cloud": self.keyword_cloud(),
            "category_distribution": self.category_distribution(),
            "state_distribution": self.state_distribution(),
            "state_sentiment": self.state_sentiment_matrix(),
            "category_sentiment": self.category_sentiment_matrix(),
            "schemes_per_lda_topic": [
                self.schemes_in_topic(i, kind="lda", limit=15)
                for i in range(len(self.lda_topic_words))
            ],
            "schemes_per_semantic_topic": [
                self.schemes_in_topic(i, kind="semantic", limit=15)
                for i in range(len(self.semantic_topic_words))
            ],
        }

    def export_to(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.full_bundle(), ensure_ascii=False, indent=2), encoding="utf-8")


# Singleton (mirrors nlp_engine pattern in this codebase)
policy_analyzer = PolicyAnalyzer()
