"""
InfraHeal AI — BM25-based Knowledge Base
==========================================
Provides retrieval-augmented generation (RAG) over runbooks and past
incidents using BM25 (Okapi) ranking.  No embedding model needed —
pure lexical retrieval with intelligent tokenisation and category
filtering.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenise(text: str) -> List[str]:
    """Simple whitespace + punctuation tokeniser with lowercasing.

    Splits on non-alphanumeric boundaries, lowercases, and removes
    tokens shorter than 2 characters.  Suitable for BM25 on
    technical infrastructure text.

    Args:
        text: Input text to tokenise.

    Returns:
        List of lowercase tokens.
    """
    tokens = re.findall(r"[a-zA-Z0-9_\-\.]+", text.lower())
    return [t for t in tokens if len(t) >= 2]


class KnowledgeBase:
    """BM25-based retrieval over runbooks and past incidents.

    Indexes documents at construction time and provides fast lexical
    search via :pypi:`rank_bm25`.  Supports category-based pre-filtering
    and formatted context generation for agent prompts.

    Example::

        kb = KnowledgeBase(runbooks=my_runbooks, past_incidents=my_incidents)
        context_str = kb.get_context("high CPU usage on web servers", category="infrastructure")
    """

    def __init__(
        self,
        runbooks: Optional[List[dict]] = None,
        past_incidents: Optional[List[dict]] = None,
    ) -> None:
        """Build BM25 indexes over runbooks and past incidents.

        Args:
            runbooks: List of runbook dicts.  Expected keys:
                ``id``, ``title``, ``category``, ``symptoms``,
                ``root_causes``, ``resolution_steps``, ``prevention``,
                ``tags``.
            past_incidents: List of past-incident dicts.  Expected
                keys: ``id``, ``title``, ``category``, ``root_cause``,
                ``resolution``, ``severity``, ``tags``.
        """
        self.runbooks = runbooks or []
        self.past_incidents = past_incidents or []

        # ── Index runbooks ──
        self._runbook_corpus: List[str] = []
        self._runbook_tokens: List[List[str]] = []
        for rb in self.runbooks:
            doc = self._runbook_to_text(rb)
            self._runbook_corpus.append(doc)
            self._runbook_tokens.append(_tokenise(doc))

        self._runbook_bm25: Optional[BM25Okapi] = None
        if self._runbook_tokens:
            self._runbook_bm25 = BM25Okapi(self._runbook_tokens)
            logger.info("Indexed %d runbooks for BM25 search", len(self.runbooks))

        # ── Index past incidents ──
        self._incident_corpus: List[str] = []
        self._incident_tokens: List[List[str]] = []
        for inc in self.past_incidents:
            doc = self._incident_to_text(inc)
            self._incident_corpus.append(doc)
            self._incident_tokens.append(_tokenise(doc))

        self._incident_bm25: Optional[BM25Okapi] = None
        if self._incident_tokens:
            self._incident_bm25 = BM25Okapi(self._incident_tokens)
            logger.info("Indexed %d past incidents for BM25 search", len(self.past_incidents))

    # ── Public Search API ────────────────────────────────────────

    def search_runbooks(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
    ) -> List[Tuple[dict, float]]:
        """Search runbooks by query with optional category filter.

        Args:
            query: Natural-language search query.
            top_k: Maximum number of results to return.
            category: If set, only return runbooks matching this category.

        Returns:
            List of ``(runbook_dict, bm25_score)`` tuples, sorted by
            descending relevance.
        """
        if not self._runbook_bm25 or not self.runbooks:
            logger.debug("No runbooks indexed — returning empty results")
            return []

        query_tokens = _tokenise(query)
        if not query_tokens:
            return []

        scores = self._runbook_bm25.get_scores(query_tokens)

        # Pair scores with runbooks and filter
        scored: List[Tuple[dict, float]] = []
        for idx, score in enumerate(scores):
            rb = self.runbooks[idx]
            if category and rb.get("category", "").lower() != category.lower():
                continue
            scored.append((rb, float(score)))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        results = scored[:top_k]

        logger.debug(
            "Runbook search for '%s': %d results (top score=%.2f)",
            query[:60],
            len(results),
            results[0][1] if results else 0.0,
        )
        return results

    def search_incidents(
        self,
        query: str,
        top_k: int = 3,
        category: Optional[str] = None,
    ) -> List[Tuple[dict, float]]:
        """Search past incidents by query.

        Args:
            query: Natural-language search query.
            top_k: Maximum number of results.
            category: Optional category filter.

        Returns:
            List of ``(incident_dict, bm25_score)`` tuples, sorted by
            descending relevance.
        """
        if not self._incident_bm25 or not self.past_incidents:
            logger.debug("No past incidents indexed — returning empty results")
            return []

        query_tokens = _tokenise(query)
        if not query_tokens:
            return []

        scores = self._incident_bm25.get_scores(query_tokens)

        scored: List[Tuple[dict, float]] = []
        for idx, score in enumerate(scores):
            inc = self.past_incidents[idx]
            if category and inc.get("category", "").lower() != category.lower():
                continue
            scored.append((inc, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = scored[:top_k]

        logger.debug(
            "Incident search for '%s': %d results (top score=%.2f)",
            query[:60],
            len(results),
            results[0][1] if results else 0.0,
        )
        return results

    def get_context(
        self,
        query: str,
        category: Optional[str] = None,
        max_runbooks: int = 5,
        max_incidents: int = 3,
    ) -> str:
        """Get a formatted context string for agent prompts.

        Combines top-matching runbooks and past incidents into a
        readable text block suitable for injection into LLM prompts.

        Args:
            query: Natural-language query.
            category: Optional category for pre-filtering.
            max_runbooks: Maximum runbook results.
            max_incidents: Maximum incident results.

        Returns:
            Formatted multi-line string with runbook and incident
            context.  Returns an empty string if no results are found.
        """
        sections: List[str] = []

        # ── Runbooks ──
        runbook_results = self.search_runbooks(query, top_k=max_runbooks, category=category)
        if runbook_results:
            sections.append("### Relevant Runbooks\n")
            for rank, (rb, score) in enumerate(runbook_results, 1):
                sections.append(self._format_runbook(rb, rank, score))

        # ── Past Incidents ──
        incident_results = self.search_incidents(query, top_k=max_incidents, category=category)
        if incident_results:
            sections.append("### Similar Past Incidents\n")
            for rank, (inc, score) in enumerate(incident_results, 1):
                sections.append(self._format_incident(inc, rank, score))

        context = "\n".join(sections)
        logger.info(
            "RAG context: %d runbooks, %d incidents, %d chars",
            len(runbook_results), len(incident_results), len(context),
        )
        return context

    # ── Document Management ──────────────────────────────────────

    def add_runbook(self, runbook: dict) -> None:
        """Add a single runbook and rebuild the BM25 index.

        Args:
            runbook: Runbook dict with standard keys.
        """
        self.runbooks.append(runbook)
        doc = self._runbook_to_text(runbook)
        self._runbook_corpus.append(doc)
        self._runbook_tokens.append(_tokenise(doc))
        self._runbook_bm25 = BM25Okapi(self._runbook_tokens)
        logger.info("Added runbook '%s' — reindexed (%d total)", runbook.get("id", "?"), len(self.runbooks))

    def add_incident(self, incident: dict) -> None:
        """Add a single past incident and rebuild the BM25 index.

        Args:
            incident: Incident dict with standard keys.
        """
        self.past_incidents.append(incident)
        doc = self._incident_to_text(incident)
        self._incident_corpus.append(doc)
        self._incident_tokens.append(_tokenise(doc))
        self._incident_bm25 = BM25Okapi(self._incident_tokens)
        logger.info("Added incident '%s' — reindexed (%d total)", incident.get("id", "?"), len(self.past_incidents))

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics.

        Returns:
            Dict with counts of indexed runbooks and incidents,
            and average document lengths.
        """
        rb_avg = (
            round(sum(len(t) for t in self._runbook_tokens) / len(self._runbook_tokens), 1)
            if self._runbook_tokens else 0
        )
        inc_avg = (
            round(sum(len(t) for t in self._incident_tokens) / len(self._incident_tokens), 1)
            if self._incident_tokens else 0
        )
        return {
            "runbooks_indexed": len(self.runbooks),
            "incidents_indexed": len(self.past_incidents),
            "avg_runbook_tokens": rb_avg,
            "avg_incident_tokens": inc_avg,
        }

    # ── Serialisation Helpers ────────────────────────────────────

    @staticmethod
    def _runbook_to_text(rb: dict) -> str:
        """Flatten a runbook dict into a searchable text document."""
        parts = [
            rb.get("title", ""),
            rb.get("category", ""),
            " ".join(rb.get("symptoms", [])),
            " ".join(rb.get("root_causes", [])),
            " ".join(rb.get("resolution_steps", [])),
            " ".join(rb.get("prevention", [])),
            " ".join(rb.get("tags", [])),
        ]
        return " ".join(filter(None, parts))

    @staticmethod
    def _incident_to_text(inc: dict) -> str:
        """Flatten a past-incident dict into a searchable text document."""
        parts = [
            inc.get("title", ""),
            inc.get("category", ""),
            inc.get("root_cause", ""),
            inc.get("resolution", ""),
            inc.get("severity", ""),
            " ".join(inc.get("tags", [])),
        ]
        return " ".join(filter(None, parts))

    @staticmethod
    def _format_runbook(rb: dict, rank: int, score: float) -> str:
        """Format a single runbook for context injection."""
        symptoms = "\n".join(f"  - {s}" for s in rb.get("symptoms", []))
        root_causes = "\n".join(f"  - {r}" for r in rb.get("root_causes", []))
        steps = "\n".join(f"  {i}. {s}" for i, s in enumerate(rb.get("resolution_steps", []), 1))
        prevention = "\n".join(f"  - {p}" for p in rb.get("prevention", []))

        return (
            f"**Runbook #{rank}** (ID: {rb.get('id', '?')}, score: {score:.2f})\n"
            f"**Title**: {rb.get('title', 'Untitled')}\n"
            f"**Category**: {rb.get('category', '?')}\n"
            f"**Symptoms**:\n{symptoms}\n"
            f"**Root Causes**:\n{root_causes}\n"
            f"**Resolution Steps**:\n{steps}\n"
            f"**Prevention**:\n{prevention}\n"
        )

    @staticmethod
    def _format_incident(inc: dict, rank: int, score: float) -> str:
        """Format a single past incident for context injection."""
        return (
            f"**Past Incident #{rank}** (ID: {inc.get('id', '?')}, score: {score:.2f})\n"
            f"**Title**: {inc.get('title', 'Untitled')}\n"
            f"**Category**: {inc.get('category', '?')}\n"
            f"**Severity**: {inc.get('severity', '?')}\n"
            f"**Root Cause**: {inc.get('root_cause', 'Unknown')}\n"
            f"**Resolution**: {inc.get('resolution', 'No resolution recorded')}\n"
        )
