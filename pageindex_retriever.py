"""
PageIndex retriever wrapper (optional)
- Uses VectifyAI PageIndex if installed (vectorless page-index)
- Falls back to a BM25 page-level retriever using rank_bm25 if PageIndex is unavailable

Provides:
- class PageIndexRetriever
    - build_index(documents)
    - retrieve(query, k)

Documents expected: list of langchain.schema.Document with metadata containing 'page' and 'document'
"""
from typing import List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Try to import PageIndex (VectifyAI). If not available, we'll fallback to BM25.
try:
    import pageindex as _pageindex  # type: ignore
    PAGEINDEX_AVAILABLE = True
    logger.info("PageIndex package available; will use PageIndex for page-level retrieval")
except Exception:
    _pageindex = None
    PAGEINDEX_AVAILABLE = False
    logger.info("PageIndex package not available; falling back to BM25 page retriever")


# BM25 fallback
try:
    from rank_bm25 import BM25Okapi
    import numpy as np
    BM25_AVAILABLE = True
except Exception:
    BM25Okapi = None
    BM25_AVAILABLE = False


class PageIndexRetriever:
    """Unified interface for page-level retrieval. Uses PageIndex if available, otherwise BM25."""

    def __init__(self, documents: List[Any] = None):
        self.documents = documents or []
        self._use_pageindex = PAGEINDEX_AVAILABLE
        self._pageindex_index = None
        self._bm25 = None
        self._tokenized = None
        if documents:
            self.build_index(documents)

    def build_index(self, documents: List[Any]):
        """Build the underlying index from page-level documents"""
        self.documents = documents
        if self._use_pageindex:
            try:
                # Build PageIndex index: create a mapping of id -> text and metadata
                idx = _pageindex.PageIndex()
                for i, doc in enumerate(documents):
                    text = doc.page_content
                    meta = doc.metadata if hasattr(doc, 'metadata') else {}
                    idx.add_page(str(i), text, metadata=meta)
                idx.build()
                self._pageindex_index = idx
                logger.info("Built PageIndex index with %d pages", len(documents))
            except Exception as e:
                logger.warning("Failed to build PageIndex index, falling back to BM25: %s", e)
                self._use_pageindex = False
                self._build_bm25(documents)
        else:
            self._build_bm25(documents)

    def _build_bm25(self, documents: List[Any]):
        if not BM25_AVAILABLE:
            logger.error("BM25 (rank_bm25) not available; page-level retrieval disabled")
            self._bm25 = None
            return
        # Tokenize documents simply
        tokenized = [self._tokenize(d.page_content) for d in documents]
        self._tokenized = tokenized
        self._bm25 = BM25Okapi(tokenized)
        logger.info("Built BM25 page-level index for %d pages", len(documents))

    def _tokenize(self, text: str):
        toks = [t.strip('.,!?;:').lower() for t in text.split() if len(t) > 2]
        return toks

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Any, float]]:
        """Retrieve top-k pages as (Document, score) tuples"""
        results = []
        if self._use_pageindex and self._pageindex_index is not None:
            try:
                hits = self._pageindex_index.query(query, top_k=k)
                for hit in hits:
                    # PageIndex hit expected to include id, text, metadata, score
                    doc_id = int(hit['id']) if 'id' in hit and str(hit['id']).isdigit() else None
                    score = hit.get('score', 1.0)
                    if doc_id is not None and doc_id < len(self.documents):
                        results.append((self.documents[doc_id], float(score)))
                return results
            except Exception as e:
                logger.warning("PageIndex query failed: %s. Falling back to BM25", e)
                # Fall through to BM25

        # BM25 fallback
        if self._bm25 is None:
            logger.error("No page-level index available (PageIndex and BM25 both unavailable)")
            return []

        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:k]
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append((self.documents[idx], float(scores[idx])))
        return results

