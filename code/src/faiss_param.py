import os
import logging
import operator
from langchain_community.vectorstores import FAISS
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sized,
    Tuple,
    Union,
)
import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
    maximal_marginal_relevance,
)
import logging
logger = logging.getLogger(__name__)


def dependable_faiss_import(no_avx2: Optional[bool] = None) -> Any:
    """
    Import faiss if available, otherwise raise error.
    If FAISS_NO_AVX2 environment variable is set, it will be considered
    to load FAISS with no AVX2 optimization.

    Args:
        no_avx2: Load FAISS strictly with no AVX2 optimization
            so that the vectorstore is portable and compatible with other devices.
    """
    if no_avx2 is None and "FAISS_NO_AVX2" in os.environ:
        no_avx2 = bool(os.getenv("FAISS_NO_AVX2"))

    try:
        if no_avx2:
            from faiss import swigfaiss as faiss
        else:
            import faiss
    except ImportError:
        raise ImportError(
            "Could not import faiss python package. "
            "Please install it with `pip install faiss-gpu` (for CUDA supported GPU) "
            "or `pip install faiss-cpu` (depending on Python version)."
        )
    return faiss


class FAISS_param(FAISS):

    def _build_ivf_search_params(self, id_selector):
        faiss = dependable_faiss_import()
        params_cls = None
        if hasattr(faiss, "SearchParametersIVF"):
            params_cls = faiss.SearchParametersIVF
        elif hasattr(faiss, "IVFSearchParameters"):
            params_cls = faiss.IVFSearchParameters

        if params_cls is None:
            raise AttributeError("No IVF search parameter class found in faiss module.")

        params = params_cls()
        params.sel = id_selector
        return params

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Union[Callable, Dict[str, Any]]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Union[Callable, Dict[str, Any]]]): Filter by metadata.
                Defaults to None. If a callable, it must take as input the
                metadata dict of Document and return a bool.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.
            **kwargs: kwargs to be passed to similarity search. Can include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of documents most similar to the query text and L2 distance
            in float for each. Lower score represents more similarity.
        """
        faiss = dependable_faiss_import()
        vector = np.array([embedding], dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        id_selector = kwargs.get("id_selector")
        manual_id_filter = False
        search_k = k if filter is None else fetch_k

        if id_selector is not None:
            search_params = self._build_ivf_search_params(id_selector)
            try:
                scores, indices = self.index.search(
                    vector,
                    search_k,
                    params=search_params,
                )
            except TypeError:
                # Some faiss python bindings do not expose `params=` on Index.search.
                # Fallback: retrieve without params and filter by selector manually.
                manual_id_filter = True
                fallback_k = search_k
                if hasattr(self.index, "ntotal"):
                    fallback_k = min(max(search_k * 8, search_k), int(self.index.ntotal))
                scores, indices = self.index.search(vector, fallback_k)
        else:
            scores, indices = self.index.search(vector, search_k)
        docs = []

        if filter is not None:
            filter_func = self._create_filter_func(filter)

        for j, i in enumerate(indices[0]):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            if manual_id_filter and not id_selector.is_member(int(i)):
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            if filter is not None:
                if filter_func(doc.metadata):
                    docs.append((doc, scores[0][j]))
            else:
                docs.append((doc, scores[0][j]))

        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None:
            cmp = (
                operator.ge
                if self.distance_strategy
                in (DistanceStrategy.MAX_INNER_PRODUCT, DistanceStrategy.JACCARD)
                else operator.le
            )
            docs = [
                (doc, similarity)
                for doc, similarity in docs
                if cmp(similarity, score_threshold)
            ]
        return docs[:k]

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        *,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Union[Callable, Dict[str, Any]]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs and their similarity scores selected using the maximal marginal
            relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents and similarity scores selected by maximal marginal
                relevance and score for each.
        """
        scores, indices = self.index.search(
            np.array([embedding], dtype=np.float32),
            fetch_k if filter is None else fetch_k * 2,
        )
        if filter is not None:
            filter_func = self._create_filter_func(filter)
            filtered_indices = []
            for i in indices[0]:
                if i == -1:
                    # This happens when not enough docs are returned.
                    continue
                _id = self.index_to_docstore_id[i]
                doc = self.docstore.search(_id)
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                if filter_func(doc.metadata):
                    filtered_indices.append(i)
            indices = np.array([filtered_indices])
        # -1 happens when not enough docs are returned.
        embeddings = [self.index.reconstruct(int(i)) for i in indices[0] if i != -1]
        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )

        docs_and_scores = []
        for i in mmr_selected:
            if indices[0][i] == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[indices[0][i]]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs_and_scores.append((doc, scores[0][i]))

        return docs_and_scores
