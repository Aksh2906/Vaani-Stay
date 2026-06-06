"""
clients.py — Cached model loaders for the Himachal Homestay RAG pipeline.

Unchanged from the Feynman pipeline — BGE models work perfectly for
Himachal travel content retrieval and reranking.

Models:
  - BAAI/bge-small-en-v1.5  → 384-dim dense embeddings
  - BAAI/bge-reranker-base  → Cross-encoder reranker
"""

import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder


@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    """
    Loads and caches the BGE embedding model.
    384-dimensional normalized embeddings.
    """
    return SentenceTransformer('BAAI/bge-small-en-v1.5')


@st.cache_resource
def get_reranker_model() -> CrossEncoder:
    """
    Loads and caches the BGE cross-encoder reranker.
    Used to score (query, window) pairs in the final reranking step.
    """
    return CrossEncoder('BAAI/bge-reranker-base')
