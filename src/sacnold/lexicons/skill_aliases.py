"""Canonical skill normalization for the Redrob ranking challenge.

Maps raw skill name strings → a canonical group key used for deduplication
and tier classification. This prevents "NLP" and "Natural Language Processing"
from counting as two different skills, and allows trust scores to be aggregated
at the concept level rather than the raw string level.

Groups are structured around the three tiers used for scoring:
  Tier A — core JD requirements (retrieval, ranking, embeddings, search infra, etc.)
  Tier B — strong supporting skills (ML, deep learning, LLMs, MLOps, etc.)
  Tier C — nice-to-have / adjacent context (data infra, LLM glue frameworks, etc.)
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Alias table: lower-cased raw skill name → canonical group key
# ---------------------------------------------------------------------------
SKILL_ALIASES: dict[str, str] = {
    # --- NLP / Language Understanding --------------------------------------
    "nlp": "nlp",
    "natural language processing": "nlp",
    "natural language understanding": "nlp",
    "nlu": "nlp",
    "text mining": "nlp",
    "text analytics": "nlp",
    "text classification": "nlp",
    "named entity recognition": "nlp",
    "ner": "nlp",
    "sentiment analysis": "nlp",
    "computational linguistics": "nlp",
    "language modelling": "nlp",
    "language modeling": "nlp",
    "question answering": "nlp",

    # --- LLMs / Transformers / Foundation Models ---------------------------
    "llm": "llm",
    "llms": "llm",
    "large language model": "llm",
    "large language models": "llm",
    "transformer": "llm",
    "transformers": "llm",
    "bert": "llm",
    "gpt": "llm",
    "t5": "llm",
    "generative ai": "llm",
    "generative-ai": "llm",
    "gen ai": "llm",

    # --- LLM Fine-tuning / PEFT --------------------------------------------
    "fine-tuning": "llm_finetune",
    "fine tuning": "llm_finetune",
    "finetuning": "llm_finetune",
    "lora": "llm_finetune",
    "qlora": "llm_finetune",
    "peft": "llm_finetune",
    "instruction tuning": "llm_finetune",
    "rlhf": "llm_finetune",
    "grpo": "llm_finetune",
    "dpo": "llm_finetune",

    # --- Machine Learning (general) ----------------------------------------
    "ml": "machine_learning",
    "machine learning": "machine_learning",
    "applied ml": "machine_learning",
    "applied machine learning": "machine_learning",
    "supervised learning": "machine_learning",
    "classification": "machine_learning",
    "regression": "machine_learning",
    "gradient boosting": "machine_learning",
    "xgboost": "machine_learning",
    "lightgbm": "machine_learning",
    "scikit-learn": "machine_learning",
    "sklearn": "machine_learning",

    # --- Deep Learning -----------------------------------------------------
    "deep learning": "deep_learning",
    "dl": "deep_learning",
    "neural networks": "deep_learning",
    "neural network": "deep_learning",
    "pytorch": "deep_learning",
    "tensorflow": "deep_learning",
    "keras": "deep_learning",
    "jax": "deep_learning",

    # --- Information Retrieval / Search ------------------------------------
    "information retrieval": "retrieval",
    "retrieval": "retrieval",
    "semantic search": "retrieval",
    "vector search": "retrieval",
    "hybrid search": "retrieval",
    "dense retrieval": "retrieval",
    "sparse retrieval": "retrieval",
    "passage retrieval": "retrieval",
    "document retrieval": "retrieval",
    "rag": "retrieval",
    "retrieval augmented generation": "retrieval",

    # --- BM25 / Lexical Search ---------------------------------------------
    "bm25": "retrieval_bm25",
    "okapi bm25": "retrieval_bm25",
    "tf-idf": "retrieval_bm25",
    "tfidf": "retrieval_bm25",
    "inverted index": "retrieval_bm25",
    "term frequency": "retrieval_bm25",

    # --- Search Infrastructure ---------------------------------------------
    "elasticsearch": "search_infra",
    "elastic search": "search_infra",
    "opensearch": "search_infra",
    "solr": "search_infra",
    "lucene": "search_infra",
    "vespa": "search_infra",
    "typesense": "search_infra",

    # --- Embeddings --------------------------------------------------------
    "embeddings": "embeddings",
    "embedding": "embeddings",
    "sentence transformers": "embeddings",
    "sentence-transformers": "embeddings",
    "sbert": "embeddings",
    "word2vec": "embeddings",
    "fasttext": "embeddings",
    "openai embeddings": "embeddings",
    "text embeddings": "embeddings",

    # --- Vector Databases --------------------------------------------------
    "faiss": "vector_db",
    "pinecone": "vector_db",
    "weaviate": "vector_db",
    "qdrant": "vector_db",
    "milvus": "vector_db",
    "chroma": "vector_db",
    "chromadb": "vector_db",
    "pgvector": "vector_db",
    "vector database": "vector_db",
    "vector db": "vector_db",
    "vector store": "vector_db",
    "ann": "vector_db",
    "approximate nearest neighbor": "vector_db",
    "nearest neighbor": "vector_db",
    "hnsw": "vector_db",

    # --- Ranking / Recommendation Systems ----------------------------------
    "ranking": "ranking",
    "learning to rank": "ranking",
    "ltr": "ranking",
    "re-ranking": "ranking",
    "reranking": "ranking",
    "cross-encoder": "ranking",
    "pointwise ranking": "ranking",
    "pairwise ranking": "ranking",
    "listwise ranking": "ranking",
    "recommendation systems": "recsys",
    "recommender systems": "recsys",
    "recommender system": "recsys",
    "recsys": "recsys",
    "recommender": "recsys",
    "collaborative filtering": "recsys",
    "content-based filtering": "recsys",
    "matrix factorization": "recsys",
    "two-tower model": "recsys",
    "candidate generation": "recsys",
    "personalization": "recsys",
    "personalisation": "recsys",

    # --- Ranking Metrics ---------------------------------------------------
    "ndcg": "ranking_metrics",
    "mrr": "ranking_metrics",
    "map": "ranking_metrics",
    "a/b testing": "ranking_metrics",
    "a/b test": "ranking_metrics",
    "offline evaluation": "ranking_metrics",
    "online evaluation": "ranking_metrics",
    "precision@k": "ranking_metrics",
    "recall@k": "ranking_metrics",

    # --- Python ------------------------------------------------------------
    "python": "python",
    "python3": "python",
    "python 3": "python",

    # --- MLOps / Model Serving ---------------------------------------------
    "mlflow": "mlops",
    "kubeflow": "mlops",
    "sagemaker": "mlops",
    "feature store": "mlops",
    "model serving": "mlops",
    "model deployment": "mlops",
    "ray": "mlops",
    "ray serve": "mlops",
    "torchserve": "mlops",
    "triton inference server": "mlops",
    "triton": "mlops",
    "bentoml": "mlops",
    "seldon": "mlops",
    "inference optimization": "mlops",
    "quantization": "mlops",

    # --- Data Infrastructure -----------------------------------------------
    "spark": "data_infra",
    "pyspark": "data_infra",
    "kafka": "data_infra",
    "airflow": "data_infra",
    "dbt": "data_infra",
    "databricks": "data_infra",
    "dask": "data_infra",
    "flink": "data_infra",

    # --- SQL / Databases ---------------------------------------------------
    "sql": "data_sql",
    "postgresql": "data_sql",
    "mysql": "data_sql",
    "bigquery": "data_sql",
    "snowflake": "data_sql",
    "redshift": "data_sql",

    # --- LLM-Glue Frameworks (neutral, but low depth signal) ---------------
    "langchain": "llm_glue",
    "llamaindex": "llm_glue",
    "llama-index": "llm_glue",
    "llama index": "llm_glue",
    "haystack": "llm_glue",
    "autogen": "llm_glue",

    # --- Computer Vision (wrong-domain for this JD) ------------------------
    "computer vision": "cv_domain",
    "opencv": "cv_domain",
    "image classification": "cv_domain",
    "object detection": "cv_domain",
    "image segmentation": "cv_domain",
    "yolo": "cv_domain",
    "cnn": "cv_domain",
    "convolutional neural network": "cv_domain",
    "gans": "cv_domain",
    "generative adversarial network": "cv_domain",

    # --- Speech / Audio (wrong-domain for this JD) -------------------------
    "speech recognition": "speech_domain",
    "asr": "speech_domain",
    "text-to-speech": "speech_domain",
    "tts": "speech_domain",
    "speech synthesis": "speech_domain",
    "audio processing": "speech_domain",
    "whisper": "speech_domain",
}

# ---------------------------------------------------------------------------
# Tier classification (after normalization)
# ---------------------------------------------------------------------------
TIER_A_GROUPS: frozenset[str] = frozenset({
    "retrieval", "ranking", "recsys", "embeddings", "search_infra",
    "vector_db", "nlp", "python", "retrieval_bm25", "ranking_metrics",
})
TIER_B_GROUPS: frozenset[str] = frozenset({
    "machine_learning", "deep_learning", "llm", "llm_finetune",
    "mlops", "data_sql",
})
TIER_C_GROUPS: frozenset[str] = frozenset({
    "data_infra", "llm_glue",
})

# Weights per tier for the tiered skill scorer
TIER_WEIGHTS: dict[str, float] = {"A": 3.0, "B": 1.5, "C": 0.5}


def normalise_skill(name: str) -> str:
    """Return the canonical group key for a raw skill name, or the lowercased name itself."""
    return SKILL_ALIASES.get(name.lower().strip(), name.lower().strip())


def skill_tier(canonical_group: str) -> str | None:
    """Return 'A', 'B', 'C', or None for non-AI/irrelevant skills."""
    if canonical_group in TIER_A_GROUPS:
        return "A"
    if canonical_group in TIER_B_GROUPS:
        return "B"
    if canonical_group in TIER_C_GROUPS:
        return "C"
    return None
