import os

from dotenv import load_dotenv
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

load_dotenv()

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
COLBERT_MODEL = "colbert-ir/colbertv2.0"
COLLECTION_NAME = "financial"

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

dense_model = TextEmbedding(DENSE_MODEL)
sparse_model = SparseTextEmbedding(SPARSE_MODEL)
colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)

query_text = "what are the main financial risks?"
query_dense = list(dense_model.query_embed([query_text]))[0].tolist()
query_sparse = list(sparse_model.query_embed([query_text]))[0].as_object()
query_colbert = list(colbert_model.query_embed([query_text]))[0].tolist()

results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        {
            "prefetch": [
                {"query": query_dense, "using": "dense", "limit": 10},
                {"query": query_sparse, "using": "sparse", "limit": 10},
            ],
            "query": models.FusionQuery(fusion=models.Fusion.RRF),
            "limit": 20,
        }
    ],
    query=query_colbert,
    using="colbert",
    limit=3,
)

max_score = max(result.score for result in results.points)

for r in results.points:
    normalized_score = r.score / max_score
    print(f"Score: {normalized_score}")
    print(f"Texto: {r.payload['text'][:100]}...")
    print("-" * 80)
