from qdrant_client import QdrantClient, models
from models.search import SearchResponse, SearchResult
from services.embeddings import EmbeddingService


class SearchService:
    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name: str):
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name
        self.embedding_service = EmbeddingService()

    def search(self, query: str, limit: int = 3):
        query_dense, query_sparse, query_colbert = self.embedding_service.embed_query(
            query
        )

        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            prefetch=[
                {
                    "prefetch": [
                        {"query": query_dense, "using": "dense", "limit": 20},
                        {"query": query_sparse, "using": "sparse", "limit": 20},
                    ],
                    "query": models.FusionQuery(fusion=models.Fusion.RRF),
                    "limit": 15,
                }
            ],
            query=query_colbert,
            using="colbert",
            limit=limit,
        )

        max_score = max(result.score for result in results.points)

        search_results = [
            SearchResult(
                score=result.score / max_score,
                text=result.payload["text"],
                metadata=result.payload["metadata"],
            )
            for result in results.points
        ]

        return SearchResponse(results=search_results)
