from groq import Groq
from config.settings import settings
from config.prompts import RAG_PROMPT
from models.rag import RAGResponse
from services.search import SearchService


class RAGService:
    def __init__(self, search_service: SearchService):
        self.search_service = search_service
        self.client = Groq(api_key=settings.groq_api_key)

    def generate_answer(self, query: str, limit: int = 3):
        search_results = self.search_service.search(query, limit)

        context = "\n\n".join(result.text for result in search_results.results)

        prompt = RAG_PROMPT.format(context=context, query=query)

        response = self.client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        metadata = [
            {
                **result.metadata,
                "score": result.score,
            }
            for result in search_results.results
        ]

        return RAGResponse(
            query=query, answer=response.choices[0].message.content, metadata=metadata
        )
