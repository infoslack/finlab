import asyncio

from config.prompts import (
    AGGREGATION_PROMPT,
    FUNDAMENTAL_PROMPT,
    FUNDAMENTAL_QUERIES,
    MOMENTUM_PROMPT,
    MOMENTUM_QUERIES,
    SENTIMENT_PROMPT,
    SENTIMENT_QUERY_TEMPLATE,
)
from config.settings import settings
from groq import AsyncGroq
from models.agent import AgentResponse

from services.search import SearchService


class AgentService:
    def __init__(self, search_service: SearchService):
        self.search_service = search_service
        self.client = AsyncGroq(api_key=settings.groq_api_key)

    def _run_queries(self, queries: list[str], limit: int, filter: dict = None):
        all_results = []
        for query in queries:
            search_results = self.search_service.search(query, limit, filter)
            all_results.extend([result.text for result in search_results.results])
        return "\n\n".join(all_results)

    async def _generate_completion(self, prompt: str):
        response = await self.client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content

    async def _analyze_fundamental(self, ticker: str, limit: int):
        filter = {"ticker": ticker, "form_type": "10-K"}
        context = self._run_queries(FUNDAMENTAL_QUERIES, limit, filter)
        prompt = FUNDAMENTAL_PROMPT.format(context=context)
        return await self._generate_completion(prompt)

    async def _analyze_momentum(self, ticker: str, limit: int):
        filter = {"ticker": ticker, "form_type": "10-Q"}
        context = self._run_queries(MOMENTUM_QUERIES, limit, filter)
        prompt = MOMENTUM_PROMPT.format(context=context)
        return await self._generate_completion(prompt)

    async def _analyze_sentiment(self, ticker: str, limit: int):
        filter = {"ticker": ticker, "source": "yahoo_finance"}
        query = SENTIMENT_QUERY_TEMPLATE.format(ticker=ticker)
        results = self.search_service.search(query, limit, filter)
        context = "\n\n".join([result.text for result in results.results])
        prompt = SENTIMENT_PROMPT.format(context=context)
        return await self._generate_completion(prompt)

    async def analyze(self, ticker: str, limit: int = 3):
        fundamental_task = self._analyze_fundamental(ticker, limit)
        momentum_task = self._analyze_momentum(ticker, limit)
        sentiment_task = self._analyze_sentiment(ticker, limit)

        (
            fundamental_analysis,
            momentum_analysis,
            sentiment_analysis,
        ) = await asyncio.gather(fundamental_task, momentum_task, sentiment_task)

        aggregation_prompt = AGGREGATION_PROMPT.format(
            fundamental=fundamental_analysis,
            momentum=momentum_analysis,
            sentiment=sentiment_analysis,
        )
        final_recomendation = await self._generate_completion(aggregation_prompt)

        return AgentResponse(
            ticker=ticker,
            fundamental_analysis=fundamental_analysis,
            momentum_analysis=momentum_analysis,
            sentiment_analysis=sentiment_analysis,
            final_recommendation=final_recomendation,
        )
