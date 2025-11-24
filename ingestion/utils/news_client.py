from typing import Dict, List

import trafilatura
import yfinance as yf


class NewsClient:
    def fetch_news(self, ticker: str, max_stories: int = 10) -> List[Dict[str, any]]:
        data = yf.Ticker(ticker)
        news = data.news

        news_data = []

        for item in news[:max_stories]:
            content = item.get("content", {})
            content_type = content.get("contentType")

            if content_type != "STORY":
                continue

            canonical_url = content.get("canonicalUrl", {})
            title = content.get("title")
            date = content.get("pubDate")
            url = canonical_url.get("url")

            if "finance.yahoo.com" not in url:
                continue

            downloaded = trafilatura.fetch_url(url)
            text_content = trafilatura.extract(downloaded)

            if text_content:
                metadata = {
                    "ticker": ticker,
                    "title": title,
                    "url": url,
                    "date": date,
                    "source": "yahoo_finance",
                }
                news_data.append({"text": text_content, "metadata": metadata})

        return news_data
