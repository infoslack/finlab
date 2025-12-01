from pydantic import BaseModel


class AgentRequest(BaseModel):
    ticker: str
    limit: int = 3


class AgentResponse(BaseModel):
    ticker: str
    fundamental_analysis: str
    momentum_analysis: str
    sentiment_analysis: str
    final_recommendation: str
