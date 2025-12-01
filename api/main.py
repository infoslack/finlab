from fastapi import FastAPI
from routers import rag, search, agent

app = FastAPI(title="Financial Search API")

app.include_router(search.router)
app.include_router(rag.router)
app.include_router(agent.router)


@app.get("/")
def root():
    return {"status": "online"}
