from fastapi import FastAPI
from pydantic import BaseModel

from app.rag_pipeline import pipeline

app = FastAPI()

class TopicRequest(BaseModel):
    topic: str

class QueryRequest(BaseModel):
    query: str

@app.post('/ingest')
def ingest(req: TopicRequest):

    result = pipeline.ingest_topic(req.topic)

    return {
        'message': result
    }

@app.post('/query')
def query(req: QueryRequest):

    return pipeline.query(req.query)