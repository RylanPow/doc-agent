import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
import datetime

#run with: uv run uvicorn main:app     , file is main,py, uv function is app

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
#the benefit of making this an inngest function is observability
async def rag_ingest_pdf(ctx: inngest.Context):
    return {"hello" : "world"}

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf])