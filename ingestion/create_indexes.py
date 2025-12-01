import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

fields_to_index = [
    "metadata.ticker",
    "metadata.form_type",
    "metadata.source",
]

for field_name in fields_to_index:
    qdrant.create_payload_index(
        collection_name="financial",
        field_name=field_name,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    print(f"índice criado para {field_name}")
