"""Embed menu docs (local multilingual model, free) and load them into MongoDB Atlas / Atlas Local."""
import json, os
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from sentence_transformers import SentenceTransformer

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?directConnection=true")
DB = "urban_tandoor"
MODEL = "paraphrase-multilingual-MiniLM-L12-v2"   # 384 dims, understands Roman Urdu reasonably

model = SentenceTransformer(MODEL)
db = MongoClient(MONGO_URI)[DB]

for fname, coll in [("menu_items.json", "menu_items"), ("dish_groups.json", "dish_groups")]:
    docs = json.load(open(fname, encoding="utf-8"))
    vecs = model.encode([d["embedding_text"] for d in docs], normalize_embeddings=True)
    for d, v in zip(docs, vecs):
        d["embedding"] = v.tolist()
    db[coll].drop()
    db[coll].insert_many(docs)
    print(coll, "inserted", len(docs))
    db[coll].create_search_index(SearchIndexModel(
        name=f"{coll}_vector_index", type="vectorSearch",
        definition={"fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine"},
            {"type": "filter", "path": "category.name"},
            {"type": "filter", "path": "dish_family"},
        ]}))
print("Done. Wait ~1 minute for the vector indexes to become READY.")
