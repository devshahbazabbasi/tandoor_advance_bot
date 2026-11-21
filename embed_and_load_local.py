import json
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
db = MongoClient("mongodb://localhost:27017")["urban_tandoor"]

for fname, coll in [("menu_items.json", "menu_items"), ("dish_groups.json", "dish_groups")]:
    docs = json.load(open(fname, encoding="utf-8"))
    vecs = model.encode([d["embedding_text"] for d in docs], normalize_embeddings=True)
    for d, v in zip(docs, vecs):
        d["embedding"] = v.tolist()
    db[coll].drop()
    db[coll].insert_many(docs)
    print(coll, "inserted", len(docs))
