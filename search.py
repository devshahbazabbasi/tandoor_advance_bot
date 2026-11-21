import os, sys
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?directConnection=true")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
db = MongoClient(MONGO_URI)["urban_tandoor"]

def search(q, coll="menu_items", k=3):
    qv = model.encode(q, normalize_embeddings=True).tolist()
    return list(db[coll].aggregate([
        {"$vectorSearch": {"index": f"{coll}_vector_index", "path": "embedding",
                           "queryVector": qv, "numCandidates": 100, "limit": k}},
        {"$project": {"embedding": 0, "embedding_text": 0, "score": {"$meta": "vectorSearchScore"}}},
    ]))

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "biryani"
    # whole family in one go:
    for r in search(q, "dish_groups", 1):
        print("FAMILY:", r["dish_family"], "->", [i["name"] + f' Rs{i["base_price_pkr"]}' for i in r["items"]])
    # individual best matches:
    for r in search(q, "menu_items", 3):
        print(round(r["score"], 3), r["name"], r["base_price_pkr"])
