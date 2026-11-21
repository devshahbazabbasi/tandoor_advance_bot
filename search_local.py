import sys
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
db = MongoClient("mongodb://localhost:27017")["urban_tandoor"]

def search(q, coll, k=3):
    qv = model.encode(q, normalize_embeddings=True)
    docs = list(db[coll].find())
    scores = np.array([d["embedding"] for d in docs]) @ qv
    best = np.argsort(-scores)[:k]
    return [(float(scores[i]), docs[i]) for i in best]

q = " ".join(sys.argv[1:]) or "biryani"
s, fam = search(q, "dish_groups", 1)[0]
print("FAMILY:", fam["dish_family"], [i["name"] for i in fam["items"]])
for s, d in search(q, "menu_items", 3):
    print(round(s, 3), d["name"], d["base_price_pkr"])
