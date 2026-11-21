import os
import json
import traceback
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()          # reads GEMINI_API_KEY from the .env file
except ImportError:
    pass

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

app = FastAPI(title="Urban Tandoor Menu API")
db = MongoClient("mongodb://localhost:27017")["urban_tandoor"]
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
NO_EMB = {"embedding": 0, "embedding_text": 0}
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ---------- request models ----------
class Variant(BaseModel):
    name: str
    price_pkr: int

class Addon(BaseModel):
    name: str
    price_pkr: int

class Alias(BaseModel):
    phrase: str
    language: str = "English"

class ItemCreate(BaseModel):
    name: str
    category_name: str
    description: str = ""
    base_price_pkr: int
    dish_family: Optional[str] = None
    prep_time_min: int = 15
    dietary_hint: str = "Non-Vegetarian"
    variants: list[Variant] = []
    addons: list[Addon] = []
    aliases: list[Alias] = []

class ItemUpdate(BaseModel):
    name: Optional[str] = None
    category_name: Optional[str] = None
    description: Optional[str] = None
    base_price_pkr: Optional[int] = None
    dish_family: Optional[str] = None
    prep_time_min: Optional[int] = None
    dietary_hint: Optional[str] = None
    status: Optional[str] = None
    variants: Optional[list[Variant]] = None
    addons: Optional[list[Addon]] = None
    aliases: Optional[list[Alias]] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]


# ---------- helpers ----------
def embed(text):
    return model.encode(text, normalize_embeddings=True).tolist()

def make_text(d):
    vtxt = "; ".join(f'{v["name"]} Rs {v["price_pkr"]}' for v in d.get("variants", []))
    atxt = ", ".join(f'{a["name"]} (Rs {a["price_pkr"]})' for a in d.get("addons", [])) or "none"
    itxt = ", ".join(r["ingredient"] for r in d.get("recipe", [])) or "not listed"
    btxt = ", ".join(b["branch_name"] for b in d.get("branches", []) if b.get("enabled")) or "none"
    names = ", ".join(a["phrase"] for a in d.get("aliases", []))
    return (f'{d["name"]} - {d["category"]["name"]} category. {d.get("description", "")}. '
            f'{d.get("dietary_hint", "")}. Price Rs {d["base_price_pkr"]}. Sizes: {vtxt}. '
            f'Prep time {d.get("prep_time_min", "")} min. Ingredients: {itxt}. '
            f'Add-ons: {atxt}. Available at: {btxt}. Also called: {names}.')

def next_id():
    nums = [int(x["_id"][4:]) for x in db.menu_items.find({}, {"_id": 1})
            if x["_id"].startswith("MENU")]
    return f"MENU{(max(nums) if nums else 0) + 1:03d}"

def category_doc(name):
    existing = db.menu_items.find_one({"category.name": name}, {"category": 1})
    if existing:
        return existing["category"]
    return {"id": "CAT-" + name.upper().replace(" ", "-"), "name": name, "description": None}

def build_variants(mid, variants):
    return [{"id": f"VAR-{mid}-{i}", "name": v["name"], "price_pkr": v["price_pkr"],
             "shown_in_menu": True, "approved_for_orders": True}
            for i, v in enumerate(variants, 1)]

def build_addons(mid, addons):
    return [{"id": f"ADD-{mid}-{i}", "name": a["name"], "price_pkr": a["price_pkr"],
             "selectable": True} for i, a in enumerate(addons, 1)]

def rebuild_family(fam):
    """Keep the dish_groups document in sync with menu_items."""
    if not fam:
        return
    members = list(db.menu_items.find({"dish_family": fam}))
    gid = "FAM-" + fam.replace(" ", "-")
    if not members:
        db.dish_groups.delete_one({"_id": gid})
        return
    keys = ("_id", "name", "base_price_pkr", "variants", "addons", "aliases",
            "branches", "recipe", "prep_time_min", "dietary_hint")
    text = f"{fam.title()} dishes at Urban Tandoor. " + " ".join(m["embedding_text"] for m in members)
    doc = {"_id": gid, "dish_family": fam, "item_ids": [m["_id"] for m in members],
           "items": [{k: m.get(k) for k in keys} for m in members],
           "embedding_text": text, "embedding": embed(text)}
    db.dish_groups.replace_one({"_id": gid}, doc, upsert=True)

def vsearch(q, coll, k):
    qv = model.encode(q, normalize_embeddings=True)
    docs = list(db[coll].find())
    scores = np.array([d["embedding"] for d in docs]) @ qv
    out = []
    for i in np.argsort(-scores)[:k]:
        d = docs[i]
        d.pop("embedding", None)
        d.pop("embedding_text", None)
        d["score"] = round(float(scores[i]), 3)
        out.append(d)
    return out


# ---------- READ ----------
@app.get("/")
def health():
    return {"menu_items": db.menu_items.count_documents({}),
            "dish_groups": db.dish_groups.count_documents({})}

@app.get("/items")
def list_items(category: Optional[str] = None, limit: int = 50):
    q = {"category.name": category} if category else {}
    return list(db.menu_items.find(q, NO_EMB).limit(limit))

@app.get("/items/{menu_id}")
def get_item(menu_id: str):
    item = db.menu_items.find_one({"_id": menu_id.upper()}, NO_EMB)
    if not item:
        raise HTTPException(404, "Item not found")
    return item

@app.get("/family/{name}")
def get_family(name: str):
    fam = db.dish_groups.find_one({"dish_family": name.lower()}, NO_EMB)
    if not fam:
        raise HTTPException(404, "Family not found")
    return fam

@app.get("/search")
def search(q: str, k: int = 3):
    return {"family": vsearch(q, "dish_groups", 1)[0],
            "items": vsearch(q, "menu_items", k)}


# ---------- CREATE ----------
@app.post("/items", status_code=201)
def create_item(body: ItemCreate):
    mid = next_id()
    fam = body.dish_family.lower() if body.dish_family else None
    d = {
        "_id": mid, "name": body.name, "slug": "utg-item-" + mid[4:],
        "category": category_doc(body.category_name), "dish_family": fam,
        "description": body.description, "base_price_pkr": body.base_price_pkr,
        "item_type": "Food", "status": "Active", "prep_time_min": body.prep_time_min,
        "dietary_hint": body.dietary_hint, "allergen_hint": "Not verified",
        "allergen_verified": False, "service_window": "All day",
        "upsell_eligible": True, "recipe_linked": False,
        "variants": build_variants(mid, [v.model_dump() for v in body.variants]),
        "addons": build_addons(mid, [a.model_dump() for a in body.addons]),
        "aliases": [a.model_dump() for a in body.aliases],
        "branches": [], "recipe": [], "in_deals": [],
    }
    d["embedding_text"] = make_text(d)
    d["embedding"] = embed(d["embedding_text"])
    db.menu_items.insert_one(d)
    rebuild_family(fam)
    return db.menu_items.find_one({"_id": mid}, NO_EMB)


# ---------- UPDATE (only the fields you send are changed) ----------
@app.put("/items/{menu_id}")
def update_item(menu_id: str, body: ItemUpdate):
    mid = menu_id.upper()
    d = db.menu_items.find_one({"_id": mid})
    if not d:
        raise HTTPException(404, "Item not found")
    old_fam = d.get("dish_family")
    changes = body.model_dump(exclude_unset=True)

    if "category_name" in changes:
        d["category"] = category_doc(changes.pop("category_name"))
    if "variants" in changes:
        d["variants"] = build_variants(mid, changes.pop("variants"))
    if "addons" in changes:
        d["addons"] = build_addons(mid, changes.pop("addons"))
    if "dish_family" in changes:
        f = changes.pop("dish_family")
        d["dish_family"] = f.lower() if f else None
    d.update(changes)

    d["embedding_text"] = make_text(d)
    d["embedding"] = embed(d["embedding_text"])
    db.menu_items.replace_one({"_id": mid}, d)
    rebuild_family(old_fam)
    rebuild_family(d.get("dish_family"))
    return db.menu_items.find_one({"_id": mid}, NO_EMB)


# ---------- DELETE ----------
@app.delete("/items/{menu_id}")
def delete_item(menu_id: str):
    mid = menu_id.upper()
    d = db.menu_items.find_one({"_id": mid}, {"dish_family": 1})
    if not d:
        raise HTTPException(404, "Item not found")
    db.menu_items.delete_one({"_id": mid})
    rebuild_family(d.get("dish_family"))
    return {"deleted": mid}


# ================= CHATBOT (Gemini + tools that read the database) =================
CHAT_SYSTEM = """You are the friendly menu assistant for Urban Tandoor & Grill (branches: PECHS and Gulshan-e-Iqbal, Karachi).
- ALWAYS use the tools to look up the menu before answering a menu question. Never answer prices, dishes or ingredients from memory.
- If the tools do not return the answer, say you are not sure and suggest asking the staff. Never invent items.
- Prices are in PKR (Rs).
- Reply in the customer's language (English, Urdu or Roman Urdu). Keep answers short and clear.
- Allergen info is NOT verified: never say a dish is safe for an allergy; tell them to confirm with staff.
- Stock numbers are an old snapshot: do not promise availability.
- Ordering is not available in this chat yet."""


def compact(d):
    """Short version of a dish, so the LLM gets only what it needs."""
    return {
        "id": d.get("_id"),
        "name": d.get("name"),
        "category": (d.get("category") or {}).get("name"),
        "description": d.get("description"),
        "base_price_pkr": d.get("base_price_pkr"),
        "sizes": [f'{v["name"]} Rs {v["price_pkr"]}' for v in d.get("variants", [])],
        "addons": [f'{a["name"]} Rs {a["price_pkr"]}' for a in d.get("addons", [])],
        "ingredients": [r["ingredient"] for r in d.get("recipe", [])],
        "available_at": [b["branch_name"] for b in d.get("branches", []) if b.get("enabled")],
        "deals": [f'{x["name"]} Rs {x["price_pkr"]}' for x in d.get("in_deals", [])],
        "prep_time_min": d.get("prep_time_min"),
        "dietary_hint": d.get("dietary_hint"),
    }


@app.post("/chat")
def chat(body: ChatRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(500, "GEMINI_API_KEY is not set on the server")

    tools_used = []

    # Tools the model can call. Gemini reads the names, type hints and docstrings.
    def search_menu(query: str, k: int = 5) -> list:
        """Search the menu by meaning. Use for any dish, price, ingredient or craving question (English, Urdu or Roman Urdu)."""
        tools_used.append({"tool": "search_menu", "input": {"query": query, "k": k}})
        return [compact(d) for d in vsearch(query, "menu_items", int(k))]

    def get_dish_family(name: str) -> dict:
        """Get ALL dishes of one family in one go, e.g. biryani, karahi, pizza, burger, kabab, shake, pasta, naan."""
        tools_used.append({"tool": "get_dish_family", "input": {"name": name}})
        fam = db.dish_groups.find_one({"dish_family": name.lower()}, NO_EMB)
        if not fam:
            return {"error": "No such dish family. Try search_menu instead."}
        return {"family": fam["dish_family"], "items": [compact(i) for i in fam["items"]]}

    def get_menu_item(menu_id: str) -> dict:
        """Get full details of one dish by its id, e.g. MENU001."""
        tools_used.append({"tool": "get_menu_item", "input": {"menu_id": menu_id}})
        d = db.menu_items.find_one({"_id": menu_id.upper()}, NO_EMB)
        return compact(d) if d else {"error": "Item not found"}

    def list_category(category: str) -> list:
        """List every dish in a category. Categories: Pakistani, BBQ, Fast Food, Pizza & Italian, Chinese & Continental, Breakfast, Desserts, Beverages."""
        tools_used.append({"tool": "list_category", "input": {"category": category}})
        items = db.menu_items.find({"category.name": category}, NO_EMB).limit(60)
        return [{"id": i["_id"], "name": i["name"], "price_pkr": i["base_price_pkr"]} for i in items]

    # Chat history -> Gemini format (the assistant is called "model" in Gemini)
    msgs = list(body.messages)[-8:]
    while msgs and msgs[0].role != "user":
        msgs.pop(0)
    if not msgs:
        raise HTTPException(400, "Send at least one user message")
    contents = [types.Content(role="user" if m.role == "user" else "model",
                              parts=[types.Part(text=m.content)]) for m in msgs]

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=CHAT_SYSTEM,
                tools=[search_menu, get_dish_family, get_menu_item, list_category],
                max_output_tokens=800,
            ),
        )
        answer = resp.text or "Sorry, I could not find an answer. Please ask the staff."
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Gemini error: {e}")

    return {"answer": answer, "tools_used": tools_used}