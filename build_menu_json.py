"""Convert the two Urban Tandoor Excel files into nested JSON (one document per menu item
+ one document per dish family such as 'biryani')."""
import json, re, math
import pandas as pd

MENU_XLSX = "Urban_Tandoor_Complete_Menu_Catalog.xlsx"
INV_XLSX = "Urban_Tandoor_Two_Branch_Inventory_Dataset.xlsx"

def clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if hasattr(v, "item"):
        v = v.item()
    return v

def rows(df):
    return [{k: clean(v) for k, v in r.items()} for r in df.to_dict("records")]

def yes(v): return str(v).strip().lower() == "yes"

menu = pd.read_excel(MENU_XLSX, sheet_name=None)
inv = pd.read_excel(INV_XLSX, sheet_name=None)

items      = menu["Menu_Items"]
variants   = menu["Variants"]
addons     = menu["Addons"].set_index("Addon_ID")
item_add   = menu["Item_Addons"]
branch_menu= menu["Branch_Menu"]
aliases    = menu["Item_Aliases"]
deals      = menu["Deals"]
deal_items = menu["Deal_Items"]
categories = menu["Categories"].set_index("Category_ID")
recipes    = inv["Recipes"]
ingredients= inv["Ingredients"].set_index("Ingredient_ID")

# dish families: keyword in item name -> family (edit freely)
FAMILIES = ["biryani","pulao","karahi","handi","nihari","haleem","naan","kabab","boti","tikka",
            "burger","sandwich","wrap","fries","pizza","pasta","chow mein","fried rice","soup",
            "paratha","omelette","shake","coffee","ice cream","brownie","juice","tea","chai"]
def family(name):
    n = name.lower()
    for f in FAMILIES:
        if f in n: return f
    return None

docs = []
for it in rows(items):
    mid = it["Menu_ID"]
    d = {
        "_id": mid,
        "name": it["Item_Name"],
        "slug": it["Slug"],
        "category": {"id": it["Category_ID"], "name": it["Category"],
                     "description": clean(categories.loc[it["Category_ID"], "Customer_Description"])},
        "dish_family": family(it["Item_Name"]),
        "description": it["Description"],
        "base_price_pkr": it["Base_Price_PKR"],
        "item_type": it["Item_Type"],
        "status": it["Catalog_Status"],
        "prep_time_min": it["Prep_Time_Min"],
        "dietary_hint": it["Dietary_Hint"],
        "allergen_hint": it["Allergen_Hint"],
        "allergen_verified": it["Allergen_Verified"] == "Verified",
        "service_window": it["Service_Window"],
        "upsell_eligible": yes(it["Upsell_Eligible"]),
        "recipe_linked": yes(it["Recipe_Linked"]),
    }
    d["variants"] = [{"id": v["Variant_ID"], "name": v["Variant_Name"], "price_pkr": v["Price_PKR"],
                      "shown_in_menu": yes(v["Shown_In_Menu"]), "approved_for_orders": yes(v["Approved_For_Orders"])}
                     for v in rows(variants[variants.Menu_ID == mid])]
    d["addons"] = []
    for m in rows(item_add[item_add.Menu_ID == mid]):
        a = addons.loc[m["Addon_ID"]]
        d["addons"].append({"id": m["Addon_ID"], "name": a["Addon_Name"], "price_pkr": int(a["Price_PKR"]),
                            "selectable": yes(m["Selectable"])})
    d["aliases"] = [{"phrase": a["Customer_Phrase"], "language": a["Language"]}
                    for a in rows(aliases[aliases.Menu_ID == mid])]
    d["branches"] = [{"branch_id": b["Branch_ID"], "branch_name": b["Branch_Name"],
                      "enabled": yes(b["Enabled_For_Order"]), "price_pkr": b["Price_PKR"],
                      "snapshot_max_portions": b["Snapshot_Max_Portions"]}
                     for b in rows(branch_menu[branch_menu.Menu_ID == mid])]
    d["recipe"] = []
    for r in rows(recipes[recipes.Menu_ID == mid]):
        ing = ingredients.loc[r["Ingredient_ID"]]
        d["recipe"].append({"ingredient_id": r["Ingredient_ID"], "ingredient": ing["Ingredient_Name"],
                            "qty_per_portion": r["Qty_Per_Portion"], "unit": ing["Unit"]})
    dids = deal_items[deal_items.Menu_ID == mid].Deal_ID.unique().tolist()
    d["in_deals"] = [{"id": x["Deal_ID"], "name": x["Deal_Name"], "price_pkr": x["Deal_Price_PKR"]}
                     for x in rows(deals[deals.Deal_ID.isin(dids)])]

    # ---- text that gets embedded (natural language, incl. Roman-Urdu aliases) ----
    names = ", ".join(a["phrase"] for a in d["aliases"])
    vtxt = "; ".join(f'{v["name"]} Rs {v["price_pkr"]}' for v in d["variants"])
    atxt = ", ".join(f'{a["name"]} (Rs {a["price_pkr"]})' for a in d["addons"]) or "none"
    ingtxt = ", ".join(r["ingredient"] for r in d["recipe"]) or "not listed"
    btxt = ", ".join(b["branch_name"] for b in d["branches"] if b["enabled"]) or "none"
    d["embedding_text"] = (
        f'{d["name"]} - {d["category"]["name"]} category. {d["description"]}. '
        f'{d["dietary_hint"]}. Price Rs {d["base_price_pkr"]}. Sizes: {vtxt}. '
        f'Prep time {d["prep_time_min"]} min. Ingredients: {ingtxt}. Add-ons: {atxt}. '
        f'Available at: {btxt}. Also called: {names}.'
    )
    docs.append(d)

# dish-family docs: everything about "biryani" in one document
groups = []
for fam in sorted({d["dish_family"] for d in docs if d["dish_family"]}):
    members = [d for d in docs if d["dish_family"] == fam]
    g = {"_id": f"FAM-{fam.replace(' ','-')}", "dish_family": fam,
         "item_ids": [m["_id"] for m in members],
         "items": [{k: m[k] for k in ("_id","name","base_price_pkr","variants","addons","aliases","branches","recipe","prep_time_min","dietary_hint")} for m in members]}
    g["embedding_text"] = f"{fam.title()} dishes at Urban Tandoor. " + " ".join(m["embedding_text"] for m in members)
    groups.append(g)

json.dump(docs, open("menu_items.json","w",encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
json.dump(groups, open("dish_groups.json","w",encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
print(len(docs), "items,", len(groups), "dish families")
