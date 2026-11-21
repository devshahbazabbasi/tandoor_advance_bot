import requests

BASE = "http://127.0.0.1:8000"
results = []

def check(name, condition):
    results.append(condition)
    print(("PASS  " if condition else "FAIL  ") + name)

# 0. Health: remember the starting counts
before = requests.get(f"{BASE}/").json()
check("Health endpoint works", "menu_items" in before)

# 1. CREATE
new_item = {
    "name": "Test Biryani",
    "category_name": "Pakistani",
    "description": "Temporary item for testing",
    "base_price_pkr": 500,
    "dish_family": "biryani",
    "variants": [{"name": "Regular", "price_pkr": 500}],
    "addons": [{"name": "Extra Raita", "price_pkr": 60}],
    "aliases": [{"phrase": "test biryani", "language": "English"}],
}
r = requests.post(f"{BASE}/items", json=new_item)
check("CREATE returns 201", r.status_code == 201)
mid = r.json().get("_id")
print("      new id:", mid)

# 2. READ one item
r = requests.get(f"{BASE}/items/{mid}")
check("READ item by id", r.status_code == 200 and r.json()["base_price_pkr"] == 500)

# 3. READ family contains the new item
r = requests.get(f"{BASE}/family/biryani")
check("READ family includes new item", mid in r.json()["item_ids"])

# 4. READ list by category
r = requests.get(f"{BASE}/items", params={"category": "Pakistani", "limit": 200})
check("READ list finds new item", any(i["_id"] == mid for i in r.json()))

# 5. SEARCH finds it
r = requests.get(f"{BASE}/search", params={"q": "test biryani", "k": 5})
check("SEARCH finds new item", any(i["_id"] == mid for i in r.json()["items"]))

# 6. UPDATE price and description
r = requests.put(f"{BASE}/items/{mid}", json={"base_price_pkr": 700, "description": "Updated"})
check("UPDATE returns 200", r.status_code == 200)
r = requests.get(f"{BASE}/items/{mid}")
check("UPDATE changed the price", r.json()["base_price_pkr"] == 700)
fam = requests.get(f"{BASE}/family/biryani").json()
price_in_family = [i["base_price_pkr"] for i in fam["items"] if i["_id"] == mid]
check("UPDATE synced to family", price_in_family == [700])

# 7. UPDATE on a missing item
r = requests.put(f"{BASE}/items/MENU99999", json={"base_price_pkr": 1})
check("UPDATE missing item gives 404", r.status_code == 404)

# 8. DELETE
r = requests.delete(f"{BASE}/items/{mid}")
check("DELETE returns 200", r.status_code == 200)
r = requests.get(f"{BASE}/items/{mid}")
check("DELETE removed the item (404)", r.status_code == 404)
fam = requests.get(f"{BASE}/family/biryani").json()
check("DELETE removed item from family", mid not in fam["item_ids"])

# 9. DELETE on a missing item
r = requests.delete(f"{BASE}/items/{mid}")
check("DELETE twice gives 404", r.status_code == 404)

# 10. Counts back to normal
after = requests.get(f"{BASE}/").json()
check("Counts back to the starting numbers", after == before)

print(f"\n{sum(results)} of {len(results)} checks passed")