# -*- coding: utf-8 -*-
"""光伏研究区 OSM 全要素抓取：道路/水系/水面/居民点/建筑/地名点/保护地/电力设施"""
import requests, json, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

S, W, N, E = 27.00, 120.16, 27.22, 120.48
BB = f"{S},{W},{N},{E}"
QUERY = f"""[out:json][timeout:300];
(
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|service|track)$"]({BB});
  way["waterway"~"^(river|stream|canal|ditch)$"]({BB});
  way["natural"="water"]({BB});
  relation["natural"="water"]({BB});
  way["landuse"="residential"]({BB});
  relation["landuse"="residential"]({BB});
  way["building"]({BB});
  node["place"~"^(town|village|hamlet|isolated_dwelling|neighbourhood|farm)$"]({BB});
  way["boundary"="protected_area"]({BB});
  relation["boundary"="protected_area"]({BB});
  way["leisure"="nature_reserve"]({BB});
  relation["leisure"="nature_reserve"]({BB});
  way["boundary"="national_park"]({BB});
  relation["boundary"="national_park"]({BB});
  way["power"~"^(line|minor_line)$"]({BB});
  node["power"="substation"]({BB});
  way["power"="plant"]({BB});
);
out geom;"""

MIRRORS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
           "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]

def fetch(q):
    for url in MIRRORS:
        for i in range(3):
            try:
                print("try", url, f"#{i+1}", flush=True)
                r = requests.post(url, data={"data": q}, timeout=420,
                                  headers={"User-Agent": "fuding-pv-siting/1.0"})
                r.raise_for_status(); return r.json()
            except Exception as e:
                print("  fail:", str(e)[:110], flush=True); time.sleep(5)
    return None

data = fetch(QUERY)
if data is None:
    print("ALL FAILED"); sys.exit(1)
from collections import Counter
els = data.get("elements", [])
cnt = Counter()
for e in els:
    t = e.get("tags", {})
    for k in ("highway", "waterway", "building", "landuse", "place", "power"):
        if k in t:
            cnt[f"{k}:{t[k]}"] += 1
    if t.get("natural") == "water": cnt["natural:water"] += 1
    if "boundary" in t or t.get("leisure") == "nature_reserve": cnt["protected"] += 1
print(f"elements={len(els)}")
for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
    print(f"  {k:36s} {v}")
json.dump(data, open(r"E:\zcode\作业\滩涂光伏选址\_tools\osm_raw_pv.json", "w", encoding="utf-8"), ensure_ascii=False)
print("saved osm_raw_pv.json")
