# -*- coding: utf-8 -*-
"""拆分抓取：轻查询（核心要素）优先保证成功；建筑重查询单独重试。各自落盘。"""
import requests, json, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

S, W, N, E = 27.00, 120.16, 27.22, 120.48
BB = f"{S},{W},{N},{E}"
CORE = f"""[out:json][timeout:180];
(
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|service|track)$"]({BB});
  way["waterway"~"^(river|stream|canal|ditch)$"]({BB});
  way["natural"="water"]({BB});
  relation["natural"="water"]({BB});
  way["landuse"="residential"]({BB});
  relation["landuse"="residential"]({BB});
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
BLDG = f"""[out:json][timeout:180];
(
  way["building"]({BB});
);
out geom;"""

MIRRORS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
           "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]

def fetch(q, tag, tries=4):
    for url in MIRRORS:
        for i in range(tries):
            try:
                print(f"[{tag}] try {url} #{i+1}", flush=True)
                r = requests.post(url, data={"data": q}, timeout=300,
                                  headers={"User-Agent": "fuding-pv-siting/1.0"})
                r.raise_for_status()
                d = r.json()
                print(f"[{tag}] OK, {len(d.get('elements', []))} elements", flush=True)
                return d
            except Exception as e:
                print(f"[{tag}] fail: {str(e)[:100]}", flush=True)
                time.sleep(6)
    return None

core = fetch(CORE, "core")
bldg = fetch(BLDG, "bldg", tries=2)

out = {"elements": []}
if core:
    out["elements"] += core["elements"]
if bldg:
    out["elements"] += bldg["elements"]
json.dump(out, open(r"E:\zcode\作业\滩涂光伏选址\_tools\osm_raw_pv.json", "w", encoding="utf-8"), ensure_ascii=False)
print(f"MERGED {len(out['elements'])} elements -> osm_raw_pv.json")
from collections import Counter
cnt = Counter()
for e in out["elements"]:
    t = e.get("tags", {})
    for k in ("highway", "waterway", "building", "landuse", "place", "power"):
        if k in t:
            cnt[f"{k}:{t[k]}"] += 1
    if t.get("natural") == "water": cnt["natural:water"] += 1
    if "boundary" in t or t.get("leisure") == "nature_reserve": cnt["protected"] += 1
for k, v in sorted(cnt.items(), key=lambda x: -x[1])[:25]:
    print(f"  {k:36s} {v}")
print("DONE")
