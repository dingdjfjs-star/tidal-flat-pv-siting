# -*- coding: utf-8 -*-
"""定位福鼎沿海城镇（店下/沙埕/秦屿/硖门/白琳），确定光伏研究区 bbox"""
import requests, json, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

QUERY = """[out:json][timeout:60];
node["place"~"^(town|village|hamlet)$"](26.95,120.05,27.30,120.55);
out;"""

MIRRORS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter"]

def main():
    data = None
    for url in MIRRORS:
        for i in range(3):
            try:
                print("try", url, flush=True)
                r = requests.post(url, data={"data": QUERY}, timeout=90,
                                  headers={"User-Agent": "fuding-pv-siting/1.0"})
                r.raise_for_status(); data = r.json(); break
            except Exception as e:
                print("  fail:", str(e)[:100], flush=True); time.sleep(3)
        if data: break
    if not data:
        sys.exit(1)
    for el in data.get("elements", []):
        t = el.get("tags", {})
        nm = t.get("name", "")
        if any(k in nm for k in ["店下", "沙埕", "秦屿", "太姥山", "硖门", "白琳", "龙安", "前岐"]):
            print(f"{nm:10s} lon={el['lon']:.4f} lat={el['lat']:.4f} place={t.get('place')}")
    json.dump(data, open(r"E:\zcode\作业\滩涂光伏选址\_tools\locate_pv.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("saved")

if __name__ == "__main__":
    main()
