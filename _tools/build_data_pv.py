# -*- coding: utf-8 -*-
"""
滩涂光伏选址 · 数据加工：OSM raw json + 已有 DEM/WorldCover 瓦片 → GeoScene 可加载图层
输出 数据/：dem.tif worldcover.tif roads/streams/waterbody/settlements/buildings/places/
            protected/powerline(线)/substation(点)/plant(面)/study_area
"""
import json, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import LineString, Polygon, Point, box, mapping
from shapely.ops import polygonize, unary_union
import geopandas as gpd

BASE = r"E:\zcode\作业\滩涂光伏选址"
TEA = r"E:\zcode\作业\白茶园选址"          # 复用已下载瓦片
OUT = os.path.join(BASE, "数据")
os.makedirs(OUT, exist_ok=True)

S, W, N, E = 27.00, 120.16, 27.22, 120.48
BBOX = box(W, S, E, N)
CRS = "EPSG:4326"

def log(*a): print(*a, flush=True)

def way_coords(geom):
    return [(g["lon"], g["lat"]) for g in geom]

def assemble_relation(rel):
    outer, inner = [], []
    for m in rel.get("members", []):
        g = m.get("geometry")
        if not g or len(g) < 2:
            continue
        line = LineString(way_coords(g))
        (inner if m.get("role") == "inner" else outer).append(line)
    if not outer:
        return None
    polys = list(polygonize(outer))
    if not polys:
        return None
    poly = unary_union(polys)
    if inner:
        h = unary_union(list(polygonize(inner)))
        if not h.is_empty:
            poly = poly.difference(h)
    return poly

def way_polygon(coords):
    if len(coords) >= 4 and coords[0] == coords[-1]:
        return Polygon(coords)
    return None

log("== 1. 解析 OSM ==")
data = json.load(open(os.path.join(BASE, "_tools", "osm_raw_pv.json"), encoding="utf-8"))
els = data.get("elements", [])
log("elements:", len(els))

roads, streams, water_polys, settle_polys, buildings, places, protected, powerlines, substations, plants = \
    [], [], [], [], [], [], [], [], [], []

for el in els:
    tags = el.get("tags", {})
    t = el.get("type")
    name = tags.get("name", "")
    if t == "node":
        if "place" in tags:
            places.append({"osm_id": el["id"], "name": name, "ptype": tags["place"],
                           "geometry": Point(el["lon"], el["lat"])})
        elif tags.get("power") == "substation":
            substations.append({"osm_id": el["id"], "name": name,
                                "kv": tags.get("voltage", ""), "geometry": Point(el["lon"], el["lat"])})
    elif t == "way":
        g = el.get("geometry")
        if not g or len(g) < 2:
            continue
        coords = way_coords(g)
        if "highway" in tags:
            roads.append({"osm_id": el["id"], "name": name, "cls": tags["highway"], "geometry": LineString(coords)})
        elif "waterway" in tags:
            streams.append({"osm_id": el["id"], "name": name, "wtype": tags["waterway"], "geometry": LineString(coords)})
        elif tags.get("natural") == "water":
            p = way_polygon(coords)
            if p is not None:
                water_polys.append({"osm_id": el["id"], "name": name, "src": "way", "geometry": p})
        elif tags.get("landuse") == "residential":
            p = way_polygon(coords)
            if p is not None:
                settle_polys.append({"osm_id": el["id"], "name": name, "src": "way", "geometry": p})
        elif "building" in tags:
            p = way_polygon(coords)
            if p is not None:
                buildings.append({"osm_id": el["id"], "name": "", "btype": tags["building"], "geometry": p})
        elif "boundary" in tags or tags.get("leisure") == "nature_reserve":
            p = way_polygon(coords)
            if p is not None:
                protected.append({"osm_id": el["id"], "name": name,
                                  "pcls": tags.get("boundary") or tags.get("leisure"), "geometry": p})
        elif tags.get("power") in ("line", "minor_line"):
            powerlines.append({"osm_id": el["id"], "name": name, "kv": tags.get("voltage", ""),
                               "pline": tags.get("power"), "geometry": LineString(coords)})
        elif tags.get("power") == "plant":
            p = way_polygon(coords)
            if p is not None:
                plants.append({"osm_id": el["id"], "name": name, "geometry": p})
    elif t == "relation":
        if tags.get("natural") == "water" or tags.get("landuse") == "residential":
            poly = assemble_relation(el)
            if poly is not None and not poly.is_empty:
                rec = {"osm_id": el["id"], "name": name,
                       "src": "water" if tags.get("natural") == "water" else "residential", "geometry": poly}
                (water_polys if rec["src"] == "water" else settle_polys).append(rec)
        elif "boundary" in tags or tags.get("leisure") == "nature_reserve":
            poly = assemble_relation(el)
            if poly is not None and not poly.is_empty:
                protected.append({"osm_id": el["id"], "name": name,
                                  "pcls": tags.get("boundary") or tags.get("leisure"), "geometry": poly})

buildings = [b for b in buildings if b]
log(f"roads={len(roads)} streams={len(streams)} water={len(water_polys)} settle={len(settle_polys)} "
    f"bldg={len(buildings)} places={len(places)} protected={len(protected)} "
    f"powerline={len(powerlines)} substation={len(substations)} plant={len(plants)}")
for p in protected:
    log("  保护地:", p["name"], p["pcls"])
for s in substations[:8]:
    log("  变电站:", s["name"] or s["osm_id"], s["kv"])

log("== 2. 写 shapefile ==")
def save_gdf(records, path, crs=CRS):
    if not records:
        log("  [skip empty]", os.path.basename(path)); return
    rows = [{k: v for k, v in r.items() if k != "geometry"} for r in records]
    gdf = gpd.GeoDataFrame(rows, geometry=[r["geometry"] for r in records], crs=crs)
    gdf.to_file(path, encoding="utf-8")
    log(f"  {os.path.basename(path):18s} {len(gdf):6d} features")

save_gdf(roads, os.path.join(OUT, "roads.shp"))
save_gdf(streams, os.path.join(OUT, "streams.shp"))
save_gdf(water_polys, os.path.join(OUT, "waterbody.shp"))
save_gdf(settle_polys, os.path.join(OUT, "settlements.shp"))
save_gdf(buildings, os.path.join(OUT, "buildings.shp"))
save_gdf(places, os.path.join(OUT, "places.shp"))
save_gdf(protected, os.path.join(OUT, "protected.shp"))
save_gdf(powerlines, os.path.join(OUT, "powerline.shp"))
save_gdf(substations, os.path.join(OUT, "substation.shp"))
save_gdf(plants, os.path.join(OUT, "plant.shp"))
study = gpd.GeoDataFrame({"name": ["研究区(福鼎沿海)"]}, geometry=[BBOX], crs=CRS)
study.to_file(os.path.join(OUT, "study_area.shp"), encoding="utf-8")
log("  study_area.shp written")

log("== 3. DEM 裁剪 ==")
with rasterio.open(os.path.join(TEA, "_raw", "dem", "N27_00_E120_00.tif")) as ds:
    clipped, tf = rio_mask(ds, [mapping(BBOX)], crop=True, filled=True)
    z = clipped[0]
    z = np.where((z < -100) | (z > 10000), -9999.0, z).astype("float32")
    prof = dict(driver="GTiff", height=z.shape[0], width=z.shape[1], count=1, dtype="float32",
                transform=tf, crs="EPSG:4326", nodata=-9999.0, compress="deflate", predictor=3)
    with rasterio.open(os.path.join(OUT, "dem.tif"), "w", **prof) as dst:
        dst.write(z, 1); dst.set_band_description(1, "elevation_m")
    v = z[z > -9000]
    log(f"  dem.tif {z.shape[1]}x{z.shape[0]}  高程 {v.min():.0f}~{v.max():.0f}m mean={v.mean():.0f}")

log("== 4. WorldCover 裁剪 ==")
with rasterio.open(os.path.join(TEA, "_raw", "worldcover", "ESA_WorldCover_10m_2021_v200_N27E120_Map.tif")) as ds:
    clipped, tf = rio_mask(ds, [mapping(BBOX)], crop=True)
    wc = clipped[0]
    prof = dict(driver="GTiff", height=wc.shape[0], width=wc.shape[1], count=1, dtype="uint8",
                transform=tf, crs="EPSG:4326", nodata=0, compress="deflate")
    with rasterio.open(os.path.join(OUT, "worldcover.tif"), "w", **prof) as dst:
        dst.write(wc, 1); dst.set_band_description(1, "worldcover_class")
    v, c = np.unique(wc, return_counts=True)
    names = {10: "林地", 20: "灌木", 30: "草地", 40: "耕地", 50: "建设", 60: "裸地", 80: "水体", 90: "湿地", 95: "红树林", 0: "nodata"}
    tot = c.sum()
    for vv, cc in zip(v, c):
        print(f"  class {int(vv):3d} {names.get(int(vv),'?'):6s} {100*cc/tot:6.2f}%")
log("ALL DONE")
