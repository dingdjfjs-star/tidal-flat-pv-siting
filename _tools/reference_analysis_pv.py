# -*- coding: utf-8 -*-
"""
滩涂光伏选址 · Python 参考分析（与白茶园 v3 方法论同构）
  L1 8条硬约束（坡度/地类准入/保护地/居民点/道路双面/并网/涉水/高程）
  L2 6因子模糊隶属度 × AHP-熵权组合赋权 → 0~100
  L3 敏感性 + 三情景（集中式/渔光互补浮体/分散式近网）+ 产能估算（太阳几何数值积分）
输出（参考/）：参考_适宜性图.png / 参考_候选地块清单.csv / 参考_候选地块.shp
              参考_权重表.csv / 参考_情景对比.csv / 参考_产能估算.csv
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_origin
from rasterio.features import rasterize, shapes
from scipy import ndimage
import geopandas as gpd
from shapely.geometry import shape
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = r"E:\zcode\作业\滩涂光伏选址"
DATA = os.path.join(BASE, "数据")
REF = os.path.join(BASE, "参考")
os.makedirs(REF, exist_ok=True)
RES = 30
CAND_THR = 65.0
CAND_MIN_HA = 10.0      # 集中式最小经济规模（分散式情景另计 5ha）
LAT = 27.10             # 研究区中心纬度

def log(*a): print(*a, flush=True)

# ---------- 1. DEM → UTM51N ----------
with rasterio.open(os.path.join(DATA, "dem.tif")) as ds:
    src_z = ds.read(1).astype("float32")
    src_z = np.where((src_z < -100) | (src_z > 10000), np.nan, src_z)
    src_t, src_crs = ds.transform, ds.crs
    B = ds.bounds
left, bottom, right, top = transform_bounds(src_crs, "EPSG:32651", B.left, B.bottom, B.right, B.top)
width = int(np.ceil((right - left) / RES)); height = int(np.ceil((top - bottom) / RES))
dst_t = from_origin(left, top, RES, RES)
z = np.full((height, width), np.nan, dtype="float32")
reproject(src_z, z, src_transform=src_t, src_crs=src_crs, src_nodata=np.nan,
          dst_transform=dst_t, dst_crs="EPSG:32651", dst_nodata=np.nan, resampling=Resampling.bilinear)
log(f"格网 {width}x{height}  高程 {np.nanmin(z):.0f}~{np.nanmax(z):.0f}m")

dz_drow, dz_dcol = np.gradient(np.nan_to_num(z, nan=0.0), RES, RES)
gx, gy = dz_dcol, -dz_drow
slope = np.degrees(np.arctan(np.hypot(gx, gy))); slope[np.isnan(z)] = np.nan
aspect = np.where(slope < 2.0, 180.0, (np.degrees(np.arctan2(-gx, gy)) + 360) % 360)  # 平地按南坡

# ---------- 2. 掩膜与距离 ----------
def grid_of(path, all_touched=False):
    if not os.path.exists(path):
        return np.zeros((height, width), "uint8")
    g = gpd.read_file(path).to_crs("EPSG:32651")
    if len(g) == 0:
        return np.zeros((height, width), "uint8")
    return rasterize(((geom, 1) for geom in g.geometry), out_shape=(height, width),
                     transform=dst_t, fill=0, all_touched=all_touched, dtype="uint8")

roads = gpd.read_file(os.path.join(DATA, "roads.shp")).to_crs("EPSG:32651")
MAJOR = ("motorway", "trunk", "primary")
roads_m = rasterize(((g, 1) for g in roads.geometry), out_shape=(height, width),
                    transform=dst_t, fill=0, all_touched=True, dtype="uint8")
major_m = rasterize(((g, 1) for g in roads[roads["cls"].isin(MAJOR)].geometry),
                    out_shape=(height, width), transform=dst_t, fill=0, all_touched=True, dtype="uint8")
streams_m = grid_of(os.path.join(DATA, "streams.shp"), all_touched=True)
water_g = gpd.read_file(os.path.join(DATA, "waterbody.shp")).to_crs("EPSG:32651")
water_g["area_m2"] = water_g.geometry.area
small_m = rasterize(((g, 1) for g in water_g[water_g.area_m2 < 10000].geometry),
                    out_shape=(height, width), transform=dst_t, fill=0, dtype="uint8")
reser_m = rasterize(((g, 1) for g in water_g[water_g.area_m2 >= 10000].geometry),
                    out_shape=(height, width), transform=dst_t, fill=0, dtype="uint8")
settle_m = grid_of(os.path.join(DATA, "settlements.shp"))
bldg_m = grid_of(os.path.join(DATA, "buildings.shp"))
places_m = grid_of(os.path.join(DATA, "places.shp"), all_touched=True)
prot_m = grid_of(os.path.join(DATA, "protected.shp"))
power_m = grid_of(os.path.join(DATA, "powerline.shp"), all_touched=True)

d_road = ndimage.distance_transform_edt(roads_m == 0) * RES
d_major = ndimage.distance_transform_edt(major_m == 0) * RES
d_small = ndimage.distance_transform_edt(small_m == 0) * RES
d_reser = ndimage.distance_transform_edt(reser_m == 0) * RES
d_water = np.minimum(d_small, d_reser)
d_settle = ndimage.distance_transform_edt((settle_m | bldg_m) == 0) * RES
d_place = ndimage.distance_transform_edt(places_m == 0) * RES
d_prot = ndimage.distance_transform_edt(prot_m == 0) * RES
d_power = (ndimage.distance_transform_edt(power_m == 0) * RES) if power_m is not None else None
log("电力线图层:", "有" if power_m is not None else "缺失（OSM 未映射）")

with rasterio.open(os.path.join(DATA, "worldcover.tif")) as ds:
    wc = np.full((height, width), 255, dtype="uint8")
    reproject(ds.read(1), wc, src_transform=ds.transform, src_crs=ds.crs,
              dst_transform=dst_t, dst_crs="EPSG:32651", dst_nodata=255, resampling=Resampling.nearest)

# ---------- 3. L1 硬约束 ----------
log("== L1 硬约束 ==")
ok = {
    "H1 坡度≤8°": slope <= 8,
    "H2 地类准入(滩涂/灌草/裸)": np.isin(wc, [20, 30, 60, 90]),
    "H3 非保护地": prot_m == 0,
    "H4 距居≥150/村镇点≥300m": (d_settle >= 150) & (d_place >= 300),
    "H5 路50-2000且高速≥200m": (d_road > 50) & (d_road <= 2000) & (d_major >= 200),
    "H6 距输电线≤3000m": (d_power <= 3000) if power_m is not None else np.ones((height, width), bool),
    "H7 距水≥50/水库≥300m": (d_water >= 50) & (d_reser >= 300),
    "H8 高程≥2m": z >= 2,
}
hard = np.ones((height, width), bool)
for nm, m in ok.items():
    hard &= m & np.isfinite(z)
    log(f"  {nm:26s} {m.sum()*0.09:8.0f} ha ({100*m.sum()/hard.size:5.1f}%)")
log(f"  {'全部通过':26s} {hard.sum()*0.09:8.0f} ha ({100*hard.sum()/hard.size:5.1f}%)")

# ---------- 4. 模糊隶属度 ----------
def lut_lookup(lut_dict):
    lut = np.zeros(256, dtype="float32")
    for k2, v2 in lut_dict.items():
        if 0 <= k2 <= 255:
            lut[k2] = v2
    return lut[wc]

LC_MU = {60: 1.0, 20: 0.8, 30: 0.8, 90: 0.7, 40: 0.0, 10: 0.0, 50: 0.0, 80: 0.0, 95: 0.2, 255: 0.0}
MU0 = {
    "slope": np.clip((8 - slope) / 6.0, 0, 1),
    "power": np.clip((3000 - d_power) / 2500.0, 0, 1) if power_m is not None else np.full((height, width), 0.5, "float32"),
    "road": np.where(d_road <= 300, 1.0, np.clip(1 - (d_road - 300) / 1700 * 0.7, 0.3, 1.0)),
    "settle": np.clip(0.5 + 0.5 * (d_settle - 150) / 650.0, 0.5, 1.0),
    "lc": lut_lookup(LC_MU),
    "prot": np.clip(0.5 + 0.5 * d_prot / 2000.0, 0.5, 1.0),
}
for k, v in MU0.items():
    v = np.asarray(v, dtype="float32"); v[np.isnan(z)] = np.nan; MU0[k] = v

AHP = {"slope": .18, "power": .28, "road": .16, "settle": .10, "lc": .16, "prot": .12}
dom = hard & np.isfinite(z)
N = int(dom.sum())
ent = {}
for k, m in MU0.items():
    x = m[dom].astype("float64")
    rng = x.max() - x.min()
    xn = (x - x.min()) / rng if rng > 0 else np.ones_like(x) * 0.5
    p = np.clip(xn / xn.sum(), 1e-12, 1)
    ent[k] = 1 - (-(p * np.log(p)).sum() / np.log(N))
s = sum(ent.values()); ent = {k: v / s for k, v in ent.items()}
comb = {k: np.sqrt(AHP[k] * ent[k]) for k in AHP}
s = sum(comb.values()); comb = {k: v / s for k, v in comb.items()}
log("权重：AHP | 熵权 | 组合")
for k in comb:
    log(f"  {k:7s} {AHP[k]:.3f} | {ent[k]:.3f} | {comb[k]:.3f}")
with open(os.path.join(REF, "参考_权重表.csv"), "w", encoding="utf-8-sig") as f:
    f.write("因子,AHP主观权重,熵权客观权重,组合权重\n")
    for k in comb:
        f.write(f"{k},{AHP[k]:.3f},{ent[k]:.3f},{comb[k]:.3f}\n")

def score(mu, w):
    out = np.zeros_like(mu["slope"])
    for k in w:
        out += w[k] * mu[k]
    return 100.0 * out

S0 = np.where(hard, score(MU0, comb), np.nan)
for lo, hi, tag in [(80, 101, "Ⅰ"), (65, 80, "Ⅱ"), (50, 65, "Ⅲ"), (0, 50, "Ⅳ")]:
    n = int(((S0 >= lo) & (S0 < hi)).sum())
    log(f"  {tag} {n*0.09:8.1f} ha ({100*n/hard.size:.1f}%)")

# ---------- 5. 产能估算（太阳几何数值积分） ----------
log("== 5. 太阳辐射/产能估算 ==")
def tilt_factor(beta_deg, gamma_deg, lat=LAT, KT=0.42, df=0.45, rho=0.2):
    """年 POA/GHI 转换系数：12 个月逐时数值积分（天外辐射 × 各向同性天空）"""
    def decl(n):  # Cooper
        return np.radians(23.45) * np.sin(np.radians(360 * (284 + n) / 365))
    def H0_day(n):  # 月日均天外辐射 kWh/m²/day（Duffie & Beckman 标准式）
        d = decl(n)
        x = -np.tan(lat) * np.tan(d)
        ws = np.arccos(np.clip(x, -1, 1))
        return (24 / np.pi) * 1.367 * (ws * np.sin(np.radians(lat)) * np.sin(d)
                                       + np.cos(np.radians(lat)) * np.cos(d) * np.sin(ws))
    beta = np.radians(beta_deg); gamma = np.radians(gamma_deg)
    phi = np.radians(lat)
    n_east_n, n_north_n, n_up_n = np.sin(beta) * np.sin(gamma), np.sin(beta) * np.cos(gamma), np.cos(beta)
    total_poa, total_ghi = 0.0, 0.0
    omegas = np.arange(-np.pi, np.pi + 1e-9, np.pi / 300)
    for doy in (15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349):
        d = decl(doy)
        up_s = np.sin(d) * np.sin(phi) + np.cos(d) * np.cos(phi) * np.cos(omegas)
        e_s = -np.cos(d) * np.sin(omegas)
        n_s = np.sin(d) * np.cos(phi) - np.cos(d) * np.sin(phi) * np.cos(omegas)
        day = up_s > 0
        cosT = up_s * n_up_n + e_s * n_east_n + n_s * n_north_n
        beam_h = np.sum(np.maximum(up_s[day], 0))
        beam_t = np.sum(np.maximum(cosT[day], 0))
        Rb = beam_t / beam_h if beam_h > 0 else 1.0
        H0 = H0_day(doy)
        GHI = KT * H0
        POA = (1 - df) * GHI * Rb + df * GHI * (1 + np.cos(beta)) / 2 + GHI * rho * (1 - np.cos(beta)) / 2
        days = {15:29, 46:31, 74:30, 105:31, 135:30, 166:31, 196:31, 227:31, 258:30, 288:31, 319:30, 349:31}[doy]
        total_poa += POA * days
        total_ghi += GHI * days
    return total_poa / total_ghi

f_flat = tilt_factor(0, 180)
log(f"  平地年转换系数 f(0°)={f_flat:.3f}")
# 敏感性：南坡 5° / 北坡 5°
log(f"  南坡5° f={tilt_factor(5,180):.3f}  北坡5° f={tilt_factor(5,0):.3f}")

GHI_REF = 1210.0   # 福鼎沿海年总辐射参照 kWh/m²（按 KT=0.42 模型标定，文献区间 1150~1300）
ETA_MOD, PR, GCR = 0.21, 0.80, 0.40   # 组件效率 / 系统效率比 / 阵列覆盖系数（0.4 ≈ 0.84 MW/ha）

def patch_energy(area_ha, beta, gamma):
    """年发电量 MWh：E = f × GHI × η × GCR × PR × 面积（覆盖系数保证装机-发电自洽）"""
    f = tilt_factor(float(beta), float(gamma)) if beta >= 1.0 else f_flat
    return f * GHI_REF * ETA_MOD * GCR * PR * area_ha * 1e4 / 1000.0  # MWh/yr

def patch_capacity_gw(area_ha):
    """装机容量 GW：面积 × 0.84 MW/ha"""
    return area_ha * 0.84 / 1000.0

# ---------- 6. 情景 ----------
log("== 6. 三经营情景 ==")
WC_S2 = {80: 1.0, 90: 1.0, 60: 0.9, 20: 0.5, 30: 0.5, 40: 0.0, 10: 0.0, 50: 0.0, 95: 0.2, 255: 0.0}

def hard_base():
    return hard
def hard_s2():  # 渔光互补浮体：放宽到水面/滩涂，坡度几乎不设限（水面），仍需并网/涉水退距
    return (np.isin(wc, [80, 90]) & (d_power <= 3000 if power_m is not None else True)
            & (d_settle >= 150) & (d_place >= 300) & (d_road > 50) & (d_road <= 2500) & (d_major >= 200)
            & (prot_m == 0) & (z >= 1) & np.isfinite(z))
def hard_s3():  # 分散式近网：并网收紧至 1500m，规模门槛降至 5ha
    return hard & (d_power <= 1500)

SCEN = {
    "S1 集中式地面": dict(hard=hard_base, mu=MU0, w=comb, minha=10.0),
    "S2 渔光互补浮体": dict(hard=hard_s2, w={"slope": .05, "power": .28, "road": .14, "settle": .08, "lc": .30, "prot": .15}, minha=10.0,
                        mu={"lc": lut_lookup(WC_S2), "slope": np.full((height, width), 1.0, "float32")}),
    "S3 分散式近网": dict(hard=hard_s3, mu=MU0, w={"slope": .15, "power": .38, "road": .18, "settle": .08, "lc": .13, "prot": .08}, minha=5.0),
}
town_pts = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy([120.324, 120.248, 120.417, 120.358], [27.161, 27.104, 27.167, 27.187]),
    crs="EPSG:4326").to_crs("EPSG:32651")
WC_NAMES = {20: "灌木", 30: "草地", 60: "裸地", 90: "滩涂湿地", 80: "水面"}

def patches_of(S, hm, minha):
    c = np.where(hm, S, np.nan) >= CAND_THR
    lab, nl = ndimage.label(c, structure=np.ones((3, 3)))
    out = []
    for i in range(1, nl + 1):
        m = lab == i
        n = int(m.sum())
        if n * 0.09 < minha:
            continue
        rows, cols = np.where(m)
        cx, cy = dst_t.c + (cols.mean() + .5) * dst_t.a, dst_t.f + (rows.mean() + .5) * dst_t.e
        dd = [((cx - p.x) ** 2 + (cy - p.y) ** 2) for p in town_pts.geometry]
        b, g_ = float(np.nanmean(slope[m])), 180.0
        av = aspect[m]; avf = av[np.isfinite(av)]
        if len(avf):
            g_ = float(np.bincount((avf.astype(int) // 5).clip(0, 71)).argmax() * 5 + 2.5)
        wv = wc[m]; wv = wv[wv != 255]
        maj = int(np.bincount(wv).argmax()) if len(wv) else 255
        out.append({"area_ha": n * 0.09, "S": float(np.nanmean(np.where(hm, S, np.nan)[m])),
                    "elev": float(np.nanmean(z[m])), "slope": b,
                    "town": ["店下镇", "太姥山镇", "沙埕镇", "龙安街道"][int(np.argmin(dd))],
                    "lc": WC_NAMES.get(maj, "-"), "beta": b, "gamma": g_,
                    "energy_GWh": patch_energy(n * 0.09, b, g_) / 1000.0,
                    "mask": m, "cx": cx, "cy": cy})
    return out

scenario_rows = []
all_masks, base_pts = {}, []
for nm, sc in SCEN.items():
    hm = sc["hard"]()
    mu = {k: v.copy() for k, v in MU0.items()}
    for k, v in sc.get("mu", {}).items():
        v = np.asarray(v, "float32"); v[np.isnan(z)] = np.nan; mu[k] = v
    w = sc["w"]; wsum = sum(w.values()); w = {k: v / wsum for k, v in w.items()}
    S = np.where(hm, score(mu, w), np.nan)
    pts = patches_of(S, hm, sc["minha"])
    pts.sort(key=lambda p: -p["S"])
    all_masks[nm] = (S >= CAND_THR) & hm
    cand_ha = sum(p["area_ha"] for p in pts)
    gw = sum(p["energy_GWh"] for p in pts)
    gw_cap = patch_capacity_gw(cand_ha)
    t1 = pts[0] if pts else None
    scenario_rows.append({"情景": nm, "硬约束通过ha": round(hm.sum() * 0.09), "候选面积ha": round(cand_ha),
                          "候选片数": len(pts), "装机潜力GW": round(gw_cap, 3),
                          "年发电量GWh": round(gw, 1),
                          "Top1": f"{t1['area_ha']:.0f}ha/{t1['S']:.0f}分/{t1['energy_GWh']:.1f}GWh/{t1['town']}" if t1 else "-"})
    log(f"  {nm:8s} 通过 {hm.sum()*0.09:6.0f} ha | 候选 {cand_ha:6.0f} ha/{len(pts)} 片 | "
        f"装机≈{gw_cap:.3f} GW | 年发电≈{gw:.0f} GWh"
        + (f" | Top1 {t1['area_ha']:.0f}ha S={t1['S']:.0f}" if t1 else ""))
with open(os.path.join(REF, "参考_情景对比.csv"), "w", encoding="utf-8-sig") as f:
    f.write("情景,硬约束通过ha,候选面积ha,候选片数,装机潜力GW,年发电量GWh,Top1\n")
    for r in scenario_rows:
        f.write(f"{r['情景']},{r['硬约束通过ha']},{r['候选面积ha']},{r['候选片数']},{r['装机潜力GW']},{r['年发电量GWh']},{r['Top1']}\n")

# 共识区（3 情景中 ≥2 入选）
cnt = np.zeros((height, width), "uint8")
for nm in all_masks:
    cnt += all_masks[nm].astype("uint8")
consensus = cnt >= 2
lab, nlc = ndimage.label(consensus, structure=np.ones((3, 3)))
cons_a = sum((lab == i).sum() * 0.09 for i in range(1, nlc + 1) if (lab == i).sum() * 0.09 >= 2)
log(f"共识区（≥2情景）：{cons_a:.0f} ha")

# 基准候选导出 + 产能表
base_pts = patches_of(S0, hard, 10.0)
base_pts.sort(key=lambda p: -p["S"])
recs, geoms = [], []
for idx, p in enumerate(base_pts, 1):
    gi = shapes(p["mask"].astype("uint8"), mask=p["mask"], transform=dst_t)
    geoms.append(max((shape(g) for g, v in gi), key=lambda q: q.area))
    recs.append({"id": idx, "town": p["town"], "area_ha": round(p["area_ha"], 1), "S": round(p["S"], 1),
                 "elev_m": round(p["elev"]), "slope_deg": round(p["slope"], 1), "landcover": p["lc"],
                 "energy_GWh": round(p["energy_GWh"], 2)})
gpd.GeoDataFrame(recs, geometry=geoms, crs="EPSG:32651").to_file(
    os.path.join(REF, "参考_候选地块.shp"), encoding="utf-8")
with open(os.path.join(REF, "参考_候选地块清单.csv"), "w", encoding="utf-8-sig") as f:
    f.write("编号,所属乡镇,面积ha,适宜性S,平均海拔m,平均坡度°,现状地类,年发电量GWh\n")
    for r in sorted(recs, key=lambda r: -r["energy_GWh"]):
        f.write(f"{r['id']},{r['town']},{r['area_ha']},{r['S']},{r['elev_m']},{r['slope_deg']},{r['landcover']},{r['energy_GWh']}\n")
log(f"基准候选 {len(recs)} 片 / {sum(r['area_ha'] for r in recs):.0f} ha / "
    f"年发电合计 {sum(r['energy_GWh'] for r in recs):.0f} GWh")
for r in recs[:5]:
    log(f"  #{r['id']} {r['town']} {r['area_ha']}ha S={r['S']} {r['landcover']} {r['energy_GWh']}GWh")

# 敏感性（基准）
def cand_area(w):
    s = np.where(hard, score(MU0, w), np.nan)
    return float(np.nansum((s >= CAND_THR)) * 0.09)
base_a = cand_area(comb)
log(f"敏感性基准: {base_a:.0f} ha")
for k in comb:
    for mult in (0.5, 1.5):
        W2 = dict(comb); W2[k] = comb[k] * mult
        rest = sum(v for kk, v in comb.items() if kk != k)
        for kk in W2:
            if kk != k:
                W2[kk] = W2[kk] / rest * (1 - W2[k])
        a = cand_area(W2)
        log(f"  {k:7s} ×{mult} → {a:7.0f} ha ({(a-base_a)/base_a*100:+.1f}%)")

# ---------- 7. 出图 ----------
zen = np.radians(90 - 45); azi = np.radians(315)
hs = (np.cos(zen) * np.cos(np.radians(slope)) +
      np.sin(zen) * np.sin(np.radians(slope)) * np.cos(azi - np.radians(np.where(np.isfinite(aspect), aspect, 180))))
ext = [left, right, bottom, top]
# 线图层裁剪到研究区（OSM way 完整几何会越界，撑大视野）
from shapely.geometry import box as sbox
study_utm = gpd.GeoSeries([sbox(120.16, 27.00, 120.48, 27.22)], crs="EPSG:4326").to_crs("EPSG:32651")
def clipped(path):
    return gpd.clip(gpd.read_file(path).to_crs("EPSG:32651"), study_utm)
fig = plt.figure(figsize=(17.5, 8.8), dpi=130)
gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 0.58, 0.58], hspace=0.28, wspace=0.1)
ax1 = fig.add_subplot(gs[:, 0])
ax1.imshow(hs, cmap="gray", vmin=0, vmax=1.15, extent=ext, origin="upper")
cmap = matplotlib.colors.ListedColormap(["#e8b64c", "#a8cf6e", "#4f9e58", "#1d6e42"])
S0c = np.where(np.isnan(S0), -1, np.digitize(S0, [50, 65, 80]))
ax1.imshow(np.ma.masked_equal(S0c, -1), cmap=cmap, vmin=0, vmax=3, extent=ext, origin="upper", alpha=0.92)
cons_edge = consensus ^ ndimage.binary_erosion(consensus, np.ones((3, 3)))
ax1.imshow(np.ma.masked_equal(cons_edge, 0), cmap=matplotlib.colors.ListedColormap(["#1f2d7a"]),
           vmin=0, vmax=1, extent=ext, origin="upper", alpha=0.9)
rds = clipped(os.path.join(DATA, "roads.shp"))
for gg, cc, lw in [(rds[rds["cls"].isin(MAJOR)], "#3a3a3a", 0.8), (rds[~rds["cls"].isin(MAJOR)], "#8a8a8a", 0.3)]:
    gg.plot(ax=ax1, color=cc, linewidth=lw)
if power_m is not None:
    clipped(os.path.join(DATA, "powerline.shp")).plot(ax=ax1, color="#c0392b", linewidth=0.8)
clipped(os.path.join(DATA, "waterbody.shp")).plot(ax=ax1, color="#3f7fbf")
clipped(os.path.join(DATA, "settlements.shp")).plot(
    ax=ax1, facecolor="none", edgecolor="#8a1f1f", linewidth=0.5)
ax1.set_xlim(left, right); ax1.set_ylim(bottom, top)
handles = [mpatches.Patch(color="#1d6e42", label="Ⅰ 最适宜 (≥80)"),
           mpatches.Patch(color="#4f9e58", label="Ⅱ 适宜 (65~80)"),
           mpatches.Patch(color="#a8cf6e", label="Ⅲ 一般 (50~65)"),
           mpatches.Patch(color="#e8b64c", label="Ⅳ 较差 (<50)"),
           mpatches.Patch(color="#dddddd", label="未通过硬约束"),
           mpatches.Patch(color="#c0392b", label="输电线路"),
           mpatches.Patch(facecolor="none", edgecolor="#1f2d7a", label="多情景共识区")]
ax1.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
ax1.set_title("福鼎沿海滩涂光伏选址参考图 v1（S1 集中式基准）\n8硬约束 + 6因子模糊隶属度 × AHP-熵权组合赋权（0~100）", fontsize=12)
ax1.set_xlabel("UTM 51N 东坐标 (m)"); ax1.set_ylabel("北坐标 (m)")
pos = [(0, 1), (0, 2), (1, 1)]
for (r_, c_), (nm, _) in zip(pos, list(SCEN.items())[:3]):
    ax = fig.add_subplot(gs[r_, c_])
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1.15, extent=ext, origin="upper")
    ax.imshow(np.ma.masked_equal(all_masks[nm], 0), cmap=matplotlib.colors.ListedColormap(["#2e7d32"]),
              vmin=0, vmax=1, extent=ext, origin="upper", alpha=0.7)
    ax.set_xlim(left, right); ax.set_ylim(bottom, top)
    ax.set_title(nm, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
axb = fig.add_subplot(gs[1, 2])
names = [r["情景"].split(" ")[0] for r in scenario_rows]
vals = [r["年发电量GWh"] for r in scenario_rows]
bars = axb.bar(names, vals, color=["#4f9e58", "#2e86c1", "#e67e22"])
for b, v in zip(bars, vals):
    axb.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
axb.set_ylabel("年发电量 (GWh/yr)")
axb.set_title("三情景年发电量对比", fontsize=10)
axb.tick_params(labelsize=8)
plt.savefig(os.path.join(REF, "参考_适宜性图.png"), bbox_inches="tight")
log("ALL DONE PV")
