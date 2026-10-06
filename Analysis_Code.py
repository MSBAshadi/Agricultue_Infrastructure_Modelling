
import argparse, os, time, warnings
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ configuration
COLS = dict(pid="id", year="year", lat="lat", lon="lon", elev="elevation", slope="slope",
            dist="dist_water", ndvi="NDVI", evi="EVI", ndwi="NDWI", rain="rainfall", lst="LST",
            sm="soil_moisture")

# Road label (deterministic rule in the paper)
ROAD_SLOPE_MAX, ROAD_ELEV_MAX = 15.0, 1000.0

SEVERE_SLOPE, SEVERE_ELEV = 25.0, 1200.0


SUIT = dict(rain_min=500.0, rain_max=1200.0, lst_min=15.0, lst_max=30.0, slope_max=15.0)

PIXEL_DEG = 0.05        
K_FOLDS, BLOCKS, SEED = 10, (5, 10, 20), 42
TOP_FRAC = 0.10


# ------------------------------------------------------------------ data + labels
def load(path):
    d = pd.read_csv(path)
    c = COLS
    d = d.rename(columns={v: k for k, v in c.items()})
    need = list(c.keys())
    miss = [n for n in need if n not in d.columns]
    if miss:
        raise SystemExit(f"Missing columns after renaming: {miss}. Edit COLS.")
    d["road"] = ((d.slope < ROAD_SLOPE_MAX) & (d.elev < ROAD_ELEV_MAX)).astype(int)
    d["irr"] = d["irrigation_need"]                             
    d["px"] = (np.floor(d.lat / PIXEL_DEG).astype(int).astype(str) + "_" +
               np.floor(d.lon / PIXEL_DEG).astype(int).astype(str))
    return d.reset_index(drop=True)


# Leakage-aware predictor sets 
FEATS = {
    "road": ["dist", "lat", "lon", "ndvi", "evi", "ndwi", "rain", "lst", "sm", "year"],
    "irr":  ["elev", "slope", "dist", "lat", "lon", "ndvi", "evi", "lst", "year"],
}
STATIC = {"road": ["dist", "lat", "lon"], "irr": ["elev", "slope", "dist", "lat", "lon"]}
TASK_TYPE = {"road": "clf", "irr": "reg"}


def make_model(kind, task, trees):
    clf = TASK_TYPE[task] == "clf"
    if kind == "RF":
        return (RandomForestClassifier if clf else RandomForestRegressor)(trees, random_state=SEED, n_jobs=-1)
    if kind == "RF_reg":   # regularised forest (sensitivity check)
        return (RandomForestClassifier if clf else RandomForestRegressor)(
            trees, min_samples_leaf=20, random_state=SEED, n_jobs=-1)
    if kind == "XGB":
        return (XGBClassifier if clf else XGBRegressor)(n_estimators=trees, random_state=SEED, n_jobs=-1,
                                                        **({"eval_metric": "logloss"} if clf else {}))
    if kind == "Linear":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000) if clf else Ridge(alpha=1.0))
    raise ValueError(kind)


# ------------------------------------------------------------------ fold construction
def folds_rows(n, rng, k=K_FOLDS):
    f = np.empty(n, int); f[rng.permutation(n)] = np.arange(n) % k; return f

def folds_groups(groups, rng, k=K_FOLDS):
    u, inv = np.unique(groups, return_inverse=True)
    g = np.empty(len(u), int); g[rng.permutation(len(u))] = np.arange(len(u)) % k
    return g[inv]

def blocks(d, k):
    pts = d.groupby("pid")[["lat", "lon"]].first()
    lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(pts.values)
    return pd.Series(lab, index=pts.index).reindex(d.pid).values


def make_folds(d, protocol, rng):
    if protocol == "P1":  return folds_rows(len(d), rng), d.pid.values
    if protocol == "P2":  return folds_groups(d.pid.values, rng), d.pid.values
    if protocol == "PIX": return folds_groups(d.px.values, rng), d.px.values
    if protocol.startswith("P3_"):
        b = blocks(d, int(protocol.split("_")[1])); return b, b
    raise ValueError(protocol)


# ------------------------------------------------------------------ core CV + metrics
def oof_predict(d, task, feats, kind, folds, trees):
    X, y = d[feats].values, d[task].values
    pred = np.zeros(len(d)); cls = np.zeros(len(d), int)
    for k in np.unique(folds):
        te = folds == k; tr = ~te
        m = make_model(kind, task, trees).fit(X[tr], y[tr])
        if TASK_TYPE[task] == "clf":
            pred[te] = m.predict_proba(X[te])[:, 1]; cls[te] = (pred[te] >= 0.5).astype(int)
        else:
            pred[te] = m.predict(X[te])
    return pred, cls

def metric(task, y, pred, cls):
    return float((cls == y).mean()) if TASK_TYPE[task] == "clf" else float(r2_score(y, pred))

def fold_mean_metric(task, y, pred, cls, folds):
    v = [metric(task, y[folds == k], pred[folds == k], cls[folds == k]) for k in np.unique(folds)]
    return float(np.mean(v)), float(np.std(v))


def cluster_boot_delta(task, y, a, b, groups, B=2000, seed=SEED):
    """Pooled difference (model a minus model b) with a cluster bootstrap over `groups`."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(groups, return_inverse=True); G = len(u)
    n = np.bincount(inv).astype(float)
    if TASK_TYPE[task] == "clf":
        ca = np.bincount(inv, weights=(a[1] == y)); cb = np.bincount(inv, weights=(b[1] == y))
        stat = lambda w: ((w @ ca) - (w @ cb)) / (w @ n)
    else:
        sy, syy = np.bincount(inv, weights=y), np.bincount(inv, weights=y * y)
        ra, rb = np.bincount(inv, weights=(y - a[0]) ** 2), np.bincount(inv, weights=(y - b[0]) ** 2)
        def stat(w):
            N = w @ n; sst = (w @ syy) - (w @ sy) ** 2 / N
            return (1 - (w @ ra) / sst) - (1 - (w @ rb) / sst)
    obs = stat(np.ones(G))
    boots = np.array([stat(np.bincount(rng.integers(0, G, G), minlength=G).astype(float)) for _ in range(B)])
    return obs, *np.percentile(boots, [2.5, 97.5])


# ------------------------------------------------------------------ analysis
def ladder_and_deltas(d, trees, reps, out):
    rows_l, rows_d = [], []
    for task in FEATS:
        feats = FEATS[task]; y = d[task].values
        for protocol in ["P1", "P2", "PIX"] + [f"P3_{k}" for k in BLOCKS]:
            n_rep = reps if protocol in ("P1", "P2", "PIX") else 1      # blocks are deterministic
            for r in range(n_rep):
                f, grp = make_folds(d, protocol, np.random.default_rng(SEED + r))
                res = {m: oof_predict(d, task, feats, m, f, trees) for m in ("RF", "XGB", "Linear")}
                for m, (p, c) in res.items():
                    mean, sd = fold_mean_metric(task, y, p, c, f)
                    rows_l.append(dict(task=task, protocol=protocol, rep=r, model=m,
                                       pooled=metric(task, y, p, c), fold_mean=mean, fold_sd=sd))
                if r == 0:
                    for m in ("RF", "XGB"):
                        obs, lo, hi = cluster_boot_delta(task, y, res[m], res["Linear"], grp)
                        rows_d.append(dict(task=task, protocol=protocol, model=m, delta=obs, lo=lo, hi=hi))
            print(f"[ladder] {task} {protocol}", flush=True)
    pd.DataFrame(rows_l).to_csv(f"{out}/ladder_repeated.csv", index=False)
    pd.DataFrame(rows_d).to_csv(f"{out}/paired_deltas.csv", index=False)


def controls(d, trees, out):
    rows = []
    for task in FEATS:
        feats, static = FEATS[task], STATIC[task]
        y = d[task].values
        # (a) single-year control: one row per point, random 10-fold (no duplication)
        for yr in sorted(d.year.unique()):
            ds = d[d.year == yr].reset_index(drop=True)
            f = folds_rows(len(ds), np.random.default_rng(SEED))
            for m in ("RF", "XGB", "Linear"):
                p, c = oof_predict(ds, task, [x for x in feats if x != "year"], m, f, trees)
                rows.append(dict(task=task, control="single_year_random_cv", detail=int(yr), model=m,
                                 value=metric(task, ds[task].values, p, c)))
        # (b) static-feature ablation under P1 and P2
        for protocol in ("P1", "P2"):
            f, _ = make_folds(d, protocol, np.random.default_rng(SEED))
            for label, fs in (("all_features", feats), ("no_coordinates", [x for x in feats if x not in ("lat", "lon")]),
                              ("no_static_features", [x for x in feats if x not in static])):
                for m in ("RF", "XGB", "Linear"):
                    p, c = oof_predict(d, task, fs, m, f, trees)
                    rows.append(dict(task=task, control=f"ablation_{protocol}", detail=label, model=m,
                                     value=metric(task, y, p, c)))
        # (c) regularised forest under P2 and P3_10
        for protocol in ("P2", "P3_10"):
            f, _ = make_folds(d, protocol, np.random.default_rng(SEED))
            for m in ("RF", "RF_reg"):
                p, c = oof_predict(d, task, feats, m, f, trees)
                rows.append(dict(task=task, control=f"regularised_{protocol}", detail="min_samples_leaf=20" if m == "RF_reg" else "default",
                                 model=m, value=metric(task, y, p, c)))
        print(f"[controls] {task}", flush=True)
    pd.DataFrame(rows).to_csv(f"{out}/controls.csv", index=False)


def temporal(d, trees, out, space_time=True, k_blocks=5):
    rows = []
    for task in ("irr",):                 # road label is static: leave-one-year-out is NOT a temporal test for it
        feats = [x for x in FEATS[task] if x != "year"]; y = d[task].values
        f = d.year.values
        res = {m: oof_predict(d, task, feats, m, f, trees) for m in ("RF", "XGB", "Linear")}
        for m in ("RF", "XGB"):
            obs, lo, hi = cluster_boot_delta(task, y, res[m], res["Linear"], d.pid.values)
            rows.append(dict(task=task, scheme="leave_one_year_out", model=m, delta=obs, lo=lo, hi=hi,
                             **{f"{k}": metric(task, y, *res[k]) for k in res}))
        if space_time:
            b = blocks(d, k_blocks); pred = {m: (np.zeros(len(d)), np.zeros(len(d), int)) for m in ("RF", "XGB", "Linear")}
            for bk in np.unique(b):
                for yr in np.unique(d.year):
                    te = (b == bk) & (d.year.values == yr); tr = (b != bk) & (d.year.values != yr)
                    for m in pred:
                        mod = make_model(m, task, trees).fit(d.loc[tr, feats].values, y[tr])
                        if TASK_TYPE[task] == "clf":
                            p = mod.predict_proba(d.loc[te, feats].values)[:, 1]; pred[m][0][te] = p; pred[m][1][te] = (p >= .5)
                        else:
                            pred[m][0][te] = mod.predict(d.loc[te, feats].values)
            for m in ("RF", "XGB"):
                obs, lo, hi = cluster_boot_delta(task, y, pred[m], pred["Linear"], b)
                rows.append(dict(task=task, scheme=f"leave_block_and_year_out_k{k_blocks}", model=m, delta=obs, lo=lo, hi=hi,
                                 **{f"{k}": metric(task, y, *pred[k]) for k in pred}))
        print(f"[temporal] {task}", flush=True)
    pd.DataFrame(rows).to_csv(f"{out}/temporal.csv", index=False)


def decision_metrics(d, trees, out):
    rows, app = [], []
    pts = d.groupby("pid").agg(lat=("lat", "first"), lon=("lon", "first"), slope=("slope", "first"),
                               elev=("elev", "first"), road=("road", "first")).reset_index()
    for protocol in ("P2", "P3_10"):
        f, _ = make_folds(d, protocol, np.random.default_rng(SEED))
        for task in ("road", "irr"):
            for m in ("RF", "XGB", "Linear"):
                p, c = oof_predict(d, task, FEATS[task], m, f, trees)
                dd = d.assign(pred=p, cls=c).groupby("pid").agg(pred=("pred", "mean"), cls=("cls", "mean"),
                                                                 truth=(task, "mean"), slope=("slope", "first"),
                                                                 elev=("elev", "first")).reset_index()
                n_top = max(1, int(round(TOP_FRAC * len(dd))))
                if task == "road":
                    top = dd.nlargest(n_top, "pred")
                    pred_feas = dd[dd.cls >= 0.5]
                    severe = ((pred_feas.slope >= SEVERE_SLOPE) | (pred_feas.elev >= SEVERE_ELEV)).mean() if len(pred_feas) else np.nan
                    rows.append(dict(protocol=protocol, task=task, model=m, top_decile_precision=(top.truth >= .5).mean(),
                                     severe_false_feasible_share=severe))
                else:
                    top_true = set(dd.nlargest(n_top, "truth").pid); top_pred = set(dd.nlargest(n_top, "pred").pid)
                    rows.append(dict(protocol=protocol, task=task, model=m,
                                     top_decile_overlap=len(top_true & top_pred) / n_top))
        # applicability: nearest training-point distance for each held-out point (km)
        fp, _ = make_folds(d, protocol, np.random.default_rng(SEED))
        p, c = oof_predict(d, "road", FEATS["road"], "RF", fp, trees)
        fold_of_pt = pd.Series(fp).groupby(d.pid.values).first()
        xy = np.radians(pts[["lat", "lon"]].values)
        for k in np.unique(fp):
            te_pts = pts.pid.isin(fold_of_pt.index[fold_of_pt == k]); tr_pts = ~te_pts
            if tr_pts.sum() == 0 or te_pts.sum() == 0: continue
            nn = NearestNeighbors(n_neighbors=1, metric="haversine").fit(xy[tr_pts.values])
            dist = nn.kneighbors(xy[te_pts.values])[0][:, 0] * 6371.0
            ok = (pd.Series(c == d.road.values).groupby(d.pid.values).mean().reindex(pts.pid[te_pts]).values)
            app.append(pd.DataFrame(dict(protocol=protocol, pid=pts.pid[te_pts].values, fold=k, nn_train_km=dist, correct_share=ok)))
        print(f"[decision] {protocol}", flush=True)
    pd.DataFrame(rows).to_csv(f"{out}/decision_metrics.csv", index=False)
    pd.concat(app).to_csv(f"{out}/applicability.csv", index=False)


def morans_i(xy_deg, z, k=8, n_perm=499, seed=SEED):
    rng = np.random.default_rng(seed)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy_deg); idx = nn.kneighbors(xy_deg)[1][:, 1:]
    z = z - z.mean()
    def stat(v):
        num = sum((v * v[idx[:, j]]).sum() for j in range(k))
        return num / (k * (v ** 2).sum())          # (N/W) * sum(w z z) / sum(z^2) with W = N*k binary weights
    obs = stat(z)
    perm = np.array([stat(rng.permutation(z)) for _ in range(n_perm)])
    return obs, (1 + (np.abs(perm) >= abs(obs)).sum()) / (n_perm + 1)


def spatial_dependence(d, out):
    rows, vg = [], []
    for task in FEATS:
        for yr in sorted(d.year.unique()):
            s = d[d.year == yr]
            I, p = morans_i(s[["lat", "lon"]].values, s[task].values.astype(float))
            rows.append(dict(task=task, year=int(yr), morans_I=I, p_perm=p))
        s = d[d.year == d.year.min()]
        xy = np.radians(s[["lat", "lon"]].values); z = s[task].values.astype(float)
        i, j = np.triu_indices(len(s), 1)
        dist = 6371 * 2 * np.arcsin(np.sqrt(np.sin((xy[i, 0] - xy[j, 0]) / 2) ** 2 + np.cos(xy[i, 0]) * np.cos(xy[j, 0]) * np.sin((xy[i, 1] - xy[j, 1]) / 2) ** 2))
        g = 0.5 * (z[i] - z[j]) ** 2
        bins = np.array([0, 1, 2, 5, 10, 20, 40, 80, 160, 1e9])
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (dist >= lo) & (dist < hi)
            if m.sum(): vg.append(dict(task=task, bin_lo_km=lo, bin_hi_km=hi, n_pairs=int(m.sum()), semivariance=float(g[m].mean())))
    pd.DataFrame(rows).to_csv(f"{out}/spatial_dependence.csv", index=False)
    pd.DataFrame(vg).to_csv(f"{out}/semivariogram.csv", index=False)
    d.groupby("px").pid.nunique().describe().to_frame("points_per_pixel").to_csv(f"{out}/pixel_summary.csv")


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--out", default="results")
    ap.add_argument("--trees", type=int, default=300); ap.add_argument("--reps", type=int, default=5,
                    help="fold-assignment seeds for P1/P2/pixel-grouped CV")
    ap.add_argument("--skip", default="", help="comma list: ladder,controls,temporal,decision,spatial")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    skip = set(a.skip.split(",")); d = load(a.csv); t0 = time.time()
    print(f"{d.pid.nunique()} points, {len(d)} rows; irr mean {d.irr.mean():.3f}; road {d.road.mean():.3f}")
    if "spatial" not in skip:  spatial_dependence(d, a.out)
    if "ladder" not in skip:   ladder_and_deltas(d, a.trees, a.reps, a.out)
    if "controls" not in skip: controls(d, a.trees, a.out)
    if "temporal" not in skip: temporal(d, a.trees, a.out)
    if "decision" not in skip: decision_metrics(d, a.trees, a.out)
    print(f"done in {time.time()-t0:.0f}s")
