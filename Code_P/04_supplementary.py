# Supplementary analyses: sample characteristics, race x wave tests, education and income, nested models,
# single-concern, subgroup and survey-mode decompositions, item correlations, separate bodily items,
# control sets and E-value.
# Input: output/analysis.pkl from 01_build_data.py (NSODAP 2012, 2019, 2025).
# Output (output/): si_numbers.json (read by 06_figures.py), 04_supplementary.xlsx
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
def flatten(obj, path=""):
    if isinstance(obj, dict):
        return [r for k, v in obj.items() for r in flatten(v, f"{path}|{k}" if path else str(k))]
    if isinstance(obj, list) and obj and not isinstance(obj[0], (dict, list)):
        return [[path] + obj]
    return [[path, obj]]


warnings.filterwarnings("ignore")
rng = np.random.default_rng(31)
COMP = ["bodyint", "doc4", "unfair"]
YEARS = [2012, 2019, 2025]
d = pd.read_pickle("output/analysis.pkl")
d["unfair"] = 5 - d["fair4"]
d["agree_doc"] = (d.doc4 >= 3).astype(float).where(d.doc4.notna())
d["agree_unfair"] = (d.fair4 <= 2).astype(float).where(d.fair4.notna())
d["agree_burial"] = (d.burial4 >= 3).astype(float).where(d.burial4.notna())
d["agree_disfig"] = (d.disfig4 >= 3).astype(float).where(d.disfig4.notna())
d["pc_support"] = (d.pc_support4 >= 3).astype(float).where(d.pc_support4.notna())
bw = d[d.race_nh.isin(["White", "Black"])].copy()
bw["black"] = (bw.race_nh == "Black").astype(float)
S = {}


def wp(x, w):
    m = x.notna()
    x, w = x[m], w[m]
    if len(x) == 0:
        return [None, None, None, 0]
    p = np.average(x, weights=w)
    ne = w.sum() ** 2 / (w ** 2).sum()
    se = np.sqrt(p * (1 - p) / ne)
    return [float(100 * p), float(100 * (p - 1.96 * se)), float(100 * (p + 1.96 * se)), int(len(x))]


# 1. Sample characteristics (unweighted n and %, all respondents)
samp = {}
labels = {"race": {"White": "White", "Black": "Black", "NativeAm": "Native American", "Asian": "Asian", "Other": "Other or multiple"},
          "hispanic": {1.0: "Hispanic ethnicity"},
          "female": {0.0: "Male", 1.0: "Female"},
          "age": {1: "18–34", 2: "35–49", 3: "50–64", 4: "65+"},
          "edu": {1: "High school or less", 2: "Some college or trade school", 3: "Bachelor's degree", 4: "Postgraduate"},
          "registered": {1.0: "Registered"}, "supporter": {1.0: "Supports donation"}}
for y in YEARS:
    s = d[d.year == y]
    col = {"N": int(len(s))}
    for v, mp in labels.items():
        for code, lab in mp.items():
            n = int((s[v] == code).sum())
            col[f"{v}|{lab}"] = [n, 100 * n / len(s)]
    col["mode"] = {"2012": "Telephone", "2019": "Telephone 2,000; web 8,000", "2025": "Telephone 2,001; web 8,007"}[str(y)]
    samp[y] = col
S["sample"] = samp


# 2. Race x wave tests (weighted logit, HC1)
def rxw(v):
    s = bw.dropna(subset=[v]).reset_index(drop=True)
    X = pd.DataFrame({"black": s.black})
    for y in [2019, 2025]:
        X[f"w{y}"] = (s.year == y).astype(float)
        X[f"bx{y}"] = s.black * X[f"w{y}"]
    f = sm.GLM(s[v], sm.add_constant(X), family=sm.families.Binomial(), var_weights=s.wt_norm).fit(cov_type="HC1")
    W = f.wald_test("bx2019 = 0, bx2025 = 0", scalar=True)
    return {"bx2019": [float(np.exp(f.params["bx2019"])), float(f.pvalues["bx2019"])],
            "bx2025": [float(np.exp(f.params["bx2025"])), float(f.pvalues["bx2025"])],
            "joint_chi2": float(W.statistic), "joint_p": float(W.pvalue)}


S["rxw"] = {v: rxw(v) for v in ["supporter", "registered", "agree_doc", "agree_unfair", "agree_burial", "agree_disfig", "pc_support", "optout"]}

# 3. Item percentages and gaps (weighted)
items = {}
for v in ["supporter", "registered", "agree_doc", "agree_unfair", "agree_burial", "agree_disfig", "pc_support", "optout"]:
    items[v] = {f"{y}_{r}": wp(bw[(bw.year == y) & (bw.race_nh == r)][v], bw[(bw.year == y) & (bw.race_nh == r)].wt)
                for y in YEARS for r in ["White", "Black"]}
S["items"] = items

# 4. Education and income
grid = {}
for var, levels in [("edu", [1, 2, 3, 4]), ("inc4", [1, 2, 3, 4])]:
    for k in levels:
        for y in YEARS:
            for r in ["White", "Black"]:
                s = bw[(bw.year == y) & (bw.race_nh == r) & (bw[var] == k)]
                grid[f"{var}{k}_{y}_{r}"] = wp(s.registered, s.wt)
S["grid"] = grid
att = {}
for y in YEARS:
    for var in ["edu", "inc4"]:
        s = bw[bw.year == y].dropna(subset=["registered", var]).reset_index(drop=True)
        f0 = sm.GLM(s.registered, sm.add_constant(s[["black"]]), family=sm.families.Binomial(), var_weights=s.wt_norm).fit()
        X1 = sm.add_constant(pd.concat([s[["black"]], pd.get_dummies(s[var].astype(int), prefix="k", drop_first=True, dtype=float)], axis=1))
        f1 = sm.GLM(s.registered, X1, family=sm.families.Binomial(), var_weights=s.wt_norm).fit()
        b0, b1 = f0.params["black"], f1.params["black"]
        att[f"{y}_{var}"] = {"or_race": float(np.exp(b0)), "or_adj": float(np.exp(b1)), "attenuation": float(100 * (b0 - b1) / b0), "n": int(len(s))}
S["attenuation"] = att


# 5. Nested logit models (pooled)
def covars(s, wave=True):
    x = pd.concat([pd.get_dummies(s.age.astype(int), prefix="a", drop_first=True, dtype=float),
                   pd.get_dummies(s.edu.astype(int), prefix="e", drop_first=True, dtype=float), s[["female"]]], axis=1)
    if wave and s.year.nunique() > 1:
        x = x.join(pd.get_dummies(s.year, prefix="y", drop_first=True, dtype=float))
    return x


base = bw.dropna(subset=["registered", "age", "edu", "female"] + COMP).reset_index(drop=True)
wave = pd.get_dummies(base.year, prefix="y", drop_first=True, dtype=float)
specs = {"M1 Race + wave": [], "M2 + age, sex, education": ["dem"], "M3 + bodily integrity": ["dem", "bodyint"],
         "M4 + physician mistrust": ["dem", "doc4"], "M5 + system distrust": ["dem", "unfair"], "M6 + all three concerns": ["dem"] + COMP}
nested = {}
for name, parts in specs.items():
    X = pd.concat([base[["black"]], wave], axis=1)
    if "dem" in parts:
        X = pd.concat([base[["black"]], covars(base)], axis=1)
    for p in parts:
        if p != "dem":
            X[p] = base[p]
    f = sm.GLM(base.registered, sm.add_constant(X), family=sm.families.Binomial(), var_weights=base.wt_norm).fit(cov_type="HC1")
    ciL = f.conf_int().loc["black"]
    nested[name] = {"black_or": float(np.exp(f.params["black"])), "ci": [float(np.exp(ciL[0])), float(np.exp(ciL[1]))],
                    "concerns": {p: float(np.exp(f.params[p])) for p in parts if p != "dem"}}
S["nested"] = nested
S["nested_n"] = int(len(base))


# 6. Single-concern versus joint KHB; subgroup KHB -Karlson–Holm–Breen decomposition
def khb(s, comp):
    s = s.reset_index(drop=True)
    X = sm.add_constant(pd.concat([s[["black"]], covars(s)], axis=1))
    X = X.loc[:, X.std() > 0].assign(const=1.0)
    w, y = s.wt_norm.values, s.registered.values
    res, a = {}, {}
    for m in comp:
        f = sm.WLS(s[m].values, X, weights=w).fit()
        res[m], a[m] = s[m].values - f.fittedvalues, f.params["black"]
    full = sm.GLM(y, X.assign(**{m: s[m].values for m in comp}), family=sm.families.Binomial(), var_weights=w).fit().params
    red = sm.GLM(y, X.assign(**{m: res[m] for m in comp}), family=sm.families.Binomial(), var_weights=w).fit().params
    out = {m: a[m] * full[m] / red["black"] for m in comp}
    out["explained"] = 1 - full["black"] / red["black"]
    return out


def resample(s):
    return pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9))) for _, g in s.groupby(["year", "race_nh"])], ignore_index=True)


def boot(s, comp, B):
    e = khb(s, comp)
    bd = pd.DataFrame([khb(resample(s), comp) for _ in range(B)])
    return {k: {"est": float(v), "ci": [float(bd[k].quantile(.025)), float(bd[k].quantile(.975))]} for k, v in e.items()}


S["alone"] = {m: boot(base, [m], 300) for m in COMP}
sub = {}
for var, levels in [("edu", {1: "High school or less", 2: "Some college", 3: "Bachelor's", 4: "Postgraduate"}),
                    ("age", {1: "18–34", 2: "35–49", 3: "50–64", 4: "65+"}), ("female", {0.0: "Male", 1.0: "Female"})]:
    for k, lab in levels.items():
        s = base[base[var] == k]
        r = boot(s, COMP, 200)
        r["n"] = int(len(s))
        sub[f"{var}|{lab}"] = r
        print(var, lab, f"{100*r['bodyint']['est']:.1f}", flush=True)
S["subgroup"] = sub


# 7. Decomposition and survey mode
def khb_mode(s, comp):
    s = s.reset_index(drop=True)
    X = sm.add_constant(pd.concat([s[["black"]], covars(s)], axis=1))
    if s["mode"].nunique() > 1:
        X["web"] = (s["mode"] == "web").astype(float)
    X = X.loc[:, X.std() > 0].assign(const=1.0)
    w, y = s.wt_norm.values, s.registered.values
    res, a = {}, {}
    for m in comp:
        f = sm.WLS(s[m].values, X, weights=w).fit()
        res[m], a[m] = s[m].values - f.fittedvalues, f.params["black"]
    full = sm.GLM(y, X.assign(**{m: s[m].values for m in comp}), family=sm.families.Binomial(), var_weights=w).fit().params
    red = sm.GLM(y, X.assign(**{m: res[m] for m in comp}), family=sm.families.Binomial(), var_weights=w).fit().params
    out = {m: a[m] * full[m] / red["black"] for m in comp}
    out["explained"] = 1 - full["black"] / red["black"]
    return out


def boot_mode(s, B=200):
    e = khb_mode(s, COMP)
    bd = pd.DataFrame([khb_mode(resample(s), COMP) for _ in range(B)])
    return {k: {"est": float(v), "ci": [float(bd[k].quantile(.025)), float(bd[k].quantile(.975))]} for k, v in e.items()} | {"n": int(len(s))}


S["mode_decomp"] = {"pooled, mode controlled": boot_mode(base)}
for yr in [2019, 2025]:
    for md in ["phone", "web"]:
        S["mode_decomp"][f"{yr} {md}"] = boot_mode(base[(base.year == yr) & (base["mode"] == md)])
        print(yr, md, f"{100*S['mode_decomp'][f'{yr} {md}']['bodyint']['est']:.1f}", flush=True)


# 8. Equality and order of item correlations
def wcorr_n(s, a, b):
    m = s[[a, b, "wt"]].dropna()
    c = np.cov(m[a], m[b], aweights=m.wt)
    ne = m.wt.sum() ** 2 / (m.wt ** 2).sum()
    return float(c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])), float(ne)


def q_test(cells):
    z = np.array([np.arctanh(r) for r, n in cells])
    w = np.array([n - 3 for r, n in cells])
    zbar = np.sum(w * z) / np.sum(w)
    q = float(np.sum(w * (z - zbar) ** 2))
    from scipy.stats import chi2
    return {"chi2": q, "df": len(cells) - 1, "p": float(chi2.sf(q, len(cells) - 1))}


PAIRS = {"burial|disfig": ("burial4", "disfig4"), "doc|burial": ("doc4", "burial4"), "doc|disfig": ("doc4", "disfig4"), "doc|unfair": ("doc4", "unfair")}
eq = {}
for k, (a, b) in PAIRS.items():
    cells = {(y, r): wcorr_n(bw[(bw.year == y) & (bw.race_nh == r)], a, b) for y in YEARS for r in ["White", "Black"]}
    eq[k] = {"all six groups": q_test(list(cells.values()))}
    for r in ["White", "Black"]:
        eq[k][f"across waves, {r}"] = q_test([cells[(y, r)] for y in YEARS])
    for y in YEARS:
        eq[k][f"across race, {y}"] = q_test([cells[(y, r)] for r in ["White", "Black"]])
S["corr_equality"] = eq
order = {}
for y in YEARS:
    for r in ["White", "Black"]:
        g = bw[(bw.year == y) & (bw.race_nh == r)].dropna(subset=["doc4", "burial4", "disfig4", "unfair"]).reset_index(drop=True)

        def diff(t):
            rb = (wcorr_n(t, "doc4", "burial4")[0] + wcorr_n(t, "doc4", "disfig4")[0]) / 2
            return rb - wcorr_n(t, "doc4", "unfair")[0]
        e = diff(g)
        bd = [diff(g.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))) for _ in range(500)]
        order[f"{y}_{r}"] = {"est": float(e), "ci": [float(np.quantile(bd, .025)), float(np.quantile(bd, .975))]}
S["corr_order"] = order

# 9. Decomposition with burial and disfigurement entered separately (pooled)
S["items_separate"] = {k: v for k, v in boot(base, ["burial4", "disfig4", "doc4", "unfair"], 300).items()}
print("items separate", {k: round(100 * v["est"], 1) for k, v in S["items_separate"].items()}, flush=True)

# 10. Control sets and E-value
def khb_ctrl(s, comp, ctrl):
    s = s.reset_index(drop=True)
    parts = [s[["black"]], pd.get_dummies(s.year, prefix="y", drop_first=True, dtype=float)]
    if "age" in ctrl:
        parts.append(pd.get_dummies(s.age.astype(int), prefix="a", drop_first=True, dtype=float))
    if "female" in ctrl:
        parts.append(s[["female"]])
    if "edu" in ctrl:
        parts.append(pd.get_dummies(s.edu.astype(int), prefix="e", drop_first=True, dtype=float))
    if "inc4" in ctrl:
        parts.append(pd.get_dummies(s.inc4.astype(int), prefix="i", drop_first=True, dtype=float))
    X = sm.add_constant(pd.concat(parts, axis=1))
    w, y = s.wt_norm.values, s.registered.values
    res, a = {}, {}
    for m in comp:
        f = sm.WLS(s[m].values, X, weights=w).fit()
        res[m], a[m] = s[m].values - f.fittedvalues, f.params["black"]
    full = sm.GLM(y, X.assign(**{m: s[m].values for m in comp}), family=sm.families.Binomial(), var_weights=w).fit().params
    red = sm.GLM(y, X.assign(**{m: res[m] for m in comp}), family=sm.families.Binomial(), var_weights=w).fit().params
    out = {m: a[m] * full[m] / red["black"] for m in comp}
    out["explained"] = 1 - full["black"] / red["black"]
    return out


CTRL = {"Wave only": [], "Wave, age, sex": ["age", "female"], "Wave, age, sex, education (main)": ["age", "female", "edu"],
        "Main plus household income": ["age", "female", "edu", "inc4"]}
cs = {}
for lab, ctrl in CTRL.items():
    s = base.dropna(subset=["inc4"]) if "inc4" in ctrl else base
    e = khb_ctrl(s, COMP, ctrl)
    bd = pd.DataFrame([khb_ctrl(resample(s), COMP, ctrl) for _ in range(200)])
    cs[lab] = {k: {"est": float(v), "ci": [float(bd[k].quantile(.025)), float(bd[k].quantile(.975))]} for k, v in e.items()} | {"n": int(len(s))}
    print(lab, f"{100*cs[lab]['bodyint']['est']:.1f}", flush=True)
S["control_sets"] = cs


def evalue_or(o):
    o = 1 / o if o < 1 else o
    rr = np.sqrt(o)
    return float(rr + np.sqrt(rr * (rr - 1)))


zb = base.copy()
for v in COMP:
    m = zb[v].notna()
    mu = np.average(zb.loc[m, v], weights=zb.loc[m, "wt"])
    sd = np.sqrt(np.average((zb.loc[m, v] - mu) ** 2, weights=zb.loc[m, "wt"]))
    zb["z_" + v] = (zb[v] - mu) / sd
X = sm.add_constant(pd.concat([zb[["black", "z_bodyint", "z_doc4", "z_unfair"]], covars(zb)], axis=1))
f = sm.GLM(zb.registered, X, family=sm.families.Binomial(), var_weights=zb.wt_norm).fit(cov_type="HC1")
ob, ci_b = float(np.exp(f.params["z_bodyint"])), np.exp(f.conf_int().loc["z_bodyint"]).values
ecol = [c for c in X.columns if c.startswith("e_")]
edu_or = {c: float(np.exp(f.params[c])) for c in ecol}
S["evalue"] = {"bodyint_or": ob, "bodyint_ci": [float(ci_b[0]), float(ci_b[1])],
               "evalue_point": evalue_or(ob), "evalue_ci": evalue_or(float(ci_b[1])),
               "edu_or": edu_or, "edu_rr_max": float(max(np.sqrt(v) if v > 1 else np.sqrt(1 / v) for v in edu_or.values())),
               "female_rr": float(np.sqrt(np.exp(abs(f.params["female"]))))}
print("evalue", S["evalue"], flush=True)

# Save
json.dump(S, open("output/si_numbers.json", "w"), indent=1)
pd.DataFrame(flatten(S)).rename(columns={0: "key"}).to_excel("output/04_supplementary.xlsx", sheet_name="all_estimates", index=False)
