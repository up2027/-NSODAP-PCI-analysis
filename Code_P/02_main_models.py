# Main models for non-Hispanic Black and White respondents (weighted).
# Input: output/analysis.pkl from 01_build_data.py (NSODAP 2012, 2019, 2025).
# Output (output/): final_numbers.json (read by 06_figures.py), 02_main_results.xlsx
# Concerns: bodily integrity (mean of burial and disfigurement items), physician mistrust, system distrust; 1-4 scales.
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
rng = np.random.default_rng(2026)
B = 500
COMP = ["bodyint", "doc4", "unfair"]
d = pd.read_pickle("output/analysis.pkl")
d["unfair"] = 5 - d["fair4"]
bw = d[d.race_nh.isin(["White", "Black"])].copy()
bw["black"] = (bw.race_nh == "Black").astype(float)
base = bw.dropna(subset=["registered", "age", "edu", "female"] + COMP).reset_index(drop=True)
R, T = {}, []


def covars(s):
    x = pd.concat([pd.get_dummies(s.age.astype(int), prefix="a", drop_first=True, dtype=float),
                   pd.get_dummies(s.edu.astype(int), prefix="e", drop_first=True, dtype=float),
                   s[["female"]]], axis=1)
    if s.year.nunique() > 1:
        x = x.join(pd.get_dummies(s.year, prefix="y", drop_first=True, dtype=float))
    return x


def khb(s):
    s = s.reset_index(drop=True)
    X = sm.add_constant(pd.concat([s[["black"]], covars(s)], axis=1))
    w, y = s.wt_norm.values, s.registered.values
    res, a = {}, {}
    for m in COMP:
        f = sm.WLS(s[m].values, X, weights=w).fit()
        res[m], a[m] = s[m].values - f.fittedvalues, f.params["black"]
    full = sm.GLM(y, X.assign(**{m: s[m].values for m in COMP}), family=sm.families.Binomial(), var_weights=w).fit().params
    red = sm.GLM(y, X.assign(**{m: res[m] for m in COMP}), family=sm.families.Binomial(), var_weights=w).fit().params
    out = {m: a[m] * full[m] / red["black"] for m in COMP}
    out["mistrust_total"] = out["doc4"] + out["unfair"]
    out["explained"] = 1 - full["black"] / red["black"]
    out["total_logit"] = red["black"]
    out["direct_or"] = float(np.exp(full["black"]))
    out["total_or"] = float(np.exp(red["black"]))
    return out


def resample(s):
    return pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))
                      for _, g in s.groupby(["year", "race_nh"])], ignore_index=True)


def ci(v):
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


# 1. Decomposition by wave and pooled
T.append("1. KHB shares of the adjusted Black-White registration gap (controls: age, sex, education; wave in pooled)")
boots = {}
for y in [2012, 2019, 2025, "pooled"]:
    s = base if y == "pooled" else base[base.year == y]
    e = khb(s)
    bd = pd.DataFrame([khb(resample(s)) for _ in range(B)])
    boots[y] = bd
    R[f"khb_{y}"] = {k: {"est": float(e[k]), "ci": ci(bd[k])} for k in e} | {"n": int(len(s)),
                                                                          "n_black": int((s.black == 1).sum())}
    T.append(f"{y}: n={len(s)} total OR {e['total_or']:.2f}, direct OR {e['direct_or']:.2f}; " + "; ".join(
        f"{k} {100*e[k]:.1f}% [{100*ci(bd[k])[0]:.1f}, {100*ci(bd[k])[1]:.1f}]"
        for k in ["bodyint", "doc4", "unfair", "mistrust_total", "explained"]))
    print(T[-1], flush=True)
dd = boots["pooled"]["bodyint"] - boots["pooled"]["mistrust_total"]
e = R["khb_pooled"]
R["pooled_body_minus_mistrust"] = {"est": e["bodyint"]["est"] - e["mistrust_total"]["est"], "ci": ci(dd)}
T.append(f"pooled bodily minus mistrust (physician + system): {100*R['pooled_body_minus_mistrust']['est']:.1f} pts "
         f"[{100*ci(dd)[0]:.1f}, {100*ci(dd)[1]:.1f}]")

# 1b. Worked example, pooled decomposition (Text S3)
s = base
X = sm.add_constant(pd.concat([s[["black"]], covars(s)], axis=1))
w, y = s.wt_norm.values, s.registered.values
fam = sm.families.Binomial()
a = {m: sm.WLS(s[m].values, X, weights=w).fit().params["black"] for m in COMP}
resid = {m: s[m].values - sm.WLS(s[m].values, X, weights=w).fit().fittedvalues for m in COMP}
b_naive = sm.GLM(y, X, family=fam, var_weights=w).fit().params["black"]
b_red = sm.GLM(y, X.assign(**resid), family=fam, var_weights=w).fit().params["black"]
full = sm.GLM(y, X.assign(**{m: s[m].values for m in COMP}), family=fam, var_weights=w).fit().params
R["worked_example"] = {"race_without_concerns": float(b_naive), "race_scale_adjusted": float(b_red),
                       "race_with_concerns": float(full["black"]),
                       "race_difference_in_concern": {m: float(a[m]) for m in COMP},
                       "concern_coefficient": {m: float(full[m]) for m in COMP}}
T.append("\n1b. Worked example (pooled): race log-odds without concerns {:.3f}, scale-adjusted {:.3f}, "
         "with concerns {:.3f}; ".format(b_naive, b_red, full["black"]) + "; ".join(
    f"{m}: race difference {a[m]:.3f} x coefficient {full[m]:.3f} = {a[m]*full[m]:.3f} "
    f"({100*a[m]*full[m]/b_red:.1f}%)" for m in COMP))

# 2. Difference between the 2019 and 2025 shares
b19, b25 = boots[2019], boots[2025]
e19, e25 = R["khb_2019"], R["khb_2025"]
chg = {}
for k in ["bodyint", "mistrust_total"]:
    chg[k] = {"est": e25[k]["est"] - e19[k]["est"], "ci": ci(b25[k].values - b19[k].values)}
dd = (b25["mistrust_total"].values - b19["mistrust_total"].values) - (b25["bodyint"].values - b19["bodyint"].values)
chg["diff"] = {"est": chg["mistrust_total"]["est"] - chg["bodyint"]["est"], "ci": ci(dd)}
R["change_2019_2025"] = chg
T.append("\n2. Change in shares 2019->2025: " + "; ".join(
    f"{k} {100*v['est']:+.1f} [{100*v['ci'][0]:+.1f}, {100*v['ci'][1]:+.1f}]" for k, v in chg.items()))

# 3. Weighted descriptives
T.append("\n3. Weighted percentages (95% CI from effective n)")


def wp(x, w):
    m = x.notna()
    x, w = x[m], w[m]
    p = np.average(x, weights=w)
    ne = w.sum() ** 2 / (w ** 2).sum()
    se = np.sqrt(p * (1 - p) / ne)
    return [float(100 * p), float(100 * (p - 1.96 * se)), float(100 * (p + 1.96 * se)), int(len(x))]


desc = {}
bw["agree_doc"] = (bw.doc4 >= 3).astype(float).where(bw.doc4.notna())
bw["agree_unfair"] = (bw.fair4 <= 2).astype(float).where(bw.fair4.notna())
bw["agree_burial"] = (bw.burial4 >= 3).astype(float).where(bw.burial4.notna())
bw["agree_disfig"] = (bw.disfig4 >= 3).astype(float).where(bw.disfig4.notna())
bw["pc_support"] = (bw.pc_support4 >= 3).astype(float).where(bw.pc_support4.notna())
VARS = ["supporter", "registered", "agree_doc", "agree_unfair", "agree_burial", "agree_disfig", "pc_support", "optout"]
for y in [2012, 2019, 2025]:
    for r in ["White", "Black"]:
        s = bw[(bw.year == y) & (bw.race_nh == r)]
        desc[f"{y}_{r}"] = {v: wp(s[v], s.wt) for v in VARS}
        sup = s[s.supporter == 1]
        desc[f"{y}_{r}"]["registered_if_support"] = wp(sup.registered, sup.wt)
        desc[f"{y}_{r}"]["edu1_registered"] = wp(s[s.edu == 1].registered, s[s.edu == 1].wt)
        desc[f"{y}_{r}"]["edu4_registered"] = wp(s[s.edu == 4].registered, s[s.edu == 4].wt)
        for k in [1, 2, 3, 4]:
            desc[f"{y}_{r}"][f"inc{k}_registered"] = wp(s[s.inc4 == k].registered, s[s.inc4 == k].wt)
        desc[f"{y}_{r}"]["n"] = int(len(s))
        T.append(f"{y} {r}: " + "; ".join(f"{v} {desc[f'{y}_{r}'][v][0]:.1f}" for v in VARS + ["registered_if_support", "edu1_registered", "edu4_registered"]))
R["desc"] = desc

# 4. Logit models of opt-out and registration on the standardized concerns
T.append("\n4. Opt-out intention and registration on standardized components (weighted logit, HC1 SE)")


def z(s, v):
    m = s[v].notna()
    mu = np.average(s.loc[m, v], weights=s.loc[m, "wt"])
    sd = np.sqrt(np.average((s.loc[m, v] - mu) ** 2, weights=s.loc[m, "wt"]))
    return (s[v] - mu) / sd


for v in COMP:
    bw["z_" + v] = z(bw, v)
ZC = ["z_bodyint", "z_doc4", "z_unfair"]
for outcome in ["optout", "registered"]:
    for grp in ["White", "Black", "both"]:
        s = bw.dropna(subset=[outcome, "age", "edu", "female"] + ZC)
        if grp != "both":
            s = s[s.race_nh == grp]
        s = s.reset_index(drop=True)
        X = sm.add_constant(pd.concat([s[ZC] if grp != "both" else s[ZC + ["black"]], covars(s)], axis=1))
        f = sm.GLM(s[outcome], X, family=sm.families.Binomial(), var_weights=s.wt_norm).fit(cov_type="HC1")
        C = f.cov_params()
        d1 = f.params["z_bodyint"] - f.params["z_doc4"]
        se1 = np.sqrt(C.loc["z_bodyint", "z_bodyint"] + C.loc["z_doc4", "z_doc4"] - 2 * C.loc["z_bodyint", "z_doc4"])
        R[f"logit_{outcome}_{grp}"] = {k: {"or": float(np.exp(f.params[k])),
                                           "ci": [float(np.exp(f.conf_int().loc[k, 0])), float(np.exp(f.conf_int().loc[k, 1]))]}
                                       for k in ZC} | {"n": int(len(s)), "body_minus_doc_logor": float(d1), "z": float(d1 / se1)}
        T.append(f"{outcome} {grp} n={len(s)}: " + "; ".join(
            f"{k} OR {R[f'logit_{outcome}_{grp}'][k]['or']:.2f} [{R[f'logit_{outcome}_{grp}'][k]['ci'][0]:.2f}, {R[f'logit_{outcome}_{grp}'][k]['ci'][1]:.2f}]"
            for k in ZC) + f"; body-minus-physician log-OR {d1:.2f}, z = {d1/se1:.1f}")

# 5. Non-registrants under a default
T.append("\n5. Non-registrants: % who would opt out under presumed consent")
nr = bw[(bw.registered == 0) & bw.optout.notna()]
dflt = {}
for y in [2012, 2019, 2025, "all"]:
    s = nr if y == "all" else nr[nr.year == y]
    for r in ["White", "Black"]:
        t = s[s.race_nh == r]
        dflt[f"{y}_{r}"] = wp(t.optout, t.wt)
for r in ["White", "Black"]:
    t = nr[nr.race_nh == r].dropna(subset=["bodyint", "doc4"])
    for v, cut in [("bodyint", 2.5), ("doc4", 2.5)]:
        hi = t[v] >= cut
        dflt[f"{r}_{v}_high"] = wp(t.optout[hi], t.wt[hi])
        dflt[f"{r}_{v}_low"] = wp(t.optout[~hi], t.wt[~hi])
    dflt[f"{r}_share_bodyint_high"] = wp((t.bodyint >= 2.5).astype(float), t.wt)
R["default"] = dflt

# 5b. Black-White difference in opt-out among non-registrants
s5 = nr.dropna(subset=["age", "female", "edu", "inc4"] + COMP).reset_index(drop=True)
ses = {}
for e in [1, 2, 3, 4]:
    for r in ["White", "Black"]:
        t = s5[(s5.edu == e) & (s5.race_nh == r)]
        ses[f"edu{e}_{r}"] = wp(t.optout, t.wt)
for v in [1, 2, 3, 4]:
    for r in ["White", "Black"]:
        t = s5[(s5.inc4 == v) & (s5.race_nh == r)]
        ses[f"inc{v}_{r}"] = wp(t.optout, t.wt)
dm = lambda c: pd.get_dummies(s5[c].astype(int), prefix=c, drop_first=True, dtype=float)
base_x = [s5[["black", "female"]], dm("age"), dm("year")]
steps = [("demog", []), ("edu", [dm("edu")]), ("edu_inc", [dm("edu"), dm("inc4")]),
         ("edu_inc_body", [dm("edu"), dm("inc4"), s5[["bodyint"]]]),
         ("edu_inc_all", [dm("edu"), dm("inc4"), s5[COMP]])]
for k, extra in steps:
    X = sm.add_constant(pd.concat(base_x + extra, axis=1))
    f = sm.GLM(s5.optout, X, family=sm.families.Binomial(), var_weights=s5.wt_norm).fit(cov_type="HC1")
    ci = f.conf_int().loc["black"]
    ses[f"or_{k}"] = {"or": float(np.exp(f.params["black"])), "ci": [float(np.exp(ci[0])), float(np.exp(ci[1]))],
                      "p": float(f.pvalues["black"]), "n": int(len(s5))}
R["default_ses"] = ses
for k, v in dflt.items():
    T.append(f"{k}: {v[0]:.1f}% [{v[1]:.1f}, {v[2]:.1f}] n={v[3]}")
for k, v in ses.items():
    T.append(f"{k}: " + (f"OR {v['or']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}] p={v['p']:.3f} n={v['n']}"
                         if k.startswith("or_") else f"{v[0]:.1f}% [{v[1]:.1f}, {v[2]:.1f}] n={v[3]}"))

# 6. Weighted item correlations
T.append("\n6. Weighted item correlations, all waves")


def wcorr(s, a, b):
    m = s[[a, b, "wt"]].dropna()
    c = np.cov(m[a], m[b], aweights=m.wt)
    return float(c[0, 1] / np.sqrt(c[0, 0] * c[1, 1]))


items = ["doc4", "unfair", "burial4", "disfig4"]
corr = {}
for y in [2012, 2019, 2025]:
    for r in ["White", "Black"]:
        s = bw[(bw.year == y) & (bw.race_nh == r)]
        corr[f"{y}_{r}"] = {f"{a}|{b}": wcorr(s, a, b) for i, a in enumerate(items) for b in items[i + 1:]}
        T.append(f"{y} {r}: " + " ".join(f"{k}={v:.2f}" for k, v in corr[f"{y}_{r}"].items()))
R["corr"] = corr

# 7. Survey mode, 2019 and 2025
R["mode"] = {}
for yr in [2019, 2025]:
    sm_ = bw[bw.year == yr].dropna(subset=["registered"]).reset_index(drop=True)
    sm_["web"] = (sm_["mode"] == "web").astype(float)
    sm_["bxw"] = sm_.black * sm_.web
    f = sm.GLM(sm_.registered, sm.add_constant(sm_[["black", "web", "bxw"]]), family=sm.families.Binomial(),
               var_weights=sm_.wt_norm).fit(cov_type="HC1")
    M = {"black_or": float(np.exp(f.params["black"])), "bxw_or": float(np.exp(f.params["bxw"])), "p": float(f.pvalues["bxw"])}
    for mode in ["phone", "web"]:
        for r in ["White", "Black"]:
            t = sm_[(sm_["mode"] == mode) & (sm_.race_nh == r)]
            M[f"{mode}_{r}"] = wp(t.registered, t.wt_mode)
    R["mode"][str(yr)] = M
    T.append(f"\n7. {yr} mode: black x web OR {M['bxw_or']:.2f}, p = {M['p']:.2f}; " +
             "; ".join(f"{k} {v[0]:.1f}" for k, v in M.items() if isinstance(v, list)))

print("\n".join(T))

# Save
json.dump(R, open("output/final_numbers.json", "w"), indent=1)
with pd.ExcelWriter("output/02_main_results.xlsx") as xw:
    pd.DataFrame({"result": T}).to_excel(xw, sheet_name="summary", index=False)
    pd.DataFrame(flatten(R)).rename(columns={0: "key"}).to_excel(xw, sheet_name="all_estimates", index=False)
