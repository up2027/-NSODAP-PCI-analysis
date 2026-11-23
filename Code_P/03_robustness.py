# Robustness checks for the decomposition: linear Oaxaca-Blinder, unweighted, agree/disagree items,
# and Hispanic and Native American vs White respondents.
# Input: output/analysis.pkl from 01_build_data.py (NSODAP 2012, 2019, 2025).
# Output: output/03_robustness.xlsx
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
rng = np.random.default_rng(99)
B = 300
COMP = ["bodyint", "doc4", "unfair"]
d = pd.read_pickle("output/analysis.pkl")
d["unfair"] = 5 - d["fair4"]
d["b_body"] = ((d.burial4 >= 3).astype(float) + (d.disfig4 >= 3).astype(float)) / 2
d["b_doc"] = (d.doc4 >= 3).astype(float)
d["b_unfair"] = (d.fair4 <= 2).astype(float)
for c in ["b_body", "b_doc", "b_unfair"]:
    d.loc[d[["burial4", "disfig4", "doc4", "fair4"]].isna().any(axis=1), c] = np.nan
R, T = {}, []


def covars(s):
    x = pd.concat([pd.get_dummies(s.age.astype(int), prefix="a", drop_first=True, dtype=float),
                   pd.get_dummies(s.edu.astype(int), prefix="e", drop_first=True, dtype=float),
                   s[["female"]], pd.get_dummies(s.year, prefix="y", drop_first=True, dtype=float)], axis=1)
    return x


def khb(s, comp, focal, weighted=True):
    s = s.reset_index(drop=True)
    g = (s.race_nh == focal).astype(float).rename("g")
    X = sm.add_constant(pd.concat([g, covars(s)], axis=1))
    w = s.wt_norm.values if weighted else np.ones(len(s))
    y = s.registered.values
    res, a = {}, {}
    for m in comp:
        f = sm.WLS(s[m].values, X, weights=w).fit()
        res[m], a[m] = s[m].values - f.fittedvalues, f.params["g"]
    full = sm.GLM(y, X.assign(**{m: s[m].values for m in comp}), family=sm.families.Binomial(), var_weights=w).fit().params
    red = sm.GLM(y, X.assign(**{m: res[m] for m in comp}), family=sm.families.Binomial(), var_weights=w).fit().params
    out = {m: a[m] * full[m] / red["g"] for m in comp}
    out["explained"] = 1 - full["g"] / red["g"]
    return out


def oaxaca(s, comp):
    s = s.reset_index(drop=True)
    g = (s.race_nh == "Black").astype(float).rename("g")
    Xc = pd.concat([s[comp], covars(s)], axis=1)
    X = sm.add_constant(pd.concat([Xc, g], axis=1))
    beta = sm.WLS(s.registered, X, weights=s.wt_norm).fit().params
    w = s.wt_norm
    mb = np.average(Xc[g == 1], axis=0, weights=w[g == 1])
    mw = np.average(Xc[g == 0], axis=0, weights=w[g == 0])
    gap = np.average(s.registered[g == 1], weights=w[g == 1]) - np.average(s.registered[g == 0], weights=w[g == 0])
    contrib = (mb - mw) * beta[Xc.columns].values
    out = {m: float(contrib[list(Xc.columns).index(m)] / gap) for m in comp}
    out["covariates"] = float(sum(contrib[i] for i, c in enumerate(Xc.columns) if c not in comp) / gap)
    out["gap_pp"] = float(100 * gap)
    return out


def resample(s):
    return pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))
                      for _, g in s.groupby(["year", "race_nh"])], ignore_index=True)


def run(name, fn, s):
    e = fn(s)
    bd = pd.DataFrame([fn(resample(s)) for _ in range(B)])
    R[name] = {k: {"est": float(v), "ci": [float(bd[k].quantile(.025)), float(bd[k].quantile(.975))]} for k, v in e.items()}
    T.append(f"{name}: " + "; ".join(f"{k} {100*v['est']:.1f}% [{100*v['ci'][0]:.1f}, {100*v['ci'][1]:.1f}]" if k != "gap_pp"
                                     else f"gap {v['est']:.1f} pp" for k, v in R[name].items()))
    print(T[-1], flush=True)


req = ["registered", "age", "edu", "female"]
bw = d[d.race_nh.isin(["White", "Black"])].dropna(subset=req + COMP)
run("oaxaca_linear", lambda s: oaxaca(s, COMP), bw)
run("khb_unweighted", lambda s: khb(s, COMP, "Black", weighted=False), bw)
bwb = d[d.race_nh.isin(["White", "Black"])].dropna(subset=req + ["b_body", "b_doc", "b_unfair"])
run("khb_binary_items", lambda s: khb(s, ["b_body", "b_doc", "b_unfair"], "Black"), bwb)
for grp in ["Hispanic", "NativeAm"]:
    s = d[d.race_nh.isin(["White", grp])].dropna(subset=req + COMP)
    run(f"khb_{grp}", lambda s, grp=grp: khb(s, COMP, grp), s)
    R[f"khb_{grp}"]["n"] = int(len(s))

# Save
with pd.ExcelWriter("output/03_robustness.xlsx") as xw:
    pd.DataFrame({"result": T}).to_excel(xw, sheet_name="summary", index=False)
    pd.DataFrame(flatten(R)).rename(columns={0: "key"}).to_excel(xw, sheet_name="all_estimates", index=False)
