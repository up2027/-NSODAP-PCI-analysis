# Eight concern items in the 2025 wave: correlations, principal components and reliability (Table S8).
# Input: output/raw2025.pkl and output/analysis.pkl from 01_build_data.py (NSODAP 2025 public-use file).
# Output: output/05_items_2025.xlsx
import numpy as np, pandas as pd
M = {"Strongly Agree": 4, "Somewhat Agree": 3, "Somewhat Disagree": 2, "Strongly Disagree": 1}
r = pd.read_pickle("output/raw2025.pkl")
d = pd.read_pickle("output/analysis.pkl")
d = d[d.year == 2025].reset_index(drop=True)
assert len(d) == len(r)


def item(code):
    c = [c for c in r.columns if str(c).startswith(code + ".") and str(c).endswith("Agreement")][0]
    return r[c].map(M).values


ITEMS = {  # higher values = more concern
    "doc_less": item("Q12L"), "doc_not_all": 5 - item("Q12P"), "sys_unfair": 5 - item("Q12N"),
    "poor_less": 5 - item("Q12O"), "minority_less": item("Q12I"),
    "burial": item("Q12A"), "disfig": item("Q12J"), "no_funeral": item("Q12R"),
}
for k, v in ITEMS.items():
    d[k] = v
out = []
bw = d[d.race_nh.isin(["White", "Black"])].copy()
names = list(ITEMS)
C = pd.DataFrame({a: {b: bw[[a, b]].dropna().corr().iloc[0, 1] for b in names} for a in names})
out.append("Item correlations, 2025, Black + White (unweighted Pearson):\n" + C.round(2).to_string())

# Principal components with varimax rotation
ev, vec = np.linalg.eigh(C.values)
out.append("eigenvalues: " + " ".join(f"{e:.2f}" for e in ev[::-1]))
L = vec[:, ::-1][:, :3] * np.sqrt(ev[::-1][:3])


def varimax(P, it=100):
    p, k = P.shape; R = np.eye(k)
    for _ in range(it):
        Lr = P @ R
        u, s, vt = np.linalg.svd(P.T @ (Lr ** 3 - Lr @ np.diag((Lr ** 2).sum(0)) / p))
        R = u @ vt
    return P @ R


LOAD = pd.DataFrame(varimax(L), index=names, columns=["PC1", "PC2", "PC3"])
LOAD = LOAD * np.sign(LOAD.loc[LOAD.abs().idxmax()].values.diagonal())  # largest loading positive
out.append("3-component varimax loadings:\n" + LOAD.round(2).to_string())


def alpha(cols):
    x = bw[cols].dropna(); k = len(cols)
    return k / (k - 1) * (1 - x.var().sum() / x.sum(axis=1).var())


out.append(f"alpha burial, disfig, no_funeral: {alpha(['burial', 'disfig', 'no_funeral']):.2f}")
print("\n".join(out))

# Save
with pd.ExcelWriter("output/05_items_2025.xlsx") as xw:
    C.to_excel(xw, sheet_name="correlations")
    LOAD.to_excel(xw, sheet_name="loadings")
    pd.DataFrame({"eigenvalue": ev[::-1]}).to_excel(xw, sheet_name="eigenvalues", index=False)
    pd.DataFrame({"result": out[2:]}).to_excel(xw, sheet_name="summary", index=False)
