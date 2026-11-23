# Figures 2-5 and Figures S1-S2.
# Input (output/): final_numbers.json (02_main_models.py), si_numbers.json (04_supplementary.py),
#   analysis.pkl and raw2025.pkl (01_build_data.py).
# Output (output/): Fig2.png to Fig5.png, FigS1.png, FigS2.png
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

R = json.load(open("output/final_numbers.json"))
YEARS = [2012, 2019, 2025]
CW, CB = "#4C72B0", "#C44E52"
FIGS = {}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def series(r, key):
    v = np.array([R["desc"][f"{y}_{r}"][key][:3] for y in YEARS])
    return v[:, 0], v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]


# Fig 2
fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
panels = [("supporter", "A  Support donation", [("White", "supporter", "White"), ("Black", "supporter", "Black")], "o"),
          ("registered", "B  Registered", [("White", "registered", "White"), ("Black", "registered", "Black")], "o"),
          ("registered_if_support", "C  Registered, among supporters",
           [("White", "registered_if_support", "White"), ("Black", "registered_if_support", "Black")], "o"),
          (None, "D  Education comparison", [("White", "edu1_registered", "White, high school or less"),
                                            ("Black", "edu4_registered", "Black, postgraduate")], "s")]
for i, (_, title, ser, mk) in enumerate(panels):
    for (r, key, lab), c, dx in zip(ser, (CW, CB), (-0.08, 0.08)):
        m, lo, hi = series(r, key)
        ax[i].errorbar(np.arange(3) + (dx if i == 3 else 0), m, yerr=[lo, hi], color=c, marker=mk, ms=7, lw=1.6, capsize=3, label=lab)
    ax[i].set_xticks(range(3), YEARS); ax[i].set_xlim(-0.3, 2.3); ax[i].set_ylim(0, 104); ax[i].set_yticks(range(0, 101, 20))
    ax[i].set_title(title, loc="left", pad=22); ax[i].set_ylabel("%")
ax[0].legend(frameon=False, loc="lower left")
ax[3].legend(frameon=False, loc="lower left", fontsize=7)
fig.tight_layout(); FIGS["Fig2"] = fig

# Fig 3
d = pd.read_pickle("output/analysis.pkl")
r25 = pd.read_pickle("output/raw2025.pkl")
M = {"Strongly Agree": 4, "Somewhat Agree": 3, "Somewhat Disagree": 2, "Strongly Disagree": 1}


def item(code):
    c = [c for c in r25.columns if str(c).startswith(code + ".") and str(c).endswith("Agreement")][0]
    return r25[c].map(M).values


d25 = d[d.year == 2025].reset_index(drop=True)
labels = {"Q12A": ("Body whole at burial", 1), "Q12J": ("Loved one disfigured", 1), "Q12R": ("No regular funeral", 1),
          "Q12L": ("Doctors less likely to save donor", 1), "Q12P": ("Doctors do not do everything", -1),
          "Q12N": ("System not fair", -1), "Q12O": ("Poor have less chance", -1), "Q12I": ("Minorities less likely", 1)}
for code, (lab, sgn) in labels.items():
    v = item(code)
    d25[lab] = v if sgn == 1 else 5 - v
s = d25[d25.race_nh.isin(["White", "Black"])]
cols = [v[0] for v in labels.values()]
C = np.zeros((8, 8))
for i, a in enumerate(cols):
    for j, b in enumerate(cols):
        if i == j:
            C[i, j] = 1.0
            continue
        m = s[[a, b, "wt"]].dropna()
        cv = np.cov(m[a], m[b], aweights=m.wt)
        C[i, j] = cv[0, 1] / np.sqrt(cv[0, 0] * cv[1, 1])
# Panel B: items ordered by component
order = ["Doctors less likely to save donor", "Body whole at burial", "Loved one disfigured", "No regular funeral",
         "System not fair", "Poor have less chance", "Doctors do not do everything", "Minorities less likely"]
idx = [cols.index(c) for c in order]
C = C[np.ix_(idx, idx)]
CBOD, CSYS = "#6a3d9a", "#9a9a9a"
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.7})
a = ax[0]
xs, ticks = [], []
for k, r in enumerate(["White", "Black"]):
    for j, y in enumerate(YEARS):
        x = k * 3.6 + j
        rb = (R["corr"][f"{y}_{r}"]["doc4|burial4"] + R["corr"][f"{y}_{r}"]["doc4|disfig4"]) / 2
        rs = R["corr"][f"{y}_{r}"]["doc4|unfair"]
        a.bar(x - 0.19, rb, 0.36, color=CBOD, label="with bodily integrity items (mean)" if x == 0 else None)
        a.bar(x + 0.19, rs, 0.36, color=CSYS, label="with system distrust" if x == 0 else None)
        ticks.append((x, str(y)))
    a.text(k * 3.6 + 1, -0.25, r, ha="center", va="top", fontsize=10, fontweight="bold")
a.axhline(0, color="#777777", lw=0.8)
a.set_xticks([t[0] for t in ticks], [t[1] for t in ticks]); a.set_ylim(-0.15, 0.7)
a.set_ylabel("Weighted correlation of physician mistrust"); a.yaxis.grid(True, color="#e6e6e6"); a.set_axisbelow(True)
a.set_title("A  Physician mistrust and the other concerns, by wave", loc="left")
a.legend(frameon=False, fontsize=8, loc="upper left")
b = ax[1]
Cm = np.where(np.tril(np.ones_like(C), -1) == 1, C, np.nan)
im = b.imshow(Cm, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
for i in range(8):
    for j in range(i):
        b.text(j, i, f"{C[i, j]:.2f}".replace("-", "\u2212"), ha="center", va="center", fontsize=6.5,
               color="white" if abs(C[i, j]) > 0.4 else "#222222")
b.set_xticks(range(7), order[:7], rotation=45, ha="right", fontsize=7.5); b.set_yticks(range(1, 8), order[1:], fontsize=7.5)
b.set_xlim(-0.5, 6.5); b.set_ylim(7.5, 0.5); b.tick_params(length=0)
for sp in b.spines.values():
    sp.set_visible(False)
for lo, hi in [(0, 4), (4, 7)]:
    b.add_patch(Rectangle((lo - 0.5, lo + 0.5), hi - lo - 1, hi - lo - 1, fill=False, lw=1.4, ec="#222222"))
b.set_title("B  Item correlations, 2025", loc="left"); fig.colorbar(im, ax=b, shrink=0.7, label="r")
FIGS["Fig3"] = fig

# Fig 4
fig, ax = plt.subplots(figsize=(7.5, 3.4))
waves = ["2012", "2019", "2025", "pooled"]
x = np.arange(4)
for key, dx, c, lab in [("bodyint", -0.17, "#6a3d9a", "Bodily integrity concerns"),
                        ("mistrust_total", 0.17, "#9a9a9a", "Physician mistrust + system distrust")]:
    est = np.array([100 * R[f"khb_{w}"][key]["est"] for w in waves])
    lo = np.array([100 * R[f"khb_{w}"][key]["ci"][0] for w in waves])
    hi = np.array([100 * R[f"khb_{w}"][key]["ci"][1] for w in waves])
    ax.bar(x + dx, est, 0.32, color=c, label=lab)
    ax.errorbar(x + dx, est, yerr=[est - lo, hi - est], fmt="none", ecolor="#333333", capsize=2)
d = R["pooled_body_minus_mistrust"]
ax.annotate(f"Difference {100 * d['est']:.1f} points\n(95% CI {100 * d['ci'][0]:.1f}\u2013{100 * d['ci'][1]:.1f})",
            (3, 40), ha="center", fontsize=8)
ax.axhline(0, color="#777777", lw=0.8); ax.axvline(2.5, color="#cccccc", lw=0.8, ls="--")
ax.set_xticks(x, ["2012", "2019", "2025", "Pooled"]); ax.set_ylabel("% of adjusted Black\u2013White gap")
ax.set_ylim(-2, 68); ax.yaxis.grid(True, color="#e6e6e6"); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2)
FIGS["Fig4"] = fig

# Fig 5
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw={"width_ratios": [1.2, 0.55, 0.9], "wspace": 0.4})
ZC = ["z_bodyint", "z_doc4", "z_unfair"]
a = ax[0]; x = np.arange(3)
for r, c, dx in [("White", CW, -0.19), ("Black", CB, 0.19)]:
    q = R[f"logit_optout_{r}"]
    p = np.array([[100 * (q[k]["or"] - 1), 100 * (q[k]["ci"][0] - 1), 100 * (q[k]["ci"][1] - 1)] for k in ZC])
    a.bar(x + dx, p[:, 0], 0.36, color=c, label=r)
    a.errorbar(x + dx, p[:, 0], yerr=[p[:, 0] - p[:, 1], p[:, 2] - p[:, 0]], fmt="none", ecolor="#333333", capsize=3)
    for xi, v, h in zip(x + dx, p[:, 0], p[:, 2]):
        a.text(xi, h + 4, f"+{v:.0f}%", ha="center", fontsize=8)
a.axhline(0, color="#777777", lw=0.8)
a.set_xticks(x, ["Bodily integrity\nconcerns", "Physician\nmistrust", "System\ndistrust"]); a.set_ylim(0, 170)
a.set_ylabel("Change in odds of opting out per SD (%)"); a.yaxis.grid(True, color="#e6e6e6"); a.set_axisbelow(True)
a.legend(frameon=False, fontsize=8, loc="upper right"); a.set_title("A  Concerns and the intention to opt out", loc="left")
b = ax[1]
for i, (r, c) in enumerate([("White", CW), ("Black", CB)]):
    m, lo, hi = R["default"][f"{r}_share_bodyint_high"][:3]
    b.bar(i, m, 0.6, color=c)
    b.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#333333", capsize=3)
    b.text(i, hi + 2, f"{m:.0f}%", ha="center", fontsize=8.5)
b.set_xticks([0, 1], ["White", "Black"]); b.set_ylim(0, 80); b.set_xlim(-0.6, 1.6)
b.set_ylabel("% of non-registrants with high\nbodily integrity concern"); b.yaxis.grid(True, color="#e6e6e6"); b.set_axisbelow(True)
b.set_title("B  High bodily integrity concern", loc="left")
c3 = ax[2]
for r, c in [("White", CW), ("Black", CB)]:
    lo_, hi_ = R["default"][f"{r}_bodyint_low"], R["default"][f"{r}_bodyint_high"]
    c3.errorbar([0, 1], [lo_[0], hi_[0]], yerr=[[lo_[0] - lo_[1], hi_[0] - hi_[1]], [lo_[2] - lo_[0], hi_[2] - hi_[0]]],
                color=c, marker="o", ms=7, lw=1.8, capsize=3, label=r)
c3.set_xticks([0, 1], ["Low bodily\nintegrity concern", "High bodily\nintegrity concern"]); c3.set_xlim(-0.35, 1.35)
c3.set_ylim(0, 80); c3.set_ylabel("% of non-registrants who would opt out"); c3.yaxis.grid(True, color="#e6e6e6")
c3.set_axisbelow(True); c3.legend(frameon=False, fontsize=8, loc="lower right")
c3.set_title("C  Non-registrants who would opt out", loc="left")
FIGS["Fig5"] = fig

# Fig S1 and Fig S2
S = json.load(open("output/si_numbers.json"))


def pts(cells):
    m = np.array([q[0] if q[0] is not None else np.nan for q in cells])
    lo = np.array([q[0] - q[1] if q[0] is not None else 0 for q in cells])
    hi = np.array([q[2] - q[0] if q[0] is not None else 0 for q in cells])
    return m, lo, hi


fig, ax = plt.subplots(2, 3, figsize=(12, 6.6))
for row, (var, labs) in enumerate([("edu", ["≤HS", "Some coll.", "BA", "Postgrad"]),
                                   ("inc4", ["<$20k", "$20–40k", "$40–75k", "≥$75k"])]):
    for j, y in enumerate(YEARS):
        a = ax[row, j]
        for r, c in [("White", CW), ("Black", CB)]:
            m, lo, hi = pts([S["grid"][f"{var}{k}_{y}_{r}"] for k in [1, 2, 3, 4]])
            a.errorbar(np.arange(4), m, yerr=[lo, hi], color=c, marker="o", ms=7, lw=1.6, capsize=3, label=r)
        a.set_xticks(range(4), labs); a.set_xlim(-0.3, 3.3); a.set_ylim(0, 104); a.set_yticks(range(0, 101, 20))
        a.set_title(f"{'AB'[row]}{j+1}  {'Education' if var == 'edu' else 'Household income'}, {y}", loc="left", pad=12)
        a.set_ylabel("% registered")
ax[0, 0].legend(frameon=False, loc="lower right")
fig.tight_layout(); FIGS["FigS1"] = fig

fig, ax = plt.subplots(2, 3, figsize=(12, 6.6))
for i, (v, lab) in enumerate([("agree_doc", "Physician mistrust"), ("agree_unfair", "System distrust"),
                              ("agree_burial", "Body whole at burial"), ("agree_disfig", "Loved one disfigured"),
                              ("pc_support", "Support presumed consent"), ("optout", "Would opt out")]):
    a = ax.flat[i]
    for r, c in [("White", CW), ("Black", CB)]:
        m, lo, hi = pts([S["items"][v][f"{y}_{r}"] for y in YEARS])
        a.errorbar(np.arange(3), m, yerr=[lo, hi], color=c, marker="o", ms=7, lw=1.6, capsize=3, label=r)
    a.set_xticks(range(3), YEARS); a.set_xlim(-0.3, 2.3); a.set_ylim(0, 83); a.set_yticks(range(0, 81, 20))
    a.set_title(f"{'ABCDEF'[i]}  {lab}", loc="left", pad=12); a.set_ylabel("% agreeing")
ax[0, 0].legend(frameon=False, loc="lower right")
fig.tight_layout(); FIGS["FigS2"] = fig

# Save
for name, f in FIGS.items():
    f.savefig(f"output/{name}.png", dpi=300, bbox_inches="tight")
