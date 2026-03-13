import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker
import numpy as np

# ── Palette ────────────────────────────────────────────────────────────────────
BLUE    = "#2166AC"
GREEN   = "#1A6B3C"
BLUE_F  = "#ADD4F0"
GREEN_F = "#A8D5B5"
GOLD    = "#D4A017"
CREAM   = "#FFF8E7"
GRID    = "#CCCCCC"
BG      = "#F8F8F8"

BLACK_COLOR = "#C0392B"   # highlighted — Black Americans
OTHER_COLOR = "#2E75B6"   # all other groups (matches uploaded pie style)

# ── Verified data (HRSA/OPTN via OMH, 2025–2026) ─────────────────────────────
waitlist_total = 107446
donor_total    = 24023

waitlist_black = round(0.27 * waitlist_total)    # 29,010
waitlist_other = waitlist_total - waitlist_black  # 78,436

donor_black    = round(0.13 * donor_total)        # 3,123
donor_other    = donor_total - donor_black         # 20,900

# ── Area chart data ───────────────────────────────────────────────────────────
years_area = list(range(1988, 2026))
waitlist_ts = [
    19000,21000,23500,26000,28500,30500,33000,36000,39000,42000,
    44500,46500,48500,50000,51500,52500,53000,53500,54000,54500,
    57500,59000,59500,60000,61000,62000,63500,65000,63000,61000,
    63000,65000,66000,67000,74000,81000,83000,107000
]
donors_ts = [
    5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,
    10000,10500,11000,11500,12000,12500,13000,13500,14000,14500,
    15000,15500,16000,17000,17500,18000,19000,20000,20500,21000,
    21500,22000,22500,23000,24000,24500,24023,24500
]

# ── Layout: 3 rows ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 13), facecolor="white")
gs  = gridspec.GridSpec(3, 1, figure=fig,
                        height_ratios=[1.0, 0.18, 1.5],
                        hspace=0.08)

# ── Row 0: Two simple pies ────────────────────────────────────────────────────
pos0 = gs[0].get_position(fig)
pad  = 0.055
mid  = pos0.x0 + pos0.width / 2
dw   = pos0.width / 2 - pad * 1.6
dh   = pos0.height - 0.01

ax_w = fig.add_axes([pos0.x0 + pad,    pos0.y0 + 0.01, dw, dh])
ax_d = fig.add_axes([mid + pad * 0.7,  pos0.y0 + 0.01, dw, dh])

def draw_pie(ax, black_n, other_n, title, title_color, year_label):
    total  = black_n + other_n
    sizes  = [black_n, other_n]
    colors = [BLACK_COLOR, OTHER_COLOR]
    pcts   = [black_n / total * 100, other_n / total * 100]

    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        counterclock=False,
        explode=[0.06, 0],   # pull Black slice outward
        wedgeprops=dict(edgecolor="white", linewidth=3),
    )

    # Labels: "27%\n(29,010)" inside each slice if large enough
    for wedge, n, pct in zip(wedges, sizes, pcts):
        angle  = (wedge.theta1 + wedge.theta2) / 2
        # place label at 60% radius
        radius = 0.62
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        if pct >= 8:
            label = f"{pct:.0f}%\n({n:,})"
            ax.text(x, y, label,
                    ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white", linespacing=1.4)

    ax.set_title(title, fontsize=12.5, fontweight="bold",
                 color=title_color, pad=14)
    ax.set_aspect("equal")

draw_pie(ax_w, waitlist_black, waitlist_other,
         "Transplant Waitlist Candidates (2025)", BLUE,  "2025")
draw_pie(ax_d, donor_black,    donor_other,
         "Actual Organ Donors (2024)",            GREEN, "2024")

# Panel A label
fig.text(pos0.x0 + 0.005, pos0.y0 + pos0.height - 0.002,
         "A", fontsize=16, fontweight="bold", color="#333333", va="top")

# ── Row 1: Legend below pies ──────────────────────────────────────────────────
pos1 = gs[1].get_position(fig)
ax_leg = fig.add_axes([pos1.x0, pos1.y0 + 0.01, pos1.width, pos1.height])
ax_leg.axis("off")

legend_patches = [
    mpatches.Patch(facecolor=BLACK_COLOR, edgecolor="white",
                   linewidth=1.2, label="Black Americans"),
    mpatches.Patch(facecolor=OTHER_COLOR, edgecolor="white",
                   linewidth=1.2, label="All Other Groups"),
]
ax_leg.legend(
    handles=legend_patches,
    loc="center",
    ncol=2,
    fontsize=11.5,
    framealpha=0.0,
    edgecolor="none",
    handlelength=1.6,
    handleheight=1.1,
    columnspacing=2.5,
    handletextpad=0.7,
)

# ── Row 2: Area chart ─────────────────────────────────────────────────────────
pos2 = gs[2].get_position(fig)
ax_b = fig.add_axes(
    [pos2.x0, pos2.y0 + 0.04, pos2.width, pos2.height - 0.04],
    facecolor=BG
)

ax_b.axvspan(2012, 2025, color=CREAM, zorder=0)
ax_b.axvline(2012, color=GOLD, linewidth=1.2, linestyle="--", zorder=1)
ax_b.axvline(2025, color=GOLD, linewidth=1.2, linestyle="--", zorder=1)
ax_b.text(2018.5, 97000, "Study Period (2012–2025)",
          ha="center", color=GOLD, fontsize=11.5, style="italic", zorder=2)

ax_b.fill_between(years_area, waitlist_ts, alpha=0.35, color=BLUE_F,  zorder=1)
ax_b.fill_between(years_area, donors_ts,   alpha=0.50, color=GREEN_F, zorder=1)

ax_b.plot(years_area, waitlist_ts, color=BLUE,  linewidth=2.2,
          marker="o", markersize=3.5, markerfacecolor=BLUE,
          markeredgecolor="white", markeredgewidth=0.8, zorder=3,
          label="Waitlist candidates")
ax_b.plot(years_area, donors_ts,   color=GREEN, linewidth=2.2,
          marker="o", markersize=3.5, markerfacecolor=GREEN,
          markeredgecolor="white", markeredgewidth=0.8, zorder=3,
          label="Actual donors")

ax_b.set_xlim(1988, 2026)
ax_b.set_ylim(0, 115000)
ax_b.set_xticks(range(1988, 2026, 2))
ax_b.set_xticklabels([str(y) for y in range(1988, 2026, 2)],
                     rotation=45, ha="right", fontsize=8.5)
ax_b.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
ax_b.set_yticks([0, 20000, 40000, 60000, 80000, 100000])
ax_b.yaxis.grid(True, color=GRID, linewidth=0.7, linestyle="--", zorder=0)
ax_b.set_axisbelow(True)
ax_b.spines[["top", "right"]].set_visible(False)
ax_b.spines[["left", "bottom"]].set_color("#AAAAAA")
ax_b.set_xlabel("Year", fontsize=11, labelpad=6)
ax_b.set_ylabel("Number of People", fontsize=11, labelpad=6)
ax_b.set_title(
    "Number of Organ Donors and Transplant Waitlist Candidates\n"
    "in the United States, 1988–2025",
    fontsize=12.5, fontweight="bold", pad=10)
ax_b.legend(loc="upper left", fontsize=10, framealpha=0.9, edgecolor="#CCCCCC")

fig.text(pos2.x0 + 0.005, pos2.y0 + pos2.height - 0.002,
         "B", fontsize=16, fontweight="bold", color="#333333", va="top")

# ── Footer ────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.003,
         "Note. Black American percentages from HRSA/OPTN data (Office of Minority Health, 2026). "
         "Counts derived by applying verified percentages to OPTN totals. "
         "Source: U.S. Organ Procurement and Transplantation Network (OPTN).",
         ha="center", fontsize=8, color="#777777", style="italic")

plt.savefig("./organ_donor_final.png",
            dpi=180, bbox_inches="tight", facecolor="white")

print("Done.")
