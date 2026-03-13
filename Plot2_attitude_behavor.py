"""
figure_main_option1_final.py
────────────────────────────
Reproduces Figure 1 (main manuscript, White vs. Black only) in the
"Classic Academic" style (Option 1).

Data source: nsodap_s1_stats_full.csv
All prevalence estimates, confidence intervals, and gap values are
read directly from the CSV — nothing is hard-coded from the old figure.

Output:
  figure_main_manuscript_final.pdf   (vector, print-ready)
  figure_main_manuscript_final.png   (300 dpi raster)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.lines import Line2D

# ── 0. Global style ───────────────────────────────────────────────────────────
rcParams['font.family']      = 'serif'
rcParams['font.serif']       = ['Georgia', 'Times New Roman', 'DejaVu Serif']
rcParams['axes.spines.top']  = False
rcParams['axes.spines.right']= False
rcParams['xtick.direction']  = 'out'
rcParams['ytick.direction']  = 'out'

# ── 1. Load & filter data ─────────────────────────────────────────────────────
CSV_PATH = 'nsodap_s1_stats_full.csv'   # adjust path if needed
df = pd.read_csv(CSV_PATH)

WAVES      = [2012, 2019, 2025]
WAVES_STR  = ['2012', '2019', '2025']
RACES_MAIN = ['White', 'Black']

# — Prevalence rows (pct + 95 % CI) ──────────────────────────────────────────
prev = (
    df[(df['section'] == 'prevalence') &
       (df['race'].isin(RACES_MAIN)) &
       (df['wave'].isin(WAVES_STR))]
    [['metric', 'race', 'wave', 'pct', 'ci_lo', 'ci_hi']]
    .copy()
)
prev['wave'] = prev['wave'].astype(int)

def get_prev(metric, race):
    sub = prev[(prev['metric'] == metric) & (prev['race'] == race)].sort_values('wave')
    return (
        sub['pct'].tolist(),
        sub['ci_lo'].tolist(),
        sub['ci_hi'].tolist(),
    )

# — Racial-gap rows (Black vs. White) ─────────────────────────────────────────
gaps = (
    df[(df['section'] == 'racial_gap') &
       (df['race'] == 'Black') &
       (df['wave'].isin(WAVES_STR))]
    [['metric', 'wave', 'gap']]
    .copy()
)
gaps['wave'] = gaps['wave'].astype(int)

def get_gap(metric):
    return (
        gaps[gaps['metric'] == metric]
        .sort_values('wave')['gap']
        .tolist()
    )

# Pull values
sup_white_pct, sup_white_lo, sup_white_hi = get_prev('support',    'White')
sup_black_pct, sup_black_lo, sup_black_hi = get_prev('support',    'Black')
reg_white_pct, reg_white_lo, reg_white_hi = get_prev('registered', 'White')
reg_black_pct, reg_black_lo, reg_black_hi = get_prev('registered', 'Black')

att_gap = get_gap('support')     # attitudinal gap  (Black − White)
beh_gap = get_gap('registered')  # behavioral gap   (Black − White)

print("=== Data pulled from CSV ===")
print(f"Support   White : {sup_white_pct}")
print(f"Support   Black : {sup_black_pct}")
print(f"Registered White: {reg_white_pct}")
print(f"Registered Black: {reg_black_pct}")
print(f"Att. gap  : {att_gap}")
print(f"Beh. gap  : {beh_gap}")

# ── 2. Colour palette ─────────────────────────────────────────────────────────
C_WHITE = '#2C5F8A'   # deep navy
C_BLACK = '#B83232'   # deep crimson
C_ATT   = '#4472C4'   # medium blue  (attitudinal gap bars)
C_BEH   = '#ED7D31'   # orange       (behavioral gap bars)

# ── 3. Figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 5.0), dpi=180, facecolor='white')
gs  = gridspec.GridSpec(
    1, 3,
    figure=fig,
    left=0.07, right=0.96,
    top=0.83,  bottom=0.14,
    wspace=0.38,
)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])
ax_c = fig.add_subplot(gs[2])

# ── 4. Helper: line panel ─────────────────────────────────────────────────────
def draw_line_panel(ax, w_pct, w_lo, w_hi, b_pct, b_lo, b_hi,
                    title, ylabel, ylim):
    for pct, lo, hi, color, marker, ls, race in [
        (w_pct, w_lo, w_hi, C_WHITE, 'o', '-',  'White'),
        (b_pct, b_lo, b_hi, C_BLACK, 's', '--', 'Black'),
    ]:
        err_lo = [p - l for p, l in zip(pct, lo)]
        err_hi = [h - p for h, p in zip(hi, pct)]
        ax.errorbar(
            WAVES, pct,
            yerr=[err_lo, err_hi],
            fmt=f'{marker}{ls}',
            color=color,
            linewidth=2,
            markersize=7,
            capsize=3,
            capthick=1.3,
            elinewidth=1.2,
            ecolor=color,
            alpha=0.92,
            zorder=3,
        )
        # End-point label
        ax.annotate(
            f'{pct[-1]:.1f}%',
            xy=(2025, pct[-1]),
            xytext=(8, 0),
            textcoords='offset points',
            fontsize=9.5,
            color=color,
            fontweight='bold',
            va='center',
        )

    ax.set_xlim(2009, 2030)
    ax.set_ylim(*ylim)
    ax.set_xticks(WAVES)
    ax.set_xticklabels([str(w) for w in WAVES], fontsize=10)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'{int(v)}%'))
    ax.tick_params(labelsize=10)
    ax.set_xlabel('Survey Wave', fontsize=10.5, labelpad=6)
    ax.set_ylabel(ylabel,        fontsize=10.5, labelpad=6)
    ax.yaxis.grid(True, color='#E8E8E8', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11.5, fontweight='bold', pad=10, loc='left')

# ── 5. Helper: gap bar panel ──────────────────────────────────────────────────
def draw_gap_panel(ax, att, beh):
    x  = np.array([0, 1, 2])
    bw = 0.32

    ax.bar(x - bw/2, att, width=bw, color=C_ATT, alpha=0.85,
           zorder=3, label='Attitudinal gap')
    ax.bar(x + bw/2, beh, width=bw, color=C_BEH, alpha=0.85,
           zorder=3, label='Behavioral gap')

    # Value labels centered inside bars
    for vals, offset in [(att, -bw/2), (beh, bw/2)]:
        for i, v in enumerate(vals):
            ax.text(
                x[i] + offset,
                v / 2,
                f'{v:+.1f}',
                ha='center', va='center',
                fontsize=8.5, color='white', fontweight='bold',
            )

    ax.axhline(0, color='#444444', linewidth=0.9, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(w) for w in WAVES], fontsize=10)
    ax.set_ylim(-30, 5)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'{int(v)} pp'))
    ax.tick_params(labelsize=10)
    ax.set_xlabel('Survey Wave',              fontsize=10.5, labelpad=6)
    ax.set_ylabel('Gap (Black − White, pp)', fontsize=10.5, labelpad=6)
    ax.yaxis.grid(True, color='#E8E8E8', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(
        '(C) Attitudinal vs. Behavioral Gap\n(Black − White)',
        fontsize=11.5, fontweight='bold', pad=10, loc='left',
    )
    ax.legend(
        fontsize=9,
        frameon=True, framealpha=0.9, edgecolor='#CCCCCC',
        loc='lower left',
    )

# ── 6. Draw panels ────────────────────────────────────────────────────────────
draw_line_panel(
    ax_a,
    sup_white_pct, sup_white_lo, sup_white_hi,
    sup_black_pct, sup_black_lo, sup_black_hi,
    title='(A) Support for Organ Donation',
    ylabel='Endorsement Rate (%)',
    ylim=(78, 102),
)
draw_line_panel(
    ax_b,
    reg_white_pct, reg_white_lo, reg_white_hi,
    reg_black_pct, reg_black_lo, reg_black_hi,
    title='(B) Registered as Organ Donor',
    ylabel='Registration Rate (%)',
    ylim=(25, 80),
)
draw_gap_panel(ax_c, att_gap, beh_gap)

# ── 7. Legend placed inside panel A (lower left) ─────────────────────────────
legend_elements = [
    Line2D([0], [0], color=C_WHITE, marker='o', linestyle='-',
           linewidth=2, markersize=7, label='White'),
    Line2D([0], [0], color=C_BLACK, marker='s', linestyle='--',
           linewidth=2, markersize=7, label='Black'),
]
ax_a.legend(
    handles=legend_elements,
    fontsize=9.5,
    frameon=True, framealpha=0.92, edgecolor='#CCCCCC',
    title='Race/Ethnicity', title_fontsize=9,
    loc='lower left',
    handlelength=2,
)

# ── 8. Figure title & note ────────────────────────────────────────────────────
fig.text(
    0.5, 0.97,
    'Racial Disparities in Organ Donation Support and Registration, 2012–2025',
    ha='center', va='top',
    fontsize=13, fontweight='bold', color='#1C1C1C',
)
fig.text(
    0.5, 0.01,
    'Note: Error bars represent 95% confidence intervals. '
    'pp = percentage points. Gaps calculated as Black − White.',
    ha='center', va='bottom',
    fontsize=8.5, color='#666666', style='italic',
)

# ── 9. Save ───────────────────────────────────────────────────────────────────
OUT_PDF = 'figure_main_manuscript_final.pdf'
OUT_PNG = 'figure_main_manuscript_final.png'

plt.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
plt.savefig(OUT_PNG, bbox_inches='tight', dpi=300, facecolor='white')
print(f"\nSaved:\n  {OUT_PDF}\n  {OUT_PNG}")
