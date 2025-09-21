# Add Wilson 95% CIs and N=400 per row to the user's current chart

import math
import matplotlib as mpl
import matplotlib.pyplot as plt

# -------------------- Styling: single font --------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12
})

# -------------------- Data (kept exactly as user provided order/values) --------------------
groups = {
    "Semantic Alignment": [
        ("VideoReward-DPO", 67.5, 32.5),
        ("T2V-Turbo-v2", 68.7, 31.3),
        ("HunyuanVideo", 62.6, 37.4),
    ],
    "Motion Quality": [
        ("VideoReward-DPO", 58.2, 41.8),
        ("T2V-Turbo-v2", 59.8, 40.2),
        ("HunyuanVideo", 57.4, 42.6),
    ],
    "Visual Quality": [
        ("VideoReward-DPO", 65.7, 34.3),
        ("T2V-Turbo-v2", 63.4, 36.6),
        ("HunyuanVideo", 68.2, 31.8),
    ],
}

# Wilson CI for binomial proportion
def wilson_ci(pct, N, z=1.96):
    p = pct / 100.0
    denom = 1 + (z**2)/N
    center = p + (z**2)/(2*N)
    margin = z * math.sqrt(p*(1-p)/N + (z**2)/(4*N**2))
    lo = max(0.0, (center - margin)/denom) * 100.0
    hi = min(1.0, (center + margin)/denom) * 100.0
    return lo, hi

# -------------------- Layout bookkeeping --------------------
bar_height = 0.8
gap = 0.6

rows = []
y_positions = []
right_labels = []
current_y = 0

for gi, (gname, comps) in enumerate(groups.items()):
    for comp, pisces, other in comps:
        rows.append((gname, comp, pisces, other))
        y_positions.append(current_y)
        right_labels.append(comp)
        current_y += 1
    if gi < len(groups) - 1:
        current_y += gap

# -------------------- Plot --------------------
fig = plt.figure(figsize=(8, 4.6), dpi=160)
ax = plt.gca()

pisces_vals = [r[2] for r in rows]
other_vals  = [r[3] for r in rows]

ax.barh(y_positions, pisces_vals, height=bar_height)
ax.barh(y_positions, other_vals, left=pisces_vals, height=bar_height)

# Percent annotations
for y, (_, _, p, o) in zip(y_positions, rows):
    ax.text(p * 0.5, y, f"{p:.1f}%", va="center", ha="center")
    ax.text(p + o * 0.5, y, f"{o:.1f}%", va="center", ha="center")

# 50% reference line
ax.axvline(50, linestyle="--", color="black", linewidth=1)

# Group headers (same offsets as user's last version)
group_midpoints = []
idx = 0
for gname, comps in groups.items():
    ys = y_positions[idx: idx + len(comps)]
    mid = sum(ys) / len(ys)
    group_midpoints.append((gname, mid))
    idx += len(comps)
    if idx < len(y_positions):
        idx += 1  # account for gap

offsets = {"Visual Quality": 0.0, "Motion Quality": -0.25, "Semantic Alignment": 0.95}
for gname, mid in group_midpoints:
    ax.text(50.5, mid + 0.9 + offsets.get(gname, 0.0),
            gname, ha="center", va="center", fontweight="bold")

# -------------------- CIs & N --------------------
N_PER_ROW = 400  # user said 400 prompts
for y, (gname, comp, p, o) in zip(y_positions, rows):
    lo, hi = wilson_ci(p, N_PER_ROW)
    # Error bar for PISCES proportion
    ax.errorbar(
        x=p, y=y,
        xerr=[[max(0, p - lo)], [max(0, hi - p)]],
        fmt="none", capsize=3, lw=1.2, color="black"
    )
    # # Show N just under the comparator label at right
    # ax.text(100, y - 0.28, f"N={N_PER_ROW}",
    #         va="center", ha="left", clip_on=False, fontsize=10)

# -------------------- Axes formatting --------------------
ax.set_xlim(0, 100)
ax.set_xlabel("Preference (%)")
ax.set_yticks(y_positions)
ax.set_yticklabels(["PISCES"] * len(y_positions))
ax.set_xticks([0, 50, 100])

# Comparator labels on the right
for y, comp in zip(y_positions, right_labels):
    ax.text(100, y, comp, va="center", ha="left", clip_on=False)

# Clean up spines
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()

# Save
out_pdf = "./pisces_human_preference_with_ci.pdf"
fig.savefig(out_pdf, bbox_inches="tight")