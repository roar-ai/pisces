# Rebuild the chart with requested tweaks:
# (1) Visual Quality header moved a bit further up
# (2) enforce a single font for all text
# (3) order groups: Visual Quality (top), Motion Quality (middle), Semantic Alignment (bottom)

import matplotlib as mpl
import matplotlib.pyplot as plt

# Single, consistent font across all text
mpl.rcParams.update({
    "font.family": "DejaVu Sans",  # widely available; change later if needed
    "font.size": 12
})

# Data in requested order
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

# Layout params
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

# Plot (single axes)
fig = plt.figure(figsize=(8, 4.6), dpi=160)
ax = plt.gca()

pisces_vals = [r[2] for r in rows]
other_vals = [r[3] for r in rows]

ax.barh(y_positions, pisces_vals, height=bar_height)
ax.barh(y_positions, other_vals, left=pisces_vals, height=bar_height)

# Percent labels
for y, (gname, comp, p, o) in zip(y_positions, rows):
    ax.text(p * 0.5, y, f"{p:.1f}%", va="center", ha="center")
    ax.text(p + o * 0.5, y, f"{o:.1f}%", va="center", ha="center")

# 50% line
ax.axvline(50, linestyle="--", color="black", linewidth=1)

# Group headers with offsets (Visual up, Motion centered, Semantic default)
group_midpoints = []
idx = 0
for gname, comps in groups.items():
    ys = y_positions[idx : idx + len(comps)]
    mid = sum(ys) / len(ys)
    group_midpoints.append((gname, mid))
    idx += len(comps)
    if idx < len(y_positions):
        idx += 1  # account for gap

offsets = {"Visual Quality": 0.0, "Motion Quality": -0.25, "Semantic Alignment": 0.95}
for gname, mid in group_midpoints:
    ax.text(50.5, mid + 0.9 + offsets.get(gname, 0.0), gname, ha="center", va="center", fontweight="bold")

# Formatting
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
# png_path = "/mnt/data/pisces_human_preference_v3.png"
pdf_path = "./pisces_human_preference.pdf"
# fig.savefig(png_path, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
# png_path, pdf_path