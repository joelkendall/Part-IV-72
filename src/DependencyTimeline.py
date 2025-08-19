# deps_timeline_pies_single_static.py
# Usage:
#   pip install pandas plotly openpyxl
#   python deps_timeline_pies_single_static.py path/to/data.xlsx
#
# Columns expected:
#   File, Total Dependencies, <category1>, <category2>, ...

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -----------------------------
# 1) Load data (CSV or Excel)
# -----------------------------
if len(sys.argv) < 2:
    print("Usage: python deps_timeline_pies_single_static.py path/to/file.(csv|xlsx)")
    sys.exit(1)

file_path = Path(sys.argv[1])
if file_path.suffix.lower() in [".xlsx", ".xls"]:
    # adjust sheet_name as needed; you used index 1 previously
    df = pd.read_excel(file_path, sheet_name=1)
else:
    df = pd.read_csv(file_path)

# Required columns
for col in ["File", "Total Dependencies"]:
    if col not in df.columns:
        raise ValueError(f"Missing required column: '{col}'")

# Categories = everything except File and Total Dependencies
categories = [c for c in df.columns if c not in ("File", "Total Dependencies")]
if not categories:
    raise ValueError("No category columns found. Add columns like extends, implements, etc.")

# Types & order (keep XLSX order; do NOT sort)
df["File"] = df["File"].astype(str)
df["Total Dependencies"] = pd.to_numeric(df["Total Dependencies"], errors="coerce").fillna(0).astype(int)
for c in categories:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

# -----------------------------
# 2) Helpers
# -----------------------------
def pie_domain_for_total(total):
    """
    Map 'total' to a horizontal domain width so larger totals look like bigger donuts.
    Uses sqrt scaling so visual area ~ total.
    """
    t = np.sqrt(max(total, 0))
    lo = np.sqrt(max(df["Total Dependencies"].min(), 1))
    hi = np.sqrt(max(df["Total Dependencies"].max(), 1))
    if hi == lo:
        width = 0.6
    else:
        width = 0.35 + 0.6 * (t - lo) / (hi - lo)  # 0.35..0.95 range
    c = 0.5
    x0 = c - width / 2
    x1 = c + width / 2
    return [x0 + 0.02, x1 - 0.02], [0.18, 0.95]

# -----------------------------
# 3) Build traces
# -----------------------------
traces = []

# (A) Donut per version (only one visible at a time)
for i in range(len(df)):
    row = df.iloc[i]
    vals = [int(row[c]) for c in categories]
    x_dom, y_dom = pie_domain_for_total(row["Total Dependencies"])
    traces.append(go.Pie(
        labels=categories,
        values=vals,
        name=row["File"],
        hole=0.45,
        textinfo="none",
        sort=False,
        direction="clockwise",
        showlegend=True,
        domain={"x": x_dom, "y": y_dom},
        hovertemplate=(
            f"<b>{row['File']}</b><br>"
            "Category: %{label}<br>"
            "Count: %{value}<br>"
            "Share: %{percent}<extra></extra>"
        ),
        visible=(i == 0),  # only first visible at load
    ))

pie_count = len(df)

# (B) Base timeline (all points small & same color)
x_idx = np.arange(len(df))
timeline_base = go.Scatter(
    x=x_idx, y=np.zeros(len(df)),
    mode="markers",
    marker=dict(size=10),
    text=df["File"],
    hovertemplate="<b>%{text}</b><extra></extra>",
    showlegend=False,
    xaxis="x2", yaxis="y2",
)
traces.append(timeline_base)

# (C) One highlight marker per version (visible only for current)
for i in range(len(df)):
    traces.append(go.Scatter(
        x=[i], y=[0],
        mode="markers",
        marker=dict(size=18),
        showlegend=False,
        xaxis="x2", yaxis="y2",
        hoverinfo="skip",
        visible=(i == 0),
    ))

highlight_count = len(df)

# -----------------------------
# 4) Figure + slider (no frames, no animation)
# -----------------------------
fig = go.Figure(data=traces)

# Build slider steps that toggle visibility arrays (static update)
steps = []
for i in range(len(df)):
    visible = [False] * (pie_count + 1 + highlight_count)

    # turn on the i-th pie
    visible[i] = True

    # base timeline always on
    visible[pie_count] = True

    # turn on the i-th highlight (offset after base timeline)
    visible[pie_count + 1 + i] = True

    steps.append({
        "method": "update",  # no animate
        "label": df.iloc[i]["File"],
        "args": [{"visible": visible},
                 {"title": {"text": f"Dependency Categories — {df.iloc[i]['File']}"}}],
    })

fig.update_layout(
    title={"text": f"Dependency Categories — {df.iloc[0]['File']}", "x": 0.5},

    # Bottom timeline axes (x2/y2)
    xaxis2=dict(domain=[0.05, 0.95], anchor="y2",
                tickmode="array",
                tickvals=x_idx,
                ticktext=[""] * len(df),  # hide long labels; hover shows names
                showgrid=False, zeroline=False, showline=True),
    yaxis2=dict(domain=[0.0, 0.12], anchor="x2",
                range=[-1, 1], visible=False),

    margin=dict(l=10, r=10, t=60, b=10),
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=0.03, xanchor="center", x=0.5),

    sliders=[{
        "active": 0,
        "pad": {"t": 25, "b": 0, "l": 40, "r": 40},
        "len": 0.9,
        "currentvalue": {"prefix": "Version: "},
        "steps": steps,
    }],
)

# -----------------------------
# 5) Export HTML
# -----------------------------
out = file_path.with_suffix(".single_pie_timeline_static.html")
fig.write_html(str(out), include_plotlyjs="cdn")
print(f"Saved: {out.resolve()}")
