from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_CSV = SCRIPT_DIR / "summaries" / "make_unique_summary_euc.csv"
OUTPUT_HTML = SCRIPT_DIR / "phase_diagram_euc.html"

# Eucryptite (LiAlSiO4) is the balanced 1:1 combination of the two end-member
# oxides: LiAlO2 + SiO2 -> LiAlSiO4. Every generated candidate in the summary
# CSV shares that same stoichiometry, so they all sit at the same composition.
X_LIALO2 = 0.0
X_EUCRYPTITE = 0.5
X_SIO2 = 1.0

# Real formation energies (eV/atom, relative to the elements), from Materials
# Project. These are the actual endpoints/hull anchor of the LiAlO2-SiO2
# pseudo-binary join -- LiAlO2 and SiO2 are compounds, not pure elements, so
# unlike a Li-O diagram they do NOT sit at y=0.
EF_LIALO2 = -3.091            # mp-3427, gamma-LiAlO2, tetragonal P4_1 2_1 2, Ehull=0.000 (ground state)
EF_SIO2 = -3.268              # mp-6930, alpha-quartz, trigonal P3_2 21 (standard SiO2 reference form)
EF_EUCRYPTITE_ALPHA = -3.219  # mp-18220, LiAlSiO4, trigonal R3 (#146), Ehull=0.000 (matches
                               # this project's established true ordered alpha-eucryptite structure)

df = pd.read_csv(SUMMARY_CSV)
e_hull = df["Energy Above Hull (eV/atom)"]

# Every candidate's absolute formation energy, recovered from its Energy Above
# Hull relative to the alpha-eucryptite hull point above.
df["formation_energy"] = EF_EUCRYPTITE_ALPHA + e_hull

fig = go.Figure()

# Hull line: bends through the eucryptite point, since alpha-eucryptite (a
# genuine MP ground state, Ehull=0.000) sits below the straight LiAlO2-SiO2
# tie-line -- i.e. it's stable against decomposing into LiAlO2 + SiO2.
fig.add_trace(go.Scatter(
    x=[X_LIALO2, X_EUCRYPTITE, X_SIO2],
    y=[EF_LIALO2, EF_EUCRYPTITE_ALPHA, EF_SIO2],
    mode="lines",
    line=dict(color="#c3c2b7", width=2),
    name="Hull",
    hoverinfo="skip",
))

# Generated eucryptite candidates, plotted at the exact eucryptite
# composition -- no jitter, so structures that share a formation energy
# genuinely overlap/stack rather than being spread apart.
fig.add_trace(go.Scatter(
    x=[X_EUCRYPTITE] * len(df),
    y=df["formation_energy"],
    mode="markers",
    marker=dict(color="#2a78d6", size=8, opacity=0.5, line=dict(width=0)),
    name="Generated candidates",
    customdata=df[["Gen ID", "Mattergen ID", "Energy Above Hull (eV/atom)"]],
    hovertemplate=(
        "Gen ID: %{customdata[0]}<br>"
        "Mattergen ID: %{customdata[1]}<br>"
        "Formation energy: %{y:.3f} eV/atom<br>"
        "E above hull: %{customdata[2]:.3f} eV/atom"
        "<extra></extra>"
    ),
))

# Reference anchors, styled distinctly from the candidate data so they read
# as known compounds rather than generated structures.
fig.add_trace(go.Scatter(
    x=[X_LIALO2, X_EUCRYPTITE, X_SIO2],
    y=[EF_LIALO2, EF_EUCRYPTITE_ALPHA, EF_SIO2],
    mode="markers+text",
    marker=dict(color="#0b0b0b", size=14, symbol="diamond"),
    text=["LiAlO2", "Eucryptite (α, R3, on hull)", "SiO2"],
    textposition=["middle left", "top center", "middle right"],
    textfont=dict(color="#0b0b0b", size=13),
    name="Reference compounds (stable)",
    hovertemplate="%{text}<br>Formation energy: %{y:.3f} eV/atom<extra></extra>",
))

fig.update_layout(
    title="Eucryptite Pseudo-Binary Phase Diagram (LiAlO2 ↔ SiO2)",
    xaxis=dict(
        title="Composition (Si / (Al + Si))",
        tickmode="array",
        tickvals=[X_LIALO2, X_EUCRYPTITE, X_SIO2],
        ticktext=["LiAlO2", "LiAlSiO4 (Eucryptite)", "SiO2"],
        range=[-0.1, 1.1],
    ),
    yaxis=dict(title="Formation Energy (eV/atom)"),
    plot_bgcolor="#fcfcfb",
    paper_bgcolor="#fcfcfb",
    font=dict(color="#0b0b0b"),
    legend=dict(bordercolor="#e1e0d9", borderwidth=1),
    hovermode="closest",
)

fig.write_html(OUTPUT_HTML)
print(f"Wrote {len(df)} candidates to {OUTPUT_HTML}")
