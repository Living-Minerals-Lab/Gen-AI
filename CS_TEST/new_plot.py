from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

SCRIPT_DIR = Path(__file__).resolve().parent
EUC_CSV = SCRIPT_DIR / "summaries" / "make_unique_summary_euc.csv"
PETALITE_CSV = SCRIPT_DIR / "summaries" / "all_unique_summary_petalite.csv"
SPODUMENE_CSV = SCRIPT_DIR / "summaries" / "all_unique_summary_spodumene.csv"
OUTPUT_HTML = SCRIPT_DIR / "phase_diagram_euc.html"

# Composition axis: x = Si / (Al + Si), the atomic tetrahedral-site fraction
# that goes from pure Al (LiAlO2) to pure Si (SiO2). Eucryptite (LiAlSiO4) is
# LiAlO2 + SiO2 (1 Al + 1 Si) -> x=0.5. Spodumene (LiAlSi2O6) is LiAlO2 + 2
# SiO2 (1 Al + 2 Si) -> x=2/3. Petalite (LiAlSi4O10) is LiAlO2 + 4 SiO2
# (1 Al + 4 Si) -> x=4/(1+4)=0.8. Every generated candidate for a given phase
# shares that phase's composition, so they all sit at one x.
X_LIALO2 = 0.0
X_EUCRYPTITE = 0.5
X_SPODUMENE = 2 / 3
X_PETALITE = 0.8
X_SIO2 = 1.0

# Real formation energies (eV/atom, relative to the elements), from Materials
# Project. LiAlO2 and SiO2 are compounds, not pure elements, so unlike a Li-O
# diagram they do NOT sit at y=0 in this "relative to the elements" frame --
# they get re-referenced to y=0 below, relative to each other instead.
EF_LIALO2 = -3.091            # mp-3427, gamma-LiAlO2, tetragonal P4_1 2_1 2, Ehull=0.000 (ground state)
EF_SIO2 = -3.268              # mp-6930, alpha-quartz, trigonal P3_2 21 (standard SiO2 reference form)
EF_EUCRYPTITE_ALPHA = -3.219  # mp-18220, LiAlSiO4, trigonal R3 (#146), Ehull=0.000 (matches
                               # this project's established true ordered alpha-eucryptite structure)
EF_PETALITE = -3.254          # mp-6442, LiAlSi4O10, monoclinic P2/c, Ehull=0.000 (ground state)

# Spodumene (LiAlSi2O6) is different from the three references above: NO
# known Materials Project entry at this composition sits at Ehull=0.000 (the
# lowest is mp-558808, triclinic P1, Ef=-3.214, Ehull=0.024 -- even the real
# natural mineral polymorph mp-6340, C2/c, is metastable at Ehull=0.034). So
# there's no single compound whose own Ef IS the local hull value here. This
# repo's own out.txt files independently confirm mp-558808's Ehull (~0.023,
# consistent across multiple directories checked) against the same local
# Li-Al-Si-O phase diagram used to score every generated candidate -- so the
# true hull value at this composition is derived self-consistently from that,
# not assumed:
EF_HULL_AT_SPODUMENE = -3.214 - 0.023  # = -3.237 eV/atom (mp-558808 Ef minus its own local Ehull)

# Atoms per formula unit -- needed because LiAlO2 (4 atoms/f.u.) and SiO2 (3
# atoms/f.u.) differ, so the per-atom energy of a mechanical LiAlO2+SiO2
# mixture is NOT the naive linear interpolation (1-x)*EF_LIALO2 + x*EF_SIO2;
# it must be atom-weighted.
ATOMS_LIALO2 = 4
ATOMS_SIO2 = 3


def tie_line(x):
    """Per-atom energy of a mechanical LiAlO2+SiO2 mixture at composition x
    (Si/(Al+Si)), atom-weighted since the two end members have different
    atoms/formula-unit."""
    return (
        (1 - x) * ATOMS_LIALO2 * EF_LIALO2 + x * ATOMS_SIO2 * EF_SIO2
    ) / ((1 - x) * ATOMS_LIALO2 + x * ATOMS_SIO2)


TIE_AT_EUCRYPTITE = tie_line(X_EUCRYPTITE)  # -3.1669 eV/atom
TIE_AT_SPODUMENE = tie_line(X_SPODUMENE)    # -3.1972 eV/atom
TIE_AT_PETALITE = tie_line(X_PETALITE)      # -3.2238 eV/atom

# Re-referenced to the LiAlO2+SiO2 tie-line instead of the elements, so both
# endpoints sit at exactly y=0 and each phase's position directly shows how
# much more stable it is than phase-separating into LiAlO2 + SiO2.
EUCRYPTITE_RELATIVE = EF_EUCRYPTITE_ALPHA - TIE_AT_EUCRYPTITE      # -0.0521 eV/atom
SPODUMENE_RELATIVE = EF_HULL_AT_SPODUMENE - TIE_AT_SPODUMENE       # -0.0398 eV/atom
PETALITE_RELATIVE = EF_PETALITE - TIE_AT_PETALITE                  # -0.0303 eV/atom


def load_candidates(csv_path, ef_reference, tie_at_x):
    df = pd.read_csv(csv_path)
    e_hull = df["Energy Above Hull (eV/atom)"]
    # Every candidate's absolute formation energy, recovered from its Energy
    # Above Hull relative to that phase's own hull point above.
    df["formation_energy"] = ef_reference + e_hull
    df["relative_energy"] = df["formation_energy"] - tie_at_x
    return df


df_euc = load_candidates(EUC_CSV, EF_EUCRYPTITE_ALPHA, TIE_AT_EUCRYPTITE)
df_spod = load_candidates(SPODUMENE_CSV, EF_HULL_AT_SPODUMENE, TIE_AT_SPODUMENE)
df_pet = load_candidates(PETALITE_CSV, EF_PETALITE, TIE_AT_PETALITE)

fig = go.Figure()

# Hull line: bends through the eucryptite, spodumene, and petalite points,
# all of which sit below the straight LiAlO2-SiO2 tie-line -- i.e. all three
# are stable against decomposing into LiAlO2 + SiO2 (spodumene's vertex uses
# the derived near-hull value above, not a genuine Ehull=0.000 compound).
fig.add_trace(go.Scatter(
    x=[X_LIALO2, X_EUCRYPTITE, X_SPODUMENE, X_PETALITE, X_SIO2],
    y=[0, EUCRYPTITE_RELATIVE, SPODUMENE_RELATIVE, PETALITE_RELATIVE, 0],
    mode="lines",
    line=dict(color="#c3c2b7", width=2),
    name="Hull",
    hoverinfo="skip",
))


def add_candidate_trace(df, x_pos, name, color):
    fig.add_trace(go.Scatter(
        x=[x_pos] * len(df),
        y=df["relative_energy"],
        mode="markers",
        marker=dict(color=color, size=8, opacity=0.5, line=dict(width=0)),
        name=name,
        customdata=df[["Gen ID", "Mattergen ID", "Energy Above Hull (eV/atom)", "formation_energy"]],
        hovertemplate=(
            "Gen ID: %{customdata[0]}<br>"
            "Mattergen ID: %{customdata[1]}<br>"
            "Energy rel. to LiAlO2+SiO2: %{y:.3f} eV/atom<br>"
            "Formation energy: %{customdata[3]:.3f} eV/atom<br>"
            "E above hull: %{customdata[2]:.3f} eV/atom"
            "<extra></extra>"
        ),
    ))


# Generated candidates, plotted at each phase's exact composition -- no
# jitter, so structures that share a relative energy genuinely overlap/stack
# rather than being spread apart. Fixed categorical colors (blue, orange, aqua).
add_candidate_trace(df_euc, X_EUCRYPTITE, "Generated candidates (Eucryptite)", "#2a78d6")
add_candidate_trace(df_spod, X_SPODUMENE, "Generated candidates (Spodumene)", "#1baf7a")
add_candidate_trace(df_pet, X_PETALITE, "Generated candidates (Petalite)", "#eb6834")

# Reference anchors, styled distinctly from the candidate data so they read
# as known compounds rather than generated structures. Spodumene's label is
# explicit that it's a derived near-hull estimate, not a confirmed Ehull=0.000
# compound like the other three.
fig.add_trace(go.Scatter(
    x=[X_LIALO2, X_EUCRYPTITE, X_SPODUMENE, X_PETALITE, X_SIO2],
    y=[0, EUCRYPTITE_RELATIVE, SPODUMENE_RELATIVE, PETALITE_RELATIVE, 0],
    mode="markers+text",
    marker=dict(color="#0b0b0b", size=14, symbol="diamond"),
    text=[
        "LiAlO2",
        "Eucryptite (α, R3, on hull)",
        "Spodumene (mp-558808 basis, Ehull≈0.02, not exactly on hull)",
        "Petalite (on hull)",
        "SiO2",
    ],
    textposition=["middle left", "top center", "bottom center", "top center", "middle right"],
    textfont=dict(color="#0b0b0b", size=13),
    name="Reference compounds",
    hovertemplate="%{text}<br>Energy rel. to LiAlO2+SiO2: %{y:.3f} eV/atom<extra></extra>",
))

fig.update_layout(
    title="LiAlO2–SiO2 Pseudo-Binary Phase Diagram (Eucryptite, Spodumene & Petalite)",
    xaxis=dict(
        title="Composition (Si / (Al + Si))",
        tickmode="array",
        tickvals=[X_LIALO2, X_EUCRYPTITE, X_SPODUMENE, X_PETALITE, X_SIO2],
        ticktext=["LiAlO2", "LiAlSiO4 (Eucryptite)", "LiAlSi2O6 (Spodumene)", "LiAlSi4O10 (Petalite)", "SiO2"],
        range=[-0.1, 1.1],
    ),
    yaxis=dict(title="Energy Relative to LiAlO2 + SiO2 (eV/atom)"),
    plot_bgcolor="#fcfcfb",
    paper_bgcolor="#fcfcfb",
    font=dict(color="#0b0b0b"),
    legend=dict(bordercolor="#e1e0d9", borderwidth=1),
    hovermode="closest",
)

fig.write_html(OUTPUT_HTML)
print(
    f"Wrote {len(df_euc)} eucryptite + {len(df_spod)} spodumene + "
    f"{len(df_pet)} petalite candidates to {OUTPUT_HTML}"
)
