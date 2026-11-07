import base64
import math
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# Electrical Grid Stability | Streamlit application
# Keep app.py, svc_model.sav and svc_scaler.sav in one folder.
# Optional: put config.toml inside a ".streamlit" folder next to app.py
# for the light theme and teal slider colour.
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "svc_model.sav"
SCALER_PATH = BASE_DIR / "svc_scaler.sav"

FEATURES = [
    "tau1", "tau2", "tau3", "tau4",
    "p1", "p2", "p3", "p4",
    "g1", "g2", "g3", "g4",
]

NAV = ["Overview", "Stability Predictor", "Model Performance", "About Project"]

# Colour tokens (also used inside the SVG illustrations)
PAPER = "#EDF1EF"
PANEL = "#F8FAF9"
INK = "#10283A"
SLATE = "#586C79"
WIRE = "#C3CFCB"
AMBER = "#E39A1B"
TEAL = "#17806D"
RED = "#CF4A2F"

# Starting values for the sliders: close to the middle of each training range.
DEFAULTS = {
    "tau1": 5.25, "tau2": 5.25, "tau3": 5.25, "tau4": 5.25,
    "p1": 3.75, "p2": -1.25, "p3": -1.25, "p4": -1.25,
    "g1": 0.52, "g2": 0.52, "g3": 0.52, "g4": 0.52,
}

# Allowed range of every input (taken from the training data).
RANGES = {
    **{f"tau{i}": (0.5, 10.0) for i in range(1, 5)},
    "p1": (1.5, 6.0),
    **{f"p{i}": (-2.0, -0.5) for i in range(2, 5)},
    **{f"g{i}": (0.05, 1.0) for i in range(1, 5)},
}

# Two real rows from Electric_Grid_Stability_.csv (rounded to 2 decimals).
STABLE_EXAMPLE = {
    "tau1": 5.96, "tau2": 3.17, "tau3": 1.50, "tau4": 8.56,
    "p1": 3.44, "p2": -0.95, "p3": -1.45, "p4": -1.04,
    "g1": 0.08, "g2": 0.20, "g3": 0.40, "g4": 0.06,
}
UNSTABLE_EXAMPLE = {
    "tau1": 8.69, "tau2": 6.94, "tau3": 6.98, "tau4": 6.01,
    "p1": 4.18, "p2": -1.31, "p3": -1.60, "p4": -1.27,
    "g1": 0.33, "g2": 0.84, "g3": 0.95, "g4": 0.83,
}

# Test accuracies from the notebook's results_df (before tuning), plus the tuned SVC.
MODEL_RESULTS = [
    ("SVC (tuned)", 0.9720, "tuned"),
    ("SVC (baseline)", 0.9635, "base-best"),
    ("XGBoost", 0.9485, "base"),
    ("Gradient Boosting", 0.9335, "base"),
    ("Random Forest", 0.9240, "base"),
    ("K-Nearest Neighbors", 0.8635, "base"),
    ("AdaBoost", 0.8610, "base"),
    ("Decision Tree", 0.8470, "base"),
    ("Gaussian Naive Bayes", 0.8430, "base"),
    ("Logistic Regression", 0.8205, "base"),
]

st.set_page_config(
    page_title="Electrical Grid Stability Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def html(markup):
    """Strip indentation and blank lines so Markdown never breaks an HTML block."""
    return "\n".join(line.strip() for line in markup.splitlines() if line.strip())


def show(markup):
    st.markdown(html(markup), unsafe_allow_html=True)


def svg_img(svg, alt, css_class=""):
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return (
        f'<img class="{css_class}" alt="{alt}" '
        f'src="data:image/svg+xml;base64,{encoded}">'
    )


# ------------------------------------------------------------
# Illustrations (generated in code, so no image files are needed)
# ------------------------------------------------------------
def grid_svg(values=None, state=None):
    """Four-node star grid: one producer in the centre, three consumers around it.

    values : the 12 inputs in FEATURES order (optional, adds labels and sizes)
    state  : None, "stable" or "unstable" (changes the colour of the current flow)
    """
    flow = {"stable": TEAL, "unstable": RED}.get(state, AMBER)
    period = "0.8s" if state == "unstable" else "1.6s"
    cx, cy, dist = 240, 215, 140
    nodes = [
        (cx + dist * math.cos(math.radians(a)), cy + dist * math.sin(math.radians(a)))
        for a in (-150, -30, 90)
    ]

    def fmt(v):
        return f"{v:+.2f}".replace("-", "\u2212")

    def text(x, y, content, size, weight, fill):
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}">{content}</text>'
        )

    p = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 450" '
        'font-family="Helvetica, Arial, sans-serif">',
        "<style>.flow{stroke-dasharray:7 11;animation:dash " + period + " linear infinite}"
        "@keyframes dash{to{stroke-dashoffset:-36}}"
        "@media (prefers-reduced-motion:reduce){.flow{animation:none}}</style>",
        '<defs><pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse">'
        '<path d="M24 0H0V24" fill="none" stroke="#DCE5E1" stroke-width="1"/></pattern></defs>',
        f'<rect width="480" height="450" fill="{PANEL}"/>',
        '<rect width="480" height="450" fill="url(#grid)"/>',
    ]

    # wires, with the current flowing from the producer to each consumer
    for x, y in nodes:
        p.append(
            f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{WIRE}" '
            'stroke-width="7" stroke-linecap="round"/>'
        )
        p.append(
            f'<line class="flow" x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" '
            f'stroke="{flow}" stroke-width="3" stroke-linecap="round"/>'
        )

    # consumers (nodes 2, 3, 4): circle size follows |p|
    for idx, (x, y) in enumerate(nodes):
        k = idx + 1
        r = 24.0
        if values:
            r = 18 + 14 * min(max((abs(values[4 + k]) - 0.5) / 1.5, 0), 1)
        if state:
            dash = ' stroke-dasharray="3 5"' if state == "unstable" else ""
            p.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r + 8:.1f}" fill="none" '
                f'stroke="{flow}" stroke-width="2"{dash}/>'
            )
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{INK}"/>')
        p.append(
            f'<path transform="translate({x:.1f} {y:.1f})" d="M-9 2L0 -8L9 2M-6 0V9H6V0" '
            'fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" '
            'stroke-linejoin="round"/>'
        )
        if idx < 2:  # labels above the two upper nodes
            values_y = y - (r + 10) - 6
            name_y = values_y - 16 if values else y - r - 18
        else:  # label below the lower node
            name_y = y + r + 26
            values_y = name_y + 16
        p.append(text(x, name_y, f"Consumer {k + 1}", 14, 700, INK))
        if values:
            p.append(
                text(
                    x, values_y,
                    f"\u03c4 {values[k]:.2f} \u00b7 p {fmt(values[4 + k])} \u00b7 \u03b3 {values[8 + k]:.2f}",
                    11.5, 500, SLATE,
                )
            )

    # producer (node 1) in the centre, labelled like a tag on the wire
    if state:
        dash = ' stroke-dasharray="3 5"' if state == "unstable" else ""
        p.append(
            f'<circle cx="{cx}" cy="{cy}" r="46" fill="none" stroke="{flow}" '
            f'stroke-width="2"{dash}/>'
        )
    p.append(f'<circle cx="{cx}" cy="{cy}" r="36" fill="{AMBER}"/>')
    p.append(
        f'<path transform="translate({cx} {cy})" d="M-17 0C-12 -14 -6 -14 0 0S12 14 17 0" '
        'fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round"/>'
    )
    pill_h = 44 if values else 26
    p.append(
        f'<rect x="{cx - 86}" y="{cy + 50}" width="172" height="{pill_h}" rx="6" '
        f'fill="{PANEL}" stroke="{WIRE}"/>'
    )
    p.append(text(cx, cy + 50 + 17, "Producer", 14, 700, INK))
    if values:
        p.append(
            text(
                cx, cy + 50 + 34,
                f"\u03c4 {values[0]:.2f} \u00b7 p {fmt(values[4])} \u00b7 \u03b3 {values[8]:.2f}",
                11.5, 500, SLATE,
            )
        )
    p.append("</svg>")
    return "".join(p)


def wave_svg(stable):
    """A disturbance that dies out (stable) or grows (unstable)."""
    w, h, mid, n = 360, 140, 70, 180
    color = TEAL if stable else RED
    points = []
    for i in range(n + 1):
        t = i / n
        if stable:
            amp = 54 * math.exp(-3.2 * t)
        else:
            amp = 4 + 50 * (math.exp(2.4 * t) - 1) / (math.exp(2.4) - 1)
        y = mid - amp * math.sin(2 * math.pi * 5 * t)
        points.append(f"{'M' if i == 0 else 'L'}{t * (w - 20) + 10:.1f} {y:.1f}")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">'
        f'<rect width="{w}" height="{h}" fill="{PANEL}"/>'
        f'<line x1="10" y1="{mid}" x2="{w - 10}" y2="{mid}" stroke="{WIRE}" '
        'stroke-width="1.5" stroke-dasharray="4 5"/>'
        f'<path d="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.6" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )


def logo_svg():
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">'
        f'<circle cx="20" cy="20" r="19" fill="{AMBER}"/>'
        '<path d="M22.5 7L11 23h8l-2 10 12-17h-8z" fill="#10283A"/></svg>'
    )


# ------------------------------------------------------------
# Styles
# ------------------------------------------------------------
def inject_styles():
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Instrument+Sans:wght@400;500;600;700&display=swap');

    :root {
        --paper: #EDF1EF; --panel: #F8FAF9; --ink: #10283A; --slate: #586C79;
        --rule: #C9D5D1; --amber: #E39A1B; --teal: #17806D; --red: #CF4A2F;
        --display: 'Bricolage Grotesque', 'Segoe UI', Helvetica, Arial, sans-serif;
        --body: 'Instrument Sans', 'Segoe UI', Helvetica, Arial, sans-serif;
    }
    html, body, [class*="css"], .stApp {
        font-family: var(--body);
    }
    .stApp { background: var(--paper); color: var(--ink); }
    .stApp [data-testid="stHeader"] { background: transparent; }
    footer, #MainMenu { visibility: hidden; }
    .block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1180px; }
    .stApp p, .stApp li, .stApp label, .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stMarkdownContainer"] { color: var(--ink); }
    .stApp [data-testid="stWidgetLabel"] p { font-weight: 600; font-size: .9rem; }

    /* sidebar */
    [data-testid="stSidebar"] { background: #0E2233; }
    .stApp [data-testid="stSidebar"] :is(p, span, label, li) { color: #E6EFEC !important; }
    .stApp [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.18); }
    .stApp [data-testid="stSidebar"] .brand-name { color: #FFFFFF; }
    .stApp [data-testid="stSidebar"] .brand-sub { color: #9DB6C3; }
    .stApp [data-testid="stSidebar"] div.stButton > button {
        background: transparent; color: #E6EFEC; border: 1px solid transparent;
        justify-content: flex-start; text-align: left; padding: .6rem .9rem; font-size: 1rem;
    }
    .stApp [data-testid="stSidebar"] div.stButton > button:hover {
        background: rgba(255,255,255,.10); color: #FFFFFF; border-color: transparent;
    }
    .stApp [data-testid="stSidebar"] div.stButton > button[kind="primary"],
    .stApp [data-testid="stSidebar"] div.stButton > button[data-testid="stBaseButton-primary"] {
        background: #E39A1B; color: #10283A; border-color: #E39A1B; font-weight: 700;
    }
    .stApp [data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover,
    .stApp [data-testid="stSidebar"] div.stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: #F2AE3A; color: #10283A; border-color: #F2AE3A;
    }
    .brand { display: flex; align-items: center; gap: .7rem; margin: .2rem 0 .3rem 0; }
    .brand img { width: 38px; height: 38px; }
    .brand-name { font-family: var(--display); font-weight: 800; font-size: 1.2rem; line-height: 1.05; }
    .brand-sub { font-size: .8rem; margin-bottom: 1rem; }

    /* typography */
    .hero-title {
        font-family: var(--display); font-weight: 800; letter-spacing: -.025em;
        font-size: clamp(2.3rem, 4.6vw, 3.7rem); line-height: 1.02; margin: 0 0 1rem 0;
    }
    .page-title {
        font-family: var(--display); font-weight: 800; letter-spacing: -.02em;
        font-size: clamp(1.9rem, 3.2vw, 2.6rem); line-height: 1.08; margin: 0 0 .5rem 0;
    }
    .lede { color: var(--slate) !important; font-size: 1.05rem; line-height: 1.65; max-width: 56ch; margin: 0 0 1.2rem 0; }
    .stApp h2, .stApp h3 {
        font-family: var(--display); font-weight: 700; letter-spacing: -.01em; color: var(--ink);
    }
    .section-title {
        font-family: var(--display); font-weight: 700; font-size: 1.5rem;
        letter-spacing: -.01em; margin: 2.4rem 0 .35rem 0;
    }
    .section-note { color: var(--slate) !important; max-width: 62ch; margin: 0 0 1rem 0; line-height: 1.6; }
    .muted { color: var(--slate) !important; font-size: .86rem; line-height: 1.55; }

    /* hero */
    .hero-grid { display: flex; flex-wrap: wrap; gap: 2.2rem; align-items: center; margin-bottom: 1.2rem; }
    .hero-copy { flex: 1 1 360px; }
    .hero-art { flex: 1 1 380px; max-width: 520px; }
    .hero-art img { width: 100%; display: block; border: 1px solid var(--rule); border-radius: 4px; }

    /* stat strip */
    .stats { display: flex; flex-wrap: wrap; border-top: 2px solid var(--ink); margin-top: 1.6rem; }
    .stat { flex: 1 1 170px; padding: 1rem 1.2rem 1rem 1.2rem; border-left: 1px solid var(--rule); }
    .stat:first-child { border-left: 0; padding-left: 0; }
    .stat-num { font-family: var(--display); font-weight: 700; font-size: 2.3rem; line-height: 1.05; letter-spacing: -.02em; }
    .stat-label { color: var(--slate) !important; font-size: .86rem; margin-top: .2rem; }

    .hero-art-img { width: 100%; display: block; border: 1px solid var(--rule); border-radius: 4px; margin-bottom: .8rem; }

    /* stable / unstable illustrations */
    .pair { display: flex; flex-wrap: wrap; gap: 1.4rem; }
    .pair figure { flex: 1 1 280px; margin: 0; }
    .pair img { width: 100%; display: block; border: 1px solid var(--rule); border-radius: 4px; }
    .pair figcaption { color: var(--slate); font-size: .9rem; line-height: 1.5; margin-top: .5rem; }
    .pair figcaption b { color: var(--ink); }

    /* glossary */
    .gloss { display: flex; flex-wrap: wrap; gap: 1.6rem; }
    .gloss-item { flex: 1 1 240px; border-top: 2px solid var(--ink); padding-top: .7rem; }
    .sym { font-family: var(--display); font-weight: 800; font-size: 2.6rem; line-height: 1; }
    .gloss-item h4 { font-family: var(--display); margin: .35rem 0 .3rem 0; font-size: 1.05rem; }
    .gloss-item p { color: var(--slate) !important; margin: 0; font-size: .92rem; line-height: 1.55; }

    /* definition rows */
    .defs { border-top: 1px solid var(--rule); max-width: 460px; }
    .defs div { display: flex; justify-content: space-between; gap: 1rem; padding: .55rem 0; border-bottom: 1px solid var(--rule); font-size: .95rem; }
    .defs span:first-child { color: var(--slate); }
    .defs span:last-child { font-weight: 600; text-align: right; }

    /* predictor */
    div[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--rule); background: var(--panel); border-radius: 6px; }
    .group-title { font-family: var(--display); font-weight: 700; font-size: 1.1rem; margin-bottom: .1rem; }
    .group-note { color: var(--slate) !important; font-size: .88rem; line-height: 1.5; margin: 0 0 .6rem 0; }
    .verdict { border-left: 8px solid var(--rule); padding: .3rem 0 .3rem 1rem; margin: 1rem 0 .6rem 0; }
    .verdict.stable { border-color: var(--teal); }
    .verdict.unstable { border-color: var(--red); }
    .verdict .word { font-family: var(--display); font-weight: 800; font-size: 2.6rem; line-height: 1; letter-spacing: -.02em; }
    .verdict.stable .word { color: var(--teal); }
    .verdict.unstable .word { color: var(--red); }
    .verdict p { margin: .4rem 0 0 0; color: var(--slate) !important; line-height: 1.5; font-size: .95rem; }
    .stale { border: 1px dashed var(--rule); padding: .7rem .9rem; border-radius: 4px; color: var(--slate) !important; font-size: .92rem; margin-top: 1rem; }
    .meter { margin: .6rem 0 .2rem 0; }
    .meter-track { position: relative; height: 10px; border-radius: 5px; background: linear-gradient(90deg, var(--teal), #D9DED6 50%, var(--red)); }
    .meter-mark { position: absolute; top: -5px; width: 4px; height: 20px; background: var(--ink); border-radius: 2px; transform: translateX(-2px); }
    .meter-ends { display: flex; justify-content: space-between; font-size: .8rem; color: var(--slate); margin-top: .3rem; }

    /* confusion matrix */
    .cm { display: grid; grid-template-columns: 110px 1fr 1fr; gap: 4px; max-width: 520px; }
    .cm-h, .cm-v { color: var(--slate); font-size: .82rem; display: flex; align-items: center; justify-content: center; text-align: center; padding: .3rem; }
    .cm-v { justify-content: flex-end; text-align: right; }
    .cm-c { padding: 1.1rem .6rem; text-align: center; border-radius: 4px; font-family: var(--display); font-weight: 700; font-size: 1.9rem; line-height: 1.1; }
    .cm-c small { display: block; font-family: var(--body); font-weight: 500; font-size: .8rem; margin-top: .25rem; }
    .cm-c.ok { background: var(--teal); color: #fff; }
    .cm-c.bad { background: #F3D9D2; color: #7A2A19; }

    /* tables */
    .tbl { width: 100%; border-collapse: collapse; font-size: .95rem; }
    .tbl th { text-align: left; font-weight: 600; color: var(--slate); border-bottom: 2px solid var(--ink); padding: .5rem .6rem .5rem 0; }
    .tbl td { border-bottom: 1px solid var(--rule); padding: .55rem .6rem .55rem 0; vertical-align: top; }
    .tbl td.num, .tbl th.num { text-align: right; font-variant-numeric: tabular-nums; }

    /* bars */
    .bars { max-width: 760px; }
    .bar-row { display: grid; grid-template-columns: 170px 1fr 64px; gap: .8rem; align-items: center; padding: .28rem 0; }
    .bar-name { font-size: .92rem; }
    .bar-track { height: 14px; background: #DDE5E1; border-radius: 3px; overflow: hidden; }
    .bar-fill { height: 100%; background: #8AA2AF; }
    .bar-row.tuned .bar-fill { background: var(--teal); }
    .bar-row.tuned .bar-name, .bar-row.tuned .bar-val { font-weight: 700; }
    .bar-row.base-best .bar-fill { background: #6FB3A5; }
    .bar-val { text-align: right; font-variant-numeric: tabular-nums; font-size: .92rem; }

    /* workflow */
    .tl { list-style: none; margin: .5rem 0 0 14px; padding: 0; border-left: 2px solid var(--rule); }
    .tl li { position: relative; padding: 0 0 1.3rem 2rem; }
    .tl-n { position: absolute; left: -15px; top: -2px; width: 28px; height: 28px; border-radius: 50%; background: var(--ink); color: #fff !important; font-weight: 700; font-size: .85rem; display: flex; align-items: center; justify-content: center; }
    .tl b { font-family: var(--display); font-size: 1.05rem; }
    .tl p { margin: .15rem 0 0 0; color: var(--slate) !important; line-height: 1.55; max-width: 62ch; }

    .note { border-left: 4px solid var(--amber); padding: .3rem 0 .3rem .9rem; margin-top: 1.6rem; color: var(--slate) !important; line-height: 1.55; max-width: 70ch; }

   /* buttons */
div.stButton > button,
div[data-testid="stDownloadButton"] > button {
    width: 100%;
    border-radius: 6px;
    font-weight: 600;
    padding: .6rem 1rem;
    background: #102F43 !important;
    color: #FFFFFF !important;
    border: 1px solid #102F43 !important;
}

/* Primary button */
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="stBaseButton-primary"] {
    background: #102F43 !important;
    color: #FFFFFF !important;
    border-color: #102F43 !important;
}

/* Hover effect */
div.stButton > button:hover,
div[data-testid="stDownloadButton"] > button:hover {
    background: #1D70B8 !important;
    color: #FFFFFF !important;
    border-color: #1D70B8 !important;
}

/* Button text */
div.stButton > button p,
div[data-testid="stDownloadButton"] > button p {
    color: #FFFFFF !important;
}

/* Keyboard focus */
button:focus-visible {
    outline: 2px solid #1D70B8 !important;
    outline-offset: 2px;
}
    """
    st.markdown(html(css), unsafe_allow_html=True)


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load the tuned SVC and the scaler fitted during notebook training."""
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        missing = []
        if not MODEL_PATH.exists():
            missing.append("svc_model.sav")
        if not SCALER_PATH.exists():
            missing.append("svc_scaler.sav")
        raise FileNotFoundError(
            "Missing model file(s): " + ", ".join(missing)
            + ". Place both saved files in the same folder as app1.py."
        )

    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)
    with SCALER_PATH.open("rb") as file:
        scaler = pickle.load(file)

    if hasattr(model, "n_features_in_") and model.n_features_in_ != len(FEATURES):
        raise ValueError(
            f"The saved model expects {model.n_features_in_} features, "
            f"but this app supplies {len(FEATURES)}."
        )
    return model, scaler


# ------------------------------------------------------------
# Reusable page parts
# ------------------------------------------------------------
def page_title(title, lede):
    show(
        f"""
        <div class="page-title">{title}</div>
        <p class="lede">{lede}</p>
        """
    )


def stat_strip(items):
    cells = "".join(
        f'<div class="stat"><div class="stat-num">{num}</div>'
        f'<div class="stat-label">{label}</div></div>'
        for num, label in items
    )
    show(f'<div class="stats">{cells}</div>')


def go_to(page):
    st.session_state["page"] = page


# ------------------------------------------------------------
# Overview
# ------------------------------------------------------------
def render_overview():
    show(
        f"""
        <div class="hero-grid">
            <div class="hero-copy">
                <div class="hero-title">Will this grid hold together?</div>
                <p class="lede">One power producer, three consumers, and a price signal that
                every participant answers with a delay. Set the delays, the power and the
                price response, and a tuned support-vector classifier calls the grid
                stable or unstable.</p>
            </div>
            <div class="hero-art">{svg_img(grid_svg(), "Star-shaped grid with one producer and three consumers")}</div>
        </div>
        """
    )

    b1, b2, _ = st.columns([1, 1, 1.4])
    b1.button("Open the predictor", type="primary", on_click=go_to, args=("Stability Predictor",))
    b2.button("See model results", on_click=go_to, args=("Model Performance",))

    stat_strip(
        [
            ("97.2%", "test accuracy, tuned SVC"),
            ("95.6%", "cross-validated macro F1"),
            ("10,000", "simulated observations"),
            ("9", "algorithms compared"),
        ]
    )

    show('<div class="section-title">What stable and unstable mean</div>')
    show(
        '<p class="section-note">Each label describes what happens after a small disturbance, '
        "such as a sudden change in demand. In the simulated grid the label comes from the "
        "stability of the system's equations.</p>"
    )
    show(
        f"""
        <div class="pair">
            <figure>{svg_img(wave_svg(True), "Oscillation that fades away")}
                <figcaption><b>Stable.</b> The disturbance dies out and the grid settles back.</figcaption>
            </figure>
            <figure>{svg_img(wave_svg(False), "Oscillation that keeps growing")}
                <figcaption><b>Unstable.</b> The disturbance grows with every swing, so the grid does not recover.</figcaption>
            </figure>
        </div>
        """
    )

    show('<div class="section-title">Reading the inputs</div>')
    show(
        '<p class="section-note">Every node in the grid has three settings, giving 12 inputs in total. '
        "Node 1 is the producer and nodes 2 to 4 are consumers.</p>"
    )
    show(
        f"""
        <div class="gloss">
            <div class="gloss-item">
                <div class="sym" style="color:{AMBER}">\u03c4</div>
                <h4>Reaction time</h4>
                <p class="muted">tau1, tau2, tau3, tau4</p>
                <p>How long a participant takes to respond to a price change. Range 0.5 to 10 seconds.</p>
            </div>
            <div class="gloss-item">
                <div class="sym" style="color:{INK}">p</div>
                <h4>Power</h4>
                <p class="muted">p1, p2, p3, p4</p>
                <p>Positive for the producer, negative for consumers (about \u22120.5 to \u22122).
                The producer's power always balances the three consumers.</p>
            </div>
            <div class="gloss-item">
                <div class="sym" style="color:{TEAL}">\u03b3</div>
                <h4>Price elasticity</h4>
                <p class="muted">g1, g2, g3, g4</p>
                <p>How strongly a participant adjusts its power when the price moves.
                Range 0.05 to 1.</p>
            </div>
        </div>
        """
    )

    show('<div class="section-title">Final model</div>')
    show(
        """
        <div class="defs">
            <div><span>Algorithm</span><span>Support vector classifier</span></div>
            <div><span>Kernel</span><span>RBF</span></div>
            <div><span>C</span><span>10</span></div>
            <div><span>SVC gamma</span><span>auto</span></div>
            <div><span>Tuning</span><span>5-fold CV, macro F1</span></div>
        </div>
        """
    )
    show(
        '<div class="note">This application demonstrates a model trained on simulated data. '
        "It is not designed for real-time grid monitoring, protection or operational decisions.</div>"
    )


# ------------------------------------------------------------
# Predictor
# ------------------------------------------------------------
def collect_values():
    """The 12 inputs in FEATURES order (the widget keys are the feature names)."""
    return [float(st.session_state[name]) for name in FEATURES]


def load_preset(preset):
    for key, value in preset.items():
        st.session_state[key] = value


def run_prediction(model, scaler):
    values = collect_values()
    key = tuple(round(v, 6) for v in values)
    try:
        scaled = scaler.transform(pd.DataFrame([values], columns=FEATURES))
        label = str(model.predict(scaled)[0]).strip().lower()
        margin = None
        if hasattr(model, "decision_function"):
            margin = float(model.decision_function(scaled)[0])
            # decision_function is positive for classes_[1]
            if str(model.classes_[1]).strip().lower() != "unstable":
                margin = -margin
        st.session_state["last_result"] = {
            "key": key, "values": values, "label": label, "margin": margin,
        }
    except Exception as exc:
        st.session_state["last_result"] = {"key": key, "error": str(exc)}


def balance_p1():
    st.session_state["p1"] = round(
        -sum(float(st.session_state[f"p{i}"]) for i in (2, 3, 4)), 2
    )


def feature_group(title, note, names, footer=None):
    with st.container(border=True):
        show(f'<div class="group-title">{title}</div><p class="group-note">{note}</p>')
        for col, name in zip(st.columns(4), names):
            lo, hi = RANGES[name]
            col.number_input(
                name, min_value=lo, max_value=hi, step=0.01, format="%.6f", key=name,
                help=f"Training range: {lo:g} to {hi:g}.",
            )
        if footer:
            footer()


def power_footer():
    gap = st.session_state["p1"] + sum(float(st.session_state[f"p{i}"]) for i in (2, 3, 4))
    if abs(gap) > 0.05:
        st.markdown(
            f'<p class="group-note">In the training data p1 = \u2212(p2 + p3 + p4). '
            f"Your p1 is {gap:+.2f} away from that.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="group-note">p1 matches \u2212(p2 + p3 + p4), as in the training data.</p>',
            unsafe_allow_html=True,
        )
    st.button("Balance p1 from p2, p3, p4", on_click=balance_p1, key="balance_p1_button")


def render_predict(model, scaler):
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)

    page_title(
        "Predict Electrical grid stability",
        "Enter the 12 input features. The diagram follows your values, and the classifier "
        "gives its verdict when you press Predict.",
    )

    values = collect_values()
    key = tuple(round(v, 6) for v in values)
    result = st.session_state.get("last_result")
    fresh = bool(result) and result.get("key") == key
    state = result.get("label") if fresh and "label" in result else None

    left, right = st.columns([1.55, 1], gap="large")

    with left:
        a, b, c = st.columns(3)
        a.button("Stable example", on_click=load_preset, args=(STABLE_EXAMPLE,),
                 help="A real row from the dataset labelled stable.")
        b.button("Unstable example", on_click=load_preset, args=(UNSTABLE_EXAMPLE,),
                 help="A real row from the dataset labelled unstable.")
        c.button("Reset", on_click=load_preset, args=(DEFAULTS,))

        feature_group(
            "tau1 to tau4: reaction time (seconds)",
            "tau1 belongs to the producer, tau2 to tau4 to the consumers.",
            ["tau1", "tau2", "tau3", "tau4"],
        )
        feature_group(
            "p1 to p4: power",
            "p1 is the producer's output (positive). p2 to p4 are the consumers' demand (negative).",
            ["p1", "p2", "p3", "p4"],
            footer=power_footer,
        )
        feature_group(
            "g1 to g4: price elasticity",
            "How strongly each participant adjusts to the price. g1 belongs to the producer.",
            ["g1", "g2", "g3", "g4"],
        )

    with right:
        show(svg_img(grid_svg(values, state), "Grid diagram that follows the input values", "hero-art-img"))
        st.button("Predict stability", type="primary", on_click=run_prediction, args=(model, scaler))

        if result and "error" in result and fresh:
            st.error(f"Prediction could not be completed: {result['error']}")
        elif fresh:
            label = result["label"]
            is_stable = label == "stable"
            word = "Stable" if is_stable else "Unstable"
            sentence = (
                "For these settings the model expects disturbances to die out."
                if is_stable
                else "For these settings the model expects disturbances to grow."
            )
            show(
                f"""
                <div class="verdict {'stable' if is_stable else 'unstable'}">
                    <div class="word">{word}</div>
                    <p>{sentence}</p>
                </div>
                """
            )
            margin = result.get("margin")
            if margin is not None:
                position = 50 + 50 * math.tanh(margin / 5)
                show(
                    f"""
                    <div class="meter">
                        <div class="meter-track"><div class="meter-mark" style="left:{position:.1f}%"></div></div>
                        <div class="meter-ends"><span>stable side</span><span>unstable side</span></div>
                    </div>
                    <p class="muted">Position relative to the SVC decision boundary. It shows how
                    far the settings sit from the boundary, not a probability.</p>
                    """
                )
            table = pd.DataFrame([result["values"]], columns=FEATURES)
            table.insert(0, "Predicted class", word)
            st.download_button(
                "Download this assessment (CSV)",
                data=table.to_csv(index=False).encode("utf-8"),
                file_name="grid_stability_assessment.csv",
                mime="text/csv",
            )
        elif result:
            show(
                '<div class="stale">The settings changed since the last prediction. '
                "Press Predict to refresh the result.</div>"
            )
        else:
            show(
                '<p class="muted">Keep every value inside the range shown for its field, which matches the '
                "training data. Predictions outside that range would not be reliable.</p>"
            )


# ------------------------------------------------------------
# Model performance
# ------------------------------------------------------------
def render_performance():
    page_title(
        "Model performance",
        "Results reported by the notebook on the held-out test set of 2,000 observations.",
    )

    stat_strip(
        [
            ("97.20%", "test accuracy"),
            ("0.97", "macro precision"),
            ("0.97", "macro recall"),
            ("0.97", "macro F1-score"),
        ]
    )

    show('<div class="section-title">Confusion matrix</div>')
    show(
        '<p class="section-note">Rows are the actual classes and columns are the predictions. '
        "Percentages are shares of each actual class.</p>"
    )
    stable_row = (695, 29)
    unstable_row = (27, 1249)

    def share(n, row):
        return f"{n / sum(row):.1%}"

    show(
        f"""
        <div class="cm">
            <div></div><div class="cm-h">Predicted stable</div><div class="cm-h">Predicted unstable</div>
            <div class="cm-v">Actual stable</div>
            <div class="cm-c ok">{stable_row[0]}<small>{share(stable_row[0], stable_row)}</small></div>
            <div class="cm-c bad">{stable_row[1]}<small>{share(stable_row[1], stable_row)}</small></div>
            <div class="cm-v">Actual unstable</div>
            <div class="cm-c bad">{unstable_row[0]}<small>{share(unstable_row[0], unstable_row)}</small></div>
            <div class="cm-c ok">{unstable_row[1]}<small>{share(unstable_row[1], unstable_row)}</small></div>
        </div>
        """
    )

    show('<div class="section-title">Class-wise evaluation</div>')
    show(
        """
        <table class="tbl">
            <tr><th>Class</th><th class="num">Precision</th><th class="num">Recall</th>
                <th class="num">F1-score</th><th class="num">Support</th></tr>
            <tr><td>Stable</td><td class="num">0.96</td><td class="num">0.96</td><td class="num">0.96</td><td class="num">724</td></tr>
            <tr><td>Unstable</td><td class="num">0.98</td><td class="num">0.98</td><td class="num">0.98</td><td class="num">1,276</td></tr>
        </table>
        """
    )

    show('<div class="section-title">Algorithm comparison</div>')
    show(
        '<p class="section-note">Test accuracy for each algorithm before tuning, with the tuned SVC '
        "added at the top. Tuning raised the SVC from 96.35% to 97.20%.</p>"
    )
    rows = "".join(
        f'<div class="bar-row {css}"><div class="bar-name">{name}</div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{acc * 100:.2f}%"></div></div>'
        f'<div class="bar-val">{acc:.2%}</div></div>'
        for name, acc, css in MODEL_RESULTS
    )
    show(f'<div class="bars">{rows}</div>')
    show(
        '<div class="note">The cross-validation macro F1 during tuning was 95.59%. '
        "The tuned test F1 is close to it, which suggests the model is not overfitting.</div>"
    )


# ------------------------------------------------------------
# About
# ------------------------------------------------------------
def render_project_details():
    page_title(
        "Methodology and dataset",
        "The data preparation and modelling decisions behind this application.",
    )

    stat_strip(
        [("10,000", "observations"), ("12", "input features"), ("2", "target classes")]
    )

    show('<div class="section-title">Input features</div>')
    show(
        """
        <table class="tbl">
            <tr><th>Features</th><th>Meaning</th><th>Training range</th></tr>
            <tr><td>tau1 to tau4</td><td>Reaction time of each participant (node 1 is the producer)</td><td>0.5 to 10 s</td></tr>
            <tr><td>p1 to p4</td><td>Power produced (positive) or consumed (negative); p1 = \u2212(p2 + p3 + p4)</td><td>p1 about 1.6 to 5.9, p2 to p4 \u22122 to \u22120.5</td></tr>
            <tr><td>g1 to g4</td><td>Coefficient proportional to price elasticity</td><td>0.05 to 1</td></tr>
        </table>
        """
    )

    show('<div class="section-title">Workflow</div>')
    steps = [
        ("Data inspection", "Reviewed data types, class distribution, missing values and feature distributions."),
        ("Data preparation", "Filled missing numerical values with column medians and capped the single detected p1 IQR outlier."),
        ("Exploratory analysis", "Examined distributions, correlations and selected feature relationships."),
        ("Train-test split", "Used an 80/20 stratified split to preserve the class proportions."),
        ("Model comparison", "Compared nine classification algorithms using accuracy, precision, recall and F1-score."),
        ("SVC optimization", "Applied RandomizedSearchCV with 5-fold CV and macro F1 scoring."),
        ("Final evaluation", "Evaluated the selected SVC on the held-out test set and saved the model with its scaler."),
    ]
    items = "".join(
        f'<li><span class="tl-n">{n}</span><b>{head}</b><p>{text}</p></li>'
        for n, (head, text) in enumerate(steps, start=1)
    )
    show(f'<ol class="tl">{items}</ol>')

    show('<div class="section-title">Selected SVC settings</div>')
    show(
        """
        <div class="defs">
            <div><span>kernel</span><span>rbf</span></div>
            <div><span>C</span><span>10</span></div>
            <div><span>gamma</span><span>auto</span></div>
            <div><span>CV folds</span><span>5</span></div>
            <div><span>Scoring</span><span>f1_macro</span></div>
        </div>
        """
    )

    show('<div class="section-title">Notes on interpretation</div>')
    st.write(
        "- The target is `stabf`, with the classes `stable` and `unstable`.\n"
        "- The related numeric column `stab` was excluded from the input features "
        "to prevent target leakage.\n"
        "- The scaler is fitted on the training data and reused for prediction inputs.\n"
        "- SMOTE was not used in the final SVC workflow.\n"
        "- The application uses the tuned SVC saved from the notebook."
    )
    show(
        '<div class="note">This is an academic machine-learning demonstration based on simulated data. '
        "It must not be used to make decisions about real electrical infrastructure.</div>"
    )


# ------------------------------------------------------------
# App shell
# ------------------------------------------------------------
def main():
    inject_styles()
    st.session_state.setdefault("page", NAV[0])

    with st.sidebar:
        show(
            f"""
            <div class="brand">{svg_img(logo_svg(), "Grid Stability logo")}
                <div class="brand-name">Electrical Grid<br>Stability</div></div>
            <div class="brand-sub">Electrical stability analytics</div>
            """
        )
        for name in NAV:
            st.button(
                name, key=f"nav_{name}", on_click=go_to, args=(name,),
                type="primary" if st.session_state["page"] == name else "secondary",
            )
        st.markdown("---")
        st.caption("Model: tuned SVC")
        st.caption("Educational project")

    try:
        model, scaler = load_artifacts()
    except Exception as exc:
        st.error("The model files could not be loaded.")
        st.code(str(exc))
        st.info(
            "From your notebook, save the tuned model as `svc_model.sav` and the fitted "
            "StandardScaler as `svc_scaler.sav`. Place both files beside `app.py`."
        )
        st.stop()

    page = st.session_state["page"]
    if page == "Overview":
        render_overview()
    elif page == "Stability Predictor":
        render_predict(model, scaler)
    elif page == "Model Performance":
        render_performance()
    else:
        render_project_details()


if __name__ == "__main__":
    main()
