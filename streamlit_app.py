import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import requests
import gzip
import io
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Armington Trade Model | Realterm",
    page_icon="🏭",
    layout="wide"
)

# -----------------------------------------------------------
# DARK MODE + REALTERM BRANDING CSS
# -----------------------------------------------------------
st.markdown("""
<style>
  /* ── Dark mode base ── */
  [data-testid="stAppViewContainer"] {
    background-color: #0f1117;
    color: #e8edf2;
  }
  [data-testid="stSidebar"] { background-color: #1a1d27; }
  [data-testid="stHeader"]  { background-color: #0f1117; }

  /* ── Realterm header banner ── */
  .realterm-header {
    background: linear-gradient(135deg, #1a2744 0%, #0d1a33 60%, #1a3a5c 100%);
    border-bottom: 3px solid #c8a96e;
    padding: 18px 32px;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .realterm-logo-text {
    font-size: 28px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 2px;
    font-family: 'Georgia', serif;
  }
  .realterm-logo-text span {
    color: #c8a96e;
  }
  .realterm-tagline {
    font-size: 12px;
    color: #a0aec0;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 2px;
  }
  .realterm-badge {
    background: rgba(200,169,110,0.15);
    border: 1px solid #c8a96e;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 11px;
    color: #c8a96e;
    letter-spacing: 1px;
    text-transform: uppercase;
  }

  /* ── Card-style sections ── */
  .stMarkdown h3 { color: #c8a96e !important; }
  .stMarkdown h2 { color: #e8edf2 !important; }

  /* ── Tables ── */
  table { background: #1a1d27 !important; }
  tr:nth-child(even) td { background: #1e2235 !important; }

  /* ── Inputs dark ── */
  [data-testid="stNumberInput"] input,
  [data-testid="stSelectbox"] select {
    background-color: #1e2235 !important;
    color: #e8edf2 !important;
    border-color: #2d3450 !important;
  }

  /* ── Primary button Realterm gold ── */
  [data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #c8a96e, #a07840) !important;
    color: #0d1a33 !important;
    font-weight: 700 !important;
    border: none !important;
    font-size: 16px !important;
  }
  [data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #d4b87e, #b08850) !important;
  }

  /* ── Dividers ── */
  hr { border-color: #2d3450 !important; }

  /* ── Captions and info ── */
  .stCaption { color: #a0aec0 !important; }
</style>

<div class="realterm-header">
  <div>
    <div class="realterm-logo-text">REAL<span>TERM</span></div>
    <div class="realterm-tagline">Investments that keep the world moving</div>
  </div>
  <div class="realterm-badge">MIT Capstone 2026 — Armington Trade Model</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
GITHUB_BASE = "https://raw.githubusercontent.com/afosgard3/MIT_Capstone_2026_Realterm/main"
PRODUCT_COUNTRY_PATH      = f"{GITHUB_BASE}/Product-Country-Mapping-All-Clean.csv.gz"
HS4_COUNTRY_BASELINE_PATH = f"{GITHUB_BASE}/HS4_Top_10_Countries_Time_Weighted_Baseline.csv"
HS2_COUNTRY_BASELINE_PATH = f"{GITHUB_BASE}/HS2_Top_10_Countries_Time_Weighted_Baseline.csv"
HS4_PORT_PATH             = f"{GITHUB_BASE}/HS4-Country-Top10-Ports-Time-Weighted.csv.gz"
HS2_PORT_PATH             = f"{GITHUB_BASE}/HS2-Country-Top10-Ports-Time-Weighted.csv"
ELASTICITY_DATA_PATH      = f"{GITHUB_BASE}/Elasticity_Data.csv"

HS2_LABELS = {
    33: "Personal care and cleaning",        34: "Personal care and cleaning",
    39: "Plastics and articles thereof",
    48: "Paper goods and printed materials", 49: "Paper goods and printed materials",
    61: "Apparel and textiles",              62: "Apparel and textiles",
    63: "Apparel and textiles",              64: "Footwear",
    70: "Glassware",                         73: "Articles of iron or steel",
    76: "Articles of aluminum",              82: "Tools and cutlery",
    83: "Miscellaneous articles of base metal",
    84: "Machinery and electrical equipment", 85: "Machinery and electrical equipment",
    87: "Automotive and bicycle parts",       94: "Furniture and bedding",
    95: "Toys, sporting goods, and misc. manufactured articles",
    96: "Toys, sporting goods, and misc. manufactured articles",
}

PRESET_SIGMA = {
    33: 5.0, 34: 5.0, 39: 4.5, 48: 3.5, 49: 6.0,
    61: 7.5, 62: 7.0, 63: 6.0, 64: 6.5, 70: 2.5,
    73: 4.0, 76: 4.0, 82: 4.5, 83: 5.0, 84: 3.0,
    85: 4.5, 87: 2.0, 94: 5.5, 95: 6.5, 96: 5.0,
}
SIGMA_OVERRIDES = {70: 2.5}

PORT_COORDS = {
    "Anchorage, AK":          (61.2181,  -149.9003),
    "Baltimore, MD":          (39.2904,   -76.6122),
    "Boston, MA":             (42.3601,   -71.0589),
    "Buffalo, NY":            (42.8864,   -78.8784),
    "Charleston, SC":         (32.7765,   -79.9311),
    "Charlotte, NC":          (35.2271,   -80.8431),
    "Chicago, IL":            (41.8781,   -87.6298),
    "Cleveland, OH":          (41.4993,   -81.6944),
    "Columbia-Snake, OR":     (46.1879,  -123.8313),
    "Dallas-Fort Worth, TX":  (32.8998,   -97.0403),
    "Denver, CO":             (39.7392,  -104.9903),
    "Detroit, MI":            (42.3314,   -83.0458),
    "El Paso, TX":            (31.7619,  -106.4850),
    "Great Falls, MT":        (47.4999,  -111.3008),
    "Honolulu, HI":           (21.3069,  -157.8583),
    "Houston-Galveston, TX":  (29.7604,   -95.3698),
    "Laredo, TX":             (27.5306,   -99.4803),
    "Los Angeles, CA":        (33.9519,  -118.2485),
    "Miami, FL":              (25.7617,   -80.1918),
    "Milwaukee, WI":          (43.0389,   -87.9065),
    "Minneapolis, MN":        (44.9778,   -93.2650),
    "Mobile, AL":             (30.6954,   -88.0399),
    "New Orleans, LA":        (29.9511,   -90.0715),
    "New York City, NY":      (40.7128,   -74.0060),
    "Nogales, AZ":            (31.3402,  -110.9340),
    "Norfolk, VA":            (36.8508,   -76.2859),
    "Ogdensburg, NY":         (44.6942,   -75.4863),
    "Pembina, ND":            (48.9661,   -97.2431),
    "Philadelphia, PA":       (39.9526,   -75.1652),
    "Phoenix, AZ":            (33.4484,  -112.0740),
    "Portland, ME":           (43.6591,   -70.2568),
    "Portland, OR":           (45.5051,  -122.6750),
    "Providence, RI":         (41.8240,   -71.4128),
    "San Diego, CA":          (32.7157,  -117.1611),
    "San Francisco, CA":      (37.7749,  -122.4194),
    "San Juan, PR":           (18.4655,   -66.1057),
    "Savannah, GA":           (32.0835,   -81.0998),
    "Seattle, WA":            (47.6062,  -122.3321),
    "St. Albans, VT":         (44.8112,   -73.0832),
    "St. Louis, MO":          (38.6270,   -90.1994),
    "Tampa, FL":              (27.9506,   -82.4572),
    "Washington, DC":         (38.9072,   -77.0369),
    "Wilmington, NC":         (34.2257,   -77.9447),
    "Wilmington, DE":         (39.7447,   -75.5484),
}

REGION_MAP = {
    "West Coast":              ["Los Angeles, CA","San Francisco, CA","Seattle, WA",
                                "Columbia-Snake, OR","Portland, OR","San Diego, CA",
                                "Anchorage, AK","Honolulu, HI"],
    "Northern Border":         ["Buffalo, NY","Detroit, MI","Ogdensburg, NY",
                                "St. Albans, VT","Pembina, ND","Great Falls, MT",
                                "Minneapolis, MN","Milwaukee, WI","Chicago, IL","Cleveland, OH"],
    "Mexico Border":           ["El Paso, TX","Laredo, TX","Nogales, AZ",
                                "San Diego, CA","Phoenix, AZ","Dallas-Fort Worth, TX"],
    "Northern Atlantic Coast": ["New York City, NY","Philadelphia, PA","Baltimore, MD",
                                "Boston, MA","Norfolk, VA","Providence, RI",
                                "Portland, ME","Washington, DC","Wilmington, DE",
                                "Wilmington, NC","Charlotte, NC"],
    "Southern Coast":          ["Savannah, GA","Miami, FL","Mobile, AL",
                                "New Orleans, LA","Houston-Galveston, TX",
                                "Tampa, FL","Charleston, SC"],
}
REGION_CENTROIDS = {
    "West Coast":              (40.5,  -122.5),
    "Northern Border":         (45.5,   -87.0),
    "Mexico Border":           (30.5,  -103.5),
    "Northern Atlantic Coast": (39.5,   -75.5),
    "Southern Coast":          (29.8,   -87.0),
}
REGION_COLORS = {
    "West Coast":              "#2980b9",
    "Northern Border":         "#8e44ad",
    "Mexico Border":           "#e67e22",
    "Northern Atlantic Coast": "#16a085",
    "Southern Coast":          "#c0392b",
}

# -----------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------
@st.cache_data(show_spinner="Loading data from GitHub...")
def load_all_data():
    def _dl_csv(url):
        r = requests.get(url, stream=True)
        r.raise_for_status()
        if url.endswith(".gz"):
            buf = io.BytesIO(r.content)
            with gzip.open(buf, "rt", encoding="utf-8") as f:
                return pd.read_csv(f)
        return pd.read_csv(StringIO(r.content.decode("utf-8")))

    df_pc   = _dl_csv(PRODUCT_COUNTRY_PATH)
    df_h4cb = _dl_csv(HS4_COUNTRY_BASELINE_PATH)
    df_h2cb = _dl_csv(HS2_COUNTRY_BASELINE_PATH)
    df_h4p  = _dl_csv(HS4_PORT_PATH)
    df_h2p  = _dl_csv(HS2_PORT_PATH)
    df_elas = _dl_csv(ELASTICITY_DATA_PATH)

    for df in [df_pc, df_h4cb, df_h2cb, df_h4p, df_h2p]:
        for c in ["HS2", "HS4", "Year", "Rank"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df_pc, df_h4cb, df_h2cb, df_h4p, df_h2p, df_elas

@st.cache_data(show_spinner="Running K-means clustering...")
def compute_clustered_sigma(_df_elas):
    df = _df_elas.copy()
    month_map = {
        'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
        'July':7,'August':8,'September':9,'October':10,'November':11,'December':12,
        'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
        'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12
    }
    df['Month_Num']  = df['Month'].str.strip().map(month_map)
    df['time_index'] = (df['Year'] - 2024) * 12 + df['Month_Num']
    value_col        = [c for c in df.columns if 'customs value' in c.lower()][0]
    df               = df.rename(columns={value_col: 'value'})
    df['value']      = pd.to_numeric(df['value'], errors='coerce').fillna(0)
    df['HS_Clean']   = df['HS'].astype(str).str.zfill(2).str[:2]
    country_hs = (df.groupby(['HS_Clean','Country'])['value']
                    .agg(avg_v='mean', std_v='std', sparsity=lambda x: (x==0).sum())
                    .reset_index())
    country_hs['cv'] = country_hs['std_v'] / (country_hs['avg_v'] + 1e-9)
    global_dna = (country_hs.groupby('HS_Clean')
                             .agg(cv=('cv','mean'), sparsity=('sparsity','mean'), avg_v=('avg_v','sum'))
                             .reset_index())
    X        = global_dna[['cv','sparsity']].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)
    global_dna['global_cluster'] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_scaled)
    cv_low  = global_dna['cv'].quantile(0.25)
    cv_high = global_dna['cv'].quantile(0.75)
    sp_mid  = global_dna['sparsity'].median()
    sigma_map = {
        "Inelastic (Staple/Necessity)":            2.5,
        "Moderate Elasticity (Standard Consumer)": 3.8,
        "Highly Elastic (Volatile/Opportunity)":   5.1,
        "Unitary/Project-Based (Inconsistent)":    1.0,
    }
    def _label(row):
        if   row['cv'] <= cv_low:      return "Inelastic (Staple/Necessity)"
        elif row['cv'] >= cv_high:     return "Highly Elastic (Volatile/Opportunity)"
        elif row['sparsity'] > sp_mid: return "Unitary/Project-Based (Inconsistent)"
        else:                          return "Moderate Elasticity (Standard Consumer)"
    global_dna['elasticity_label'] = global_dna.apply(_label, axis=1)
    global_dna['sigma']            = global_dna['elasticity_label'].map(sigma_map)
    result = {int(r['HS_Clean']): round(float(r['sigma']),1) for _,r in global_dna.iterrows()}
    result.update(SIGMA_OVERRIDES)
    return result

# -----------------------------------------------------------
# FORMAT HELPERS
# -----------------------------------------------------------
def _pct(v, dec=1):
    return f"{v*100:.{dec}f}%" if pd.notna(v) else "—"

def _dollar(v):
    if pd.isna(v): return "—"
    neg = v < 0; a = abs(v)
    if   a >= 1e9: s = f"${a/1e9:,.1f}B"
    elif a >= 1e6: s = f"${a/1e6:,.1f}M"
    elif a >= 1e3: s = f"${a/1e3:,.1f}K"
    else:          s = f"${a:,.0f}"
    return f"({s})" if neg else s

def _chg_color(v):
    if v >  0.0005: return "#1a7d3a"
    if v < -0.0005: return "#c0392b"
    return "#555555"

def _sign(v): return "+" if v > 0 else ""

def _badge(val, threshold=0.005):
    if pd.isna(val): return "—"
    if   val >  threshold: bg, fg, sign = "#d4edda","#155724","+"
    elif val < -threshold: bg, fg, sign = "#fde8e8","#721c24",""
    else:                  bg, fg, sign = "#f0f0f0","#444444","+"
    return f'<span style="background:{bg};color:{fg};font-weight:bold;padding:2px 6px;border-radius:3px">{sign}{_pct(val)}</span>'

def _dbadge(val, threshold=1_000_000):
    if pd.isna(val): return "—"
    if   val >  threshold: bg, fg, sign = "#d4edda","#155724","+"
    elif val < -threshold: bg, fg, sign = "#fde8e8","#721c24",""
    else:                  bg, fg, sign = "#f0f0f0","#444444","+"
    return f'<span style="background:{bg};color:{fg};font-weight:bold;padding:2px 6px;border-radius:3px">{sign}{_dollar(val)}</span>'

THEAD = "#1a3a5c"
TS    = "border-collapse:collapse;font-size:13px;width:100%"

# -----------------------------------------------------------
# MODEL FUNCTIONS
# -----------------------------------------------------------
def get_country_baseline(df_h4cb, df_h2cb, hs2, hs4=None):
    df = df_h4cb[df_h4cb["HS4"]==hs4].copy() if hs4 is not None else df_h2cb[df_h2cb["HS2"]==hs2].copy()
    if df.empty: raise ValueError(f"No baseline data for {'HS4 '+str(hs4) if hs4 else 'HS2 '+str(hs2)}")
    df = df.rename(columns={"Old Tariff":"OldTariff"})
    return df.nsmallest(10,"Rank").reset_index(drop=True)[["Rank","Country of Origin","BaselineShare","OldTariff"]]

def get_total_imports_2025(df_pc, hs2, hs4=None):
    df = df_pc[df_pc["HS2"]==hs2].copy()
    if hs4 is not None: df = df[df["HS4"]==hs4]
    return df[df["Year"]==2025]["Customs Value $ (Consumption)"].sum()

def run_armington(baseline_df, new_tariffs_dict, sigma, total_imports):
    df = baseline_df.copy()
    df["NewTariff"]      = df["Country of Origin"].map(new_tariffs_dict).fillna(df["OldTariff"])
    df["TariffChange"]   = df["NewTariff"] - df["OldTariff"]
    df["Weight"]         = df["BaselineShare"] * ((1+df["TariffChange"]) ** (-sigma))
    sum_w                = df["Weight"].sum()
    df["NewShare"]       = df["Weight"] / sum_w
    df["ShareChangePct"] = df["NewShare"] - df["BaselineShare"]
    df["BaselineQty"]    = df["BaselineShare"] * total_imports
    df["NewQty"]         = df["NewShare"]      * total_imports
    df["GrowthValue"]    = df["NewQty"] - df["BaselineQty"]
    df["PriceFactor"]    = 1 + df["TariffChange"]
    return df

def build_port_impact(df_h4p, df_h2p, result_df, hs2, hs4=None):
    ports = df_h4p[df_h4p["HS4"]==hs4].copy() if hs4 is not None else df_h2p[df_h2p["HS2"]==hs2].copy()
    if ports.empty: return pd.DataFrame()
    m = ports.merge(result_df[["Country of Origin","BaselineShare","NewShare"]], on="Country of Origin", how="inner")
    if m.empty: return pd.DataFrame()
    m["OldContrib"] = m["BaselineShare_x"] * m["BaselineShare_y"]
    m["NewContrib"] = m["BaselineShare_x"] * m["NewShare"]
    s = m.groupby("Port of Entry", as_index=False).agg(OldShare=("OldContrib","sum"), NewShare=("NewContrib","sum"))
    s["Change"] = s["NewShare"] - s["OldShare"]
    return s.sort_values("OldShare", ascending=False).reset_index(drop=True)

def assign_region(port):
    for region, ports in REGION_MAP.items():
        if port in ports: return region
    return "Other"

def aggregate_to_regions(df):
    df = df.copy()
    df["Region"] = df["Port of Entry"].apply(assign_region)
    reg = (df[df["Region"]!="Other"]
           .groupby("Region", as_index=False)
           .agg(OldShare=("OldShare","sum"), NewShare=("NewShare","sum")))
    reg["Change"] = reg["NewShare"] - reg["OldShare"]
    return reg

# -----------------------------------------------------------
# HTML RENDERERS
# -----------------------------------------------------------
def render_baseline_html(df, label, sigma, total):
    h = (f"<h3 style='margin:14px 0 4px'>Baseline — {label}</h3>"
         f"<p style='margin:0 0 8px;color:#444'><b>2025 Total Imports:</b> {_dollar(total)}"
         f" &nbsp;|&nbsp; <b>σ = {sigma}</b>"
         f" &nbsp;|&nbsp; <span style='color:#888;font-size:12px'>Shares time-weighted 2020–2025, normalized to top-10</span></p>"
         f'<table style="{TS}"><tr style="background:{THEAD};color:white;text-align:center">'
         f'<th style="padding:6px 8px">#</th><th style="padding:6px 8px">Country</th>'
         f'<th style="padding:6px 8px">Baseline Share</th><th style="padding:6px 8px">Old Tariff</th></tr>')
    for _,r in df.iterrows():
        bg = "#fff" if int(r["Rank"])%2 else "#f7f9fc"
        h += (f'<tr style="background:{bg};text-align:right">'
              f'<td style="text-align:center;padding:5px 8px">{int(r["Rank"])}</td>'
              f'<td style="text-align:left;padding:5px 8px"><b>{r["Country of Origin"]}</b></td>'
              f'<td style="padding:5px 8px">{_pct(r["BaselineShare"])}</td>'
              f'<td style="padding:5px 8px">{_pct(r["OldTariff"])}</td></tr>')
    h += (f'<tr style="background:#e8edf2;font-weight:bold;text-align:right">'
          f'<td colspan="2" style="padding:5px 8px">Total</td>'
          f'<td style="padding:5px 8px">{_pct(df["BaselineShare"].sum())}</td>'
          f'<td style="padding:5px 8px">—</td></tr></table>')
    return h

def render_results_html(result, label, sigma, total):
    h = (f"<h3 style='margin:20px 0 4px'>📊 Results — {label} &nbsp;|&nbsp; σ = {sigma}</h3>"
         f"<p style='margin:0 0 8px;color:#444'><b>2025 Total Imports:</b> {_dollar(total)}</p>"
         f'<table style="{TS}"><tr style="background:{THEAD};color:white;text-align:center">')
    for c in ["#","Country","Baseline %","Old Tariff","New Tariff ★","Tariff Δ","New %","Share Δ","Baseline $","New $","Growth"]:
        h += f'<th style="padding:6px 8px">{c}</th>'
    h += '</tr>'
    for _,r in result.iterrows():
        bg = "#fff" if int(r["Rank"])%2 else "#f7f9fc"
        h += (f'<tr style="background:{bg};text-align:right">'
              f'<td style="text-align:center;padding:5px 8px">{int(r["Rank"])}</td>'
              f'<td style="text-align:left;padding:5px 8px"><b>{r["Country of Origin"]}</b></td>'
              f'<td style="padding:5px 8px">{_pct(r["BaselineShare"])}</td>'
              f'<td style="padding:5px 8px">{_pct(r["OldTariff"])}</td>'
              f'<td style="background:#fffde7;font-weight:bold;padding:5px 8px">{_pct(r["NewTariff"])}</td>'
              f'<td style="padding:5px 8px">{_pct(r["TariffChange"])}</td>'
              f'<td style="padding:5px 8px"><b>{_pct(r["NewShare"])}</b></td>'
              f'<td style="padding:5px 8px">{_badge(r["ShareChangePct"],0.001)}</td>'
              f'<td style="padding:5px 8px">{_dollar(r["BaselineQty"])}</td>'
              f'<td style="padding:5px 8px">{_dollar(r["NewQty"])}</td>'
              f'<td style="padding:5px 8px">{_dbadge(r["GrowthValue"])}</td></tr>')
    h += (f'<tr style="background:#e8edf2;font-weight:bold;text-align:right">'
          f'<td colspan="2" style="padding:5px 8px">✓ Checks</td>'
          f'<td style="padding:5px 8px">{_pct(result["BaselineShare"].sum())}</td>'
          f'<td colspan="4" style="padding:5px 8px">—</td>'
          f'<td style="padding:5px 8px">—</td>'
          f'<td style="padding:5px 8px">{_dollar(result["BaselineQty"].sum())}</td>'
          f'<td style="padding:5px 8px">{_dollar(result["NewQty"].sum())}</td>'
          f'<td style="padding:5px 8px">{_dbadge(result["GrowthValue"].sum())}</td></tr></table>')
    return h

def render_port_table_html(port_df, reg_df, total_imports):
    def _row(name, old_s, new_s, chg, is_region=False):
        cc = _chg_color(chg); sign = _sign(chg)
        bg = "#f0f4f8" if is_region else "#ffffff"
        fw = "bold" if is_region else "normal"
        return (f'<tr style="background:{bg};font-weight:{fw};text-align:right">'
                f'<td style="text-align:left;padding:5px 8px">{name}</td>'
                f'<td style="padding:5px 8px">{old_s*100:.2f}%</td>'
                f'<td style="padding:5px 8px">{new_s*100:.2f}%</td>'
                f'<td style="padding:5px 8px;color:{cc}">{sign}{chg*100:.2f}pp</td>'
                f'<td style="padding:5px 8px">{_dollar(old_s*total_imports)}</td>'
                f'<td style="padding:5px 8px">{_dollar(new_s*total_imports)}</td>'
                f'<td style="padding:5px 8px;color:{cc}">{sign}{_dollar(chg*total_imports)}</td></tr>')

    h  = f'<table style="{TS}"><tr style="background:{THEAD};color:white;text-align:center">'
    for c in ["Port / Region","Before","After","Change (pp)","Before ($)","After ($)","Volume Δ"]:
        h += f'<th style="padding:6px 8px">{c}</th>'
    h += '</tr>'
    port_df2           = port_df.copy()
    port_df2["Region"] = port_df2["Port of Entry"].apply(assign_region)
    for region in ["West Coast","Northern Border","Mexico Border","Northern Atlantic Coast","Southern Coast"]:
        reg_row = reg_df[reg_df["Region"]==region]
        if reg_row.empty: continue
        rr = reg_row.iloc[0]
        h += _row(f"▶ {region}", rr["OldShare"], rr["NewShare"], rr["Change"], is_region=True)
        for _,pr in port_df2[port_df2["Region"]==region].sort_values("OldShare",ascending=False).iterrows():
            h += _row(f"&nbsp;&nbsp;&nbsp;{pr['Port of Entry']}", pr["OldShare"], pr["NewShare"], pr["Change"])
    other = port_df2[port_df2["Region"]=="Other"].sort_values("OldShare",ascending=False)
    if not other.empty:
        h += _row("▶ Other / Interior", other["OldShare"].sum(), other["NewShare"].sum(), other["Change"].sum(), is_region=True)
        for _,pr in other.iterrows():
            h += _row(f"&nbsp;&nbsp;&nbsp;{pr['Port of Entry']}", pr["OldShare"], pr["NewShare"], pr["Change"])
    h += "</table>"
    return h

def build_folium_map(port_df, label, total_imports, mode="individual"):
    m = folium.Map(location=[38.5,-95.0], zoom_start=4, tiles="CartoDB positron", prefer_canvas=True)
    if mode == "individual":
        df = port_df.copy()
        df["lat"] = df["Port of Entry"].map(lambda p: PORT_COORDS.get(p,(None,None))[0])
        df["lon"] = df["Port of Entry"].map(lambda p: PORT_COORDS.get(p,(None,None))[1])
        df = df.dropna(subset=["lat","lon"])
        max_s = df["OldShare"].max() or 1
        for _,r in df.iterrows():
            chg   = r["Change"]
            fc    = "#1a7d3a" if chg>0.0005 else ("#c0392b" if chg<-0.0005 else "#7f8c8d")
            bc    = "#0f5a29" if chg>0.0005 else ("#8e1a1a" if chg<-0.0005 else "#5a6467")
            r_old = max(7,  min(50,(r["OldShare"]/max_s)**0.5*50))
            r_new = max(4,  min(50,(r["NewShare"]/max_s)**0.5*50))
            popup = folium.Popup(
                f'<b>{r["Port of Entry"]}</b><br>'
                f'Before: {r["OldShare"]*100:.2f}% → After: {r["NewShare"]*100:.2f}%<br>'
                f'<b style="color:{_chg_color(chg)}">{_sign(chg)}{chg*100:.2f}pp</b>'
                f' | Vol Δ: {_sign(chg)}{_dollar(chg*total_imports)}', max_width=260)
            folium.CircleMarker([r["lat"],r["lon"]], radius=r_old, color=bc, fill=True,
                                fill_color=fc, fill_opacity=0.2, weight=2,
                                popup=popup, tooltip=folium.Tooltip(f"{r['Port of Entry']}: {_sign(chg)}{chg*100:.2f}pp",sticky=True)).add_to(m)
            folium.CircleMarker([r["lat"],r["lon"]], radius=r_new, color=bc, fill=True,
                                fill_color=fc, fill_opacity=0.85, weight=1.5).add_to(m)
    else:
        reg_df = aggregate_to_regions(port_df)
        max_s  = reg_df["OldShare"].max() or 1
        for _,r in reg_df.iterrows():
            region = r["Region"]; chg = r["Change"]
            color  = REGION_COLORS.get(region,"#555")
            fc     = "#1a7d3a" if chg>0.0005 else ("#c0392b" if chg<-0.0005 else "#7f8c8d")
            bc     = "#0f5a29" if chg>0.0005 else ("#8e1a1a" if chg<-0.0005 else "#5a6467")
            lat,lon = REGION_CENTROIDS.get(region,(38,-95))
            r_old  = max(15, min(70,(r["OldShare"]/max_s)**0.5*70))
            r_new  = max(8,  min(70,(r["NewShare"]/max_s)**0.5*70))
            ports_in = [p for p in REGION_MAP.get(region,[]) if p in port_df["Port of Entry"].values]
            popup = folium.Popup(
                f'<b style="color:{color}">{region}</b><br>'
                f'Before: {r["OldShare"]*100:.2f}% → After: {r["NewShare"]*100:.2f}%<br>'
                f'<b style="color:{_chg_color(chg)}">{_sign(chg)}{chg*100:.2f}pp</b>'
                f' | Vol Δ: {_sign(chg)}{_dollar(chg*total_imports)}<br>'
                f'<small>{"  •  ".join(ports_in)}</small>', max_width=280)
            folium.CircleMarker([lat,lon], radius=r_old, color=bc, fill=True,
                                fill_color=fc, fill_opacity=0.2, weight=2.5,
                                popup=popup, tooltip=folium.Tooltip(f"{region}: {_sign(chg)}{chg*100:.2f}pp",sticky=True)).add_to(m)
            folium.CircleMarker([lat,lon], radius=r_new, color=bc, fill=True,
                                fill_color=fc, fill_opacity=0.85, weight=2).add_to(m)
            folium.Marker([lat+1.8,lon], icon=folium.DivIcon(
                html=f'<div style="font-family:Arial;font-size:11px;font-weight:bold;color:{color};white-space:nowrap">{region}</div>',
                icon_size=(180,20), icon_anchor=(90,10))).add_to(m)
    legend = (f'<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;'
              f'padding:14px 18px;border-radius:8px;border:1px solid #ccc;font-family:Arial;font-size:12px">'
              f'<b>{label}</b><br><br>'
              f'<span style="color:#1a7d3a">●</span> Gaining &nbsp;'
              f'<span style="color:#c0392b">●</span> Losing &nbsp;'
              f'<span style="color:#7f8c8d">●</span> Flat<br>'
              f'<span style="font-size:11px;color:#666">Outer ring = before &nbsp; Filled = after</span></div>')
    m.get_root().html.add_child(folium.Element(legend))
    return m

# ===========================================================
# SESSION STATE INIT
# ===========================================================
if "results" not in st.session_state:
    st.session_state.results = None
if "port_df" not in st.session_state:
    st.session_state.port_df = None
if "reg_df" not in st.session_state:
    st.session_state.reg_df = None
if "result_label" not in st.session_state:
    st.session_state.result_label = None
if "result_total" not in st.session_state:
    st.session_state.result_total = None
if "result_sigma" not in st.session_state:
    st.session_state.result_sigma = None

# ===========================================================
# LOAD DATA
# ===========================================================
try:
    df_pc, df_h4cb, df_h2cb, df_h4p, df_h2p, df_elas = load_all_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

try:
    clustered_sigma = compute_clustered_sigma(df_elas)
except Exception as e:
    clustered_sigma = {}
    st.warning(f"Clustering failed, using presets: {e}")

# ===========================================================
# UI
# ===========================================================
st.title("🏭 Armington Trade Model")
st.markdown("*Constant Elasticity of Substitution — tariff impact on country & port shares*")
st.markdown("**Baseline shares:** time-weighted 2020–2025 (weights 1–6), normalized to top-10 = 100%")
st.divider()

# -----------------------------------------------------------
# STEP 1 — SELECTION
# -----------------------------------------------------------
st.subheader("Step 1 — Select Product, Level & Elasticity")
col1, col2 = st.columns([2, 1])

with col1:
    available_hs2 = sorted(df_pc["HS2"].dropna().unique().astype(int).tolist())
    hs2_options   = {f"HS2 {h} — {HS2_LABELS.get(h,'')}": h for h in available_hs2}
    hs2_label     = st.selectbox("HS2 Category", options=list(hs2_options.keys()))
    hs2           = hs2_options[hs2_label]

    level = st.radio("Level", ["HS2 level (full category)", "HS4 level (specific product)"], horizontal=True)

    hs4 = None
    if "HS4" in level:
        sort_by_value = st.checkbox("Sort HS4 by 2025 Customs Value")
        sub = df_pc[df_pc["HS2"]==hs2].copy()
        sub["HS4"] = pd.to_numeric(sub["HS4"], errors="coerce")
        val25 = sub[sub["Year"]==2025].groupby("HS4")["Customs Value $ (Consumption)"].sum().rename("val25")
        meta  = sub.drop_duplicates("HS4").set_index("HS4")[["HS4 Description"]].join(val25, how="left")
        meta["val25"] = meta["val25"].fillna(0)
        codes = (meta.sort_values("val25",ascending=False).index.dropna().astype(int).tolist()
                 if sort_by_value else sorted(meta.index.dropna().astype(int).tolist()))
        desc = meta["HS4 Description"].to_dict(); val = meta["val25"].to_dict()
        def _fmt(c):
            v = val.get(c,0)
            v_str = f"  ${v/1e6:,.0f}M" if v>=1e6 else (f"  ${v/1e3:,.0f}K" if v>0 else "")
            return f"HS4 {c} — {str(desc.get(c,''))[:45]}{v_str}"
        hs4_options = {_fmt(c): c for c in codes}
        hs4_label   = st.selectbox("HS4 Product", options=list(hs4_options.keys()))
        hs4         = hs4_options[hs4_label]

with col2:
    sigma_mode = st.radio(
        "Elasticity (σ) Mode",
        ["Custom — set σ manually", "Claude AI — expert-scored σ", "Data-driven — K-means"]
    )
    if "Custom" in sigma_mode:
        sigma = st.number_input("σ value", min_value=1.0, max_value=10.0, value=4.0, step=0.1)
        st.caption("Enter any σ between 1.0 and 10.0")
    elif "Claude" in sigma_mode:
        sigma = float(PRESET_SIGMA.get(hs2, 4.0))
        st.number_input("σ value", value=sigma, disabled=True)
        st.caption(f"Claude AI scoring for HS2 {hs2}: **σ = {sigma}**")
    else:
        sigma = float(clustered_sigma.get(hs2, PRESET_SIGMA.get(hs2, 4.0)))
        st.number_input("σ value", value=sigma, disabled=True)
        st.caption(f"K-means cluster for HS2 {hs2}: **σ = {sigma}**")

st.divider()

# -----------------------------------------------------------
# STEP 2 — BASELINE & TARIFF INPUTS
# -----------------------------------------------------------
st.subheader("Step 2 — Review Baseline & Enter New Tariff Rates")

try:
    baseline = get_country_baseline(df_h4cb, df_h2cb, hs2, hs4)
    total    = get_total_imports_2025(df_pc, hs2, hs4)
    label    = (f"HS4 {hs4} — {df_pc[df_pc['HS4']==hs4]['HS4 Description'].iloc[0][:50]}"
                if hs4 else f"HS2 {hs2} — {HS2_LABELS.get(hs2,'')}")
except Exception as e:
    st.error(f"Error loading baseline: {e}")
    st.stop()

st.markdown(render_baseline_html(baseline, label, sigma, total), unsafe_allow_html=True)
st.markdown("**Enter new tariff rates (%) — pre-filled with 2025 effective rates:**")

new_tariffs = {}
cols = st.columns(2)
for i, (_,r) in enumerate(baseline.iterrows()):
    country    = r["Country of Origin"]
    old_tariff = round(float(r["OldTariff"]) * 100, 1)
    with cols[i % 2]:
        new_val = st.number_input(
            f"{int(r['Rank'])}. {country}  (Share: {_pct(r['BaselineShare'])} | Old: {old_tariff}%)",
            min_value=0.0, max_value=500.0,
            value=old_tariff, step=1.0,
            key=f"tariff_{country}_{hs2}_{hs4}"
        )
        new_tariffs[country] = new_val

st.divider()

# -----------------------------------------------------------
# STEP 3 — RUN MODEL BUTTON
# -----------------------------------------------------------
if st.button("🚀 Run Armington Model", type="primary", use_container_width=True):
    with st.spinner("Running model..."):
        try:
            result  = run_armington(baseline, {c: v/100 for c,v in new_tariffs.items()}, sigma, total)
            port_df = build_port_impact(df_h4p, df_h2p, result, hs2, hs4)
            reg_df  = aggregate_to_regions(port_df) if not port_df.empty else pd.DataFrame()

            # Store everything in session state
            st.session_state.results      = result
            st.session_state.port_df      = port_df
            st.session_state.reg_df       = reg_df
            st.session_state.result_label = label
            st.session_state.result_total = total
            st.session_state.result_sigma = sigma
        except Exception as e:
            st.error(f"Model error: {e}")

# -----------------------------------------------------------
# DISPLAY RESULTS (from session state — persists across reruns)
# -----------------------------------------------------------
if st.session_state.results is not None:
    result = st.session_state.results
    port_df = st.session_state.port_df
    reg_df  = st.session_state.reg_df
    label   = st.session_state.result_label
    total   = st.session_state.result_total
    sigma   = st.session_state.result_sigma

    st.subheader("📊 Armington Results")
    st.markdown(render_results_html(result, label, sigma, total), unsafe_allow_html=True)
    st.divider()

    if port_df is not None and not port_df.empty:
        st.subheader("🚢 Port Impact")
        map_mode   = st.radio("Map View", ["Individual Ports", "Regional View"], horizontal=True, key="map_toggle")
        mode_key   = "individual" if "Individual" in map_mode else "regional"
        folium_map = build_folium_map(port_df, label, total, mode_key)
        st_folium(folium_map, width=1200, height=550, returned_objects=[])
        st.markdown("**Port Impact Summary**")
        st.markdown(render_port_table_html(port_df, reg_df, total), unsafe_allow_html=True)
    else:
        st.info("No port data found for this selection.")
