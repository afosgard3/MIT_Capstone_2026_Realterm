import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import requests
import gzip
import io
import zipfile
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Armington Trade Model | Realterm",
    page_icon="🏭",
    layout="wide"
)

LOGO_B64 = "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"

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background-color: #0f1117; color: #f0f4f8; }
  [data-testid="stHeader"] { background-color: #0f1117 !important; border-bottom: 1px solid #1e2235; }
  p, li { color: #f0f4f8 !important; }
  label { color: #e8edf2 !important; font-size: 14px !important; }
  .stMarkdown p { color: #f0f4f8 !important; }
  .stMarkdown h1, .stMarkdown h2 { color: #ffffff !important; }
  .stMarkdown h3 { color: #c8a96e !important; }
  .stCaption, small { color: #c0cad6 !important; }
  [data-testid="stSelectbox"] * { color: #f0f4f8 !important; }
  [data-testid="stSelectbox"] > div > div > div { background-color: #1e2235 !important; border-color: #3a4060 !important; }
  [data-testid="stSelectbox"] svg { fill: #c8a96e !important; }
  [data-baseweb="popover"] * { background-color: #1e2235 !important; color: #f0f4f8 !important; }
  [data-baseweb="menu"] li:hover { background-color: #2d3450 !important; }
  [data-testid="stNumberInput"] input { background-color: #1e2235 !important; color: #f0f4f8 !important; border-color: #3a4060 !important; }
  [data-testid="stNumberInput"] label { color: #c0cad6 !important; }
  /* Radio — target all possible selectors to override Streamlit red */
  [data-testid="stRadio"] label { color: #f0f4f8 !important; }
  [data-testid="stRadio"] svg circle { fill: #c8a96e !important; stroke: #c8a96e !important; }
  [data-baseweb="radio"] div { border-color: #c8a96e !important; }
  [data-baseweb="radio"] [class*="selected"] div,
  [data-baseweb="radio"] div[class*="selected"] { background: #c8a96e !important; border-color: #c8a96e !important; }
  input[type=radio]:checked ~ div { background: #c8a96e !important; }
  /* Accent color override — this changes Streamlit's primary red to gold */
  :root { --primary-color: #c8a96e !important; }
  [data-testid="stCheckbox"] label { color: #f0f4f8 !important; }
  [data-testid="stButton"] button[kind="primary"] { background: linear-gradient(135deg, #c8a96e, #a07840) !important; color: #0d1a33 !important; font-weight: 700 !important; border: none !important; font-size: 16px !important; }
  [data-testid="stDownloadButton"] button { background-color: #1e2235 !important; color: #c8a96e !important; border: 1px solid #c8a96e !important; font-weight: 600 !important; }
  hr { border-color: #2d3450 !important; }
  table { background: #13161f !important; width: 100% !important; }
  td, th { color: #f0f4f8 !important; }
  tr:nth-child(even) td { background: #1a1d2e !important; }
  tr:nth-child(odd) td { background: #13161f !important; }
</style>
""", unsafe_allow_html=True)

# Realterm header banner — rendered as normal HTML (no margin hack needed)
st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d1a33 0%,#1a2744 100%);border:1px solid #2d3a5c;border-radius:8px;padding:18px 32px;margin-bottom:24px;display:flex;align-items:center;justify-content:space-between;">
  <div style="display:flex;align-items:center;gap:24px;">
    <img src="data:image/png;base64,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" style="height:30px;width:auto;filter:brightness(0) invert(1);" />
    <div style="width:1px;height:36px;background:rgba(200,169,110,0.5);"></div>
    <div>
      <div style="font-size:15px;color:#f0f4f8;font-weight:700;letter-spacing:0.5px;">Armington Trade Model</div>
      <div style="font-size:11px;color:#c8a96e;letter-spacing:2px;text-transform:uppercase;margin-top:3px;">MIT Capstone 2026</div>
    </div>
  </div>
  <div style="font-size:11px;color:#8090a8;letter-spacing:1px;text-transform:uppercase;text-align:right;line-height:2;">Investments that keep<br>the world moving</div>
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
    33:"Personal care and cleaning",34:"Personal care and cleaning",
    39:"Plastics and articles thereof",
    48:"Paper goods and printed materials",49:"Paper goods and printed materials",
    61:"Apparel and textiles",62:"Apparel and textiles",63:"Apparel and textiles",
    64:"Footwear",70:"Glassware",73:"Articles of iron or steel",
    76:"Articles of aluminum",82:"Tools and cutlery",
    83:"Miscellaneous articles of base metal",
    84:"Machinery and electrical equipment",85:"Machinery and electrical equipment",
    87:"Automotive and bicycle parts",94:"Furniture and bedding",
    95:"Toys, sporting goods, and misc. manufactured articles",
    96:"Toys, sporting goods, and misc. manufactured articles",
}

PRESET_SIGMA = {
    33:5.0,34:5.0,39:4.5,48:3.5,49:6.0,61:7.5,62:7.0,63:6.0,64:6.5,70:2.5,
    73:4.0,76:4.0,82:4.5,83:5.0,84:3.0,85:4.5,87:2.0,94:5.5,95:6.5,96:5.0,
}
SIGMA_OVERRIDES = {70: 2.5}

PORT_COORDS = {
    "Anchorage, AK":(61.2181,-149.9003),"Baltimore, MD":(39.2904,-76.6122),
    "Boston, MA":(42.3601,-71.0589),"Buffalo, NY":(42.8864,-78.8784),
    "Charleston, SC":(32.7765,-79.9311),"Charlotte, NC":(35.2271,-80.8431),
    "Chicago, IL":(41.8781,-87.6298),"Cleveland, OH":(41.4993,-81.6944),
    "Columbia-Snake, OR":(46.1879,-123.8313),"Dallas-Fort Worth, TX":(32.8998,-97.0403),
    "Denver, CO":(39.7392,-104.9903),"Detroit, MI":(42.3314,-83.0458),
    "El Paso, TX":(31.7619,-106.4850),"Great Falls, MT":(47.4999,-111.3008),
    "Honolulu, HI":(21.3069,-157.8583),"Houston-Galveston, TX":(29.7604,-95.3698),
    "Laredo, TX":(27.5306,-99.4803),"Los Angeles, CA":(33.9519,-118.2485),
    "Miami, FL":(25.7617,-80.1918),"Milwaukee, WI":(43.0389,-87.9065),
    "Minneapolis, MN":(44.9778,-93.2650),"Mobile, AL":(30.6954,-88.0399),
    "New Orleans, LA":(29.9511,-90.0715),"New York City, NY":(40.7128,-74.0060),
    "Nogales, AZ":(31.3402,-110.9340),"Norfolk, VA":(36.8508,-76.2859),
    "Ogdensburg, NY":(44.6942,-75.4863),"Pembina, ND":(48.9661,-97.2431),
    "Philadelphia, PA":(39.9526,-75.1652),"Phoenix, AZ":(33.4484,-112.0740),
    "Portland, ME":(43.6591,-70.2568),"Portland, OR":(45.5051,-122.6750),
    "Providence, RI":(41.8240,-71.4128),"San Diego, CA":(32.7157,-117.1611),
    "San Francisco, CA":(37.7749,-122.4194),"San Juan, PR":(18.4655,-66.1057),
    "Savannah, GA":(32.0835,-81.0998),"Seattle, WA":(47.6062,-122.3321),
    "St. Albans, VT":(44.8112,-73.0832),"St. Louis, MO":(38.6270,-90.1994),
    "Tampa, FL":(27.9506,-82.4572),"Washington, DC":(38.9072,-77.0369),
    "Wilmington, NC":(34.2257,-77.9447),"Wilmington, DE":(39.7447,-75.5484),
}

REGION_MAP = {
    "West Coast":["Los Angeles, CA","San Francisco, CA","Seattle, WA","Columbia-Snake, OR","Portland, OR","San Diego, CA","Anchorage, AK","Honolulu, HI"],
    "Northern Border":["Buffalo, NY","Detroit, MI","Ogdensburg, NY","St. Albans, VT","Pembina, ND","Great Falls, MT","Minneapolis, MN","Milwaukee, WI","Chicago, IL","Cleveland, OH"],
    "Mexico Border":["El Paso, TX","Laredo, TX","Nogales, AZ","San Diego, CA","Phoenix, AZ","Dallas-Fort Worth, TX"],
    "Northern Atlantic Coast":["New York City, NY","Philadelphia, PA","Baltimore, MD","Boston, MA","Norfolk, VA","Providence, RI","Portland, ME","Washington, DC","Wilmington, DE","Wilmington, NC","Charlotte, NC"],
    "Southern Coast":["Savannah, GA","Miami, FL","Mobile, AL","New Orleans, LA","Houston-Galveston, TX","Tampa, FL","Charleston, SC"],
}
REGION_CENTROIDS = {"West Coast":(40.5,-122.5),"Northern Border":(45.5,-87.0),"Mexico Border":(30.5,-103.5),"Northern Atlantic Coast":(39.5,-75.5),"Southern Coast":(29.8,-87.0)}
REGION_COLORS    = {"West Coast":"#2980b9","Northern Border":"#8e44ad","Mexico Border":"#e67e22","Northern Atlantic Coast":"#16a085","Southern Coast":"#c0392b"}

# -----------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------
@st.cache_data(show_spinner="Loading data from GitHub...")
def load_all_data():
    def _dl(url):
        r = requests.get(url, stream=True); r.raise_for_status()
        if url.endswith(".gz"):
            with gzip.open(io.BytesIO(r.content), "rt", encoding="utf-8") as f:
                return pd.read_csv(f)
        return pd.read_csv(StringIO(r.content.decode("utf-8")))
    df_pc=_dl(PRODUCT_COUNTRY_PATH); df_h4cb=_dl(HS4_COUNTRY_BASELINE_PATH)
    df_h2cb=_dl(HS2_COUNTRY_BASELINE_PATH); df_h4p=_dl(HS4_PORT_PATH)
    df_h2p=_dl(HS2_PORT_PATH); df_elas=_dl(ELASTICITY_DATA_PATH)
    for df in [df_pc,df_h4cb,df_h2cb,df_h4p,df_h2p]:
        for c in ["HS2","HS4","Year","Rank"]:
            if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df_pc,df_h4cb,df_h2cb,df_h4p,df_h2p,df_elas

@st.cache_data(show_spinner="Running K-means clustering...")
def compute_clustered_sigma(_df):
    df=_df.copy()
    mm={'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,'November':11,'December':12,'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    df['Month_Num']=df['Month'].str.strip().map(mm)
    df['time_index']=(df['Year']-2024)*12+df['Month_Num']
    vc=[c for c in df.columns if 'customs value' in c.lower()][0]
    df=df.rename(columns={vc:'value'}); df['value']=pd.to_numeric(df['value'],errors='coerce').fillna(0)
    df['HS_Clean']=df['HS'].astype(str).str.zfill(2).str[:2]
    ch=(df.groupby(['HS_Clean','Country'])['value'].agg(avg_v='mean',std_v='std',sparsity=lambda x:(x==0).sum()).reset_index())
    ch['cv']=ch['std_v']/(ch['avg_v']+1e-9)
    gd=(ch.groupby('HS_Clean').agg(cv=('cv','mean'),sparsity=('sparsity','mean'),avg_v=('avg_v','sum')).reset_index())
    X=StandardScaler().fit_transform(gd[['cv','sparsity']].fillna(0))
    gd['cluster']=KMeans(n_clusters=4,random_state=42,n_init=10).fit_predict(X)
    cvl=gd['cv'].quantile(0.25); cvh=gd['cv'].quantile(0.75); spm=gd['sparsity'].median()
    sm={"Inelastic":2.5,"Moderate":3.8,"Elastic":5.1,"ProjectBased":1.0}
    def _lbl(r):
        if r['cv']<=cvl: return "Inelastic"
        elif r['cv']>=cvh: return "Elastic"
        elif r['sparsity']>spm: return "ProjectBased"
        else: return "Moderate"
    gd['label']=gd.apply(_lbl,axis=1); gd['sigma']=gd['label'].map(sm)
    res={int(r['HS_Clean']):round(float(r['sigma']),1) for _,r in gd.iterrows()}
    res.update(SIGMA_OVERRIDES); return res

# -----------------------------------------------------------
# HELPERS
# -----------------------------------------------------------
def _pct(v,dec=1): return f"{v*100:.{dec}f}%" if pd.notna(v) else "—"
def _dollar(v):
    if pd.isna(v): return "—"
    neg=v<0; a=abs(v)
    if a>=1e9: s=f"${a/1e9:,.1f}B"
    elif a>=1e6: s=f"${a/1e6:,.1f}M"
    elif a>=1e3: s=f"${a/1e3:,.1f}K"
    else: s=f"${a:,.0f}"
    return f"({s})" if neg else s
def _cc(v): return "#1a7d3a" if v>0.0005 else ("#c0392b" if v<-0.0005 else "#555555")
def _sign(v): return "+" if v>0 else ""
def _badge(v,t=0.005):
    if pd.isna(v): return "—"
    if v>t: bg,fg,s="#d4edda","#155724","+"
    elif v<-t: bg,fg,s="#fde8e8","#721c24",""
    else: bg,fg,s="#f0f0f0","#444","+"
    return f'<span style="background:{bg};color:{fg};font-weight:bold;padding:2px 6px;border-radius:3px">{s}{_pct(v)}</span>'
def _dbadge(v,t=1_000_000):
    if pd.isna(v): return "—"
    if v>t: bg,fg,s="#d4edda","#155724","+"
    elif v<-t: bg,fg,s="#fde8e8","#721c24",""
    else: bg,fg,s="#f0f0f0","#444","+"
    return f'<span style="background:{bg};color:{fg};font-weight:bold;padding:2px 6px;border-radius:3px">{s}{_dollar(v)}</span>'

THEAD="#1a3a5c"; TS="border-collapse:collapse;font-size:13px;width:100%"

# -----------------------------------------------------------
# MODEL
# -----------------------------------------------------------
def get_baseline(df_h4cb,df_h2cb,hs2,hs4=None):
    df=df_h4cb[df_h4cb["HS4"]==hs4].copy() if hs4 else df_h2cb[df_h2cb["HS2"]==hs2].copy()
    if df.empty: raise ValueError(f"No data for {'HS4 '+str(hs4) if hs4 else 'HS2 '+str(hs2)}")
    df=df.rename(columns={"Old Tariff":"OldTariff"})
    return df.nsmallest(10,"Rank").reset_index(drop=True)[["Rank","Country of Origin","BaselineShare","OldTariff"]]

def get_total(df_pc,hs2,hs4=None):
    df=df_pc[df_pc["HS2"]==hs2].copy()
    if hs4: df=df[df["HS4"]==hs4]
    return df[df["Year"]==2025]["Customs Value $ (Consumption)"].sum()

def run_model(baseline,tariffs,sigma,total):
    df=baseline.copy()
    df["NewTariff"]=df["Country of Origin"].map(tariffs).fillna(df["OldTariff"])
    df["TariffChange"]=df["NewTariff"]-df["OldTariff"]
    df["Weight"]=df["BaselineShare"]*((1+df["TariffChange"])**(-sigma))
    sw=df["Weight"].sum(); df["NewShare"]=df["Weight"]/sw
    df["ShareChangePct"]=df["NewShare"]-df["BaselineShare"]
    df["BaselineQty"]=df["BaselineShare"]*total; df["NewQty"]=df["NewShare"]*total
    df["GrowthValue"]=df["NewQty"]-df["BaselineQty"]; df["PriceFactor"]=1+df["TariffChange"]
    return df

def get_ports(df_h4p,df_h2p,result,hs2,hs4=None):
    p=df_h4p[df_h4p["HS4"]==hs4].copy() if hs4 else df_h2p[df_h2p["HS2"]==hs2].copy()
    if p.empty: return pd.DataFrame()
    m=p.merge(result[["Country of Origin","BaselineShare","NewShare"]],on="Country of Origin",how="inner")
    if m.empty: return pd.DataFrame()
    m["OldContrib"]=m["BaselineShare_x"]*m["BaselineShare_y"]
    m["NewContrib"]=m["BaselineShare_x"]*m["NewShare"]
    s=m.groupby("Port of Entry",as_index=False).agg(OldShare=("OldContrib","sum"),NewShare=("NewContrib","sum"))
    s["Change"]=s["NewShare"]-s["OldShare"]
    return s.sort_values("OldShare",ascending=False).reset_index(drop=True)

def assign_region(p):
    for r,ports in REGION_MAP.items():
        if p in ports: return r
    return "Other"

def agg_regions(df):
    df=df.copy(); df["Region"]=df["Port of Entry"].apply(assign_region)
    reg=(df[df["Region"]!="Other"].groupby("Region",as_index=False).agg(OldShare=("OldShare","sum"),NewShare=("NewShare","sum")))
    reg["Change"]=reg["NewShare"]-reg["OldShare"]; return reg

# -----------------------------------------------------------
# HTML RENDERERS
# -----------------------------------------------------------
def render_baseline(df,label,sigma,total):
    h=(f"<h3 style='margin:14px 0 4px'>Baseline — {label}</h3>"
       f"<p style='margin:0 0 8px;color:#c0cad6'><b style='color:#f0f4f8'>2025 Total Imports:</b> {_dollar(total)}"
       f" &nbsp;|&nbsp; <b style='color:#f0f4f8'>σ = {sigma}</b>"
       f" &nbsp;|&nbsp; <span style='color:#888;font-size:12px'>Shares time-weighted 2020–2025, normalized to top-10</span></p>"
       f'<table style="{TS}"><tr style="background:{THEAD};color:white;text-align:center">')
    h+=f'<th style="padding:6px 8px;text-align:center">#</th>'
    h+=f'<th style="padding:6px 8px;text-align:left">Country</th>'
    h+=f'<th style="padding:6px 8px;text-align:right">Baseline Share</th>'
    h+=f'<th style="padding:6px 8px;text-align:right">Old Tariff</th>'
    h+='</tr>'
    for _,r in df.iterrows():
        bg="#1a1d2e" if int(r["Rank"])%2 else "#13161f"
        h+=(f'<tr style="background:{bg}">'
            f'<td style="text-align:center;padding:5px 8px;color:#f0f4f8">{int(r["Rank"])}</td>'
            f'<td style="text-align:left;padding:5px 12px;color:#f0f4f8"><b>{r["Country of Origin"]}</b></td>'
            f'<td style="text-align:right;padding:5px 12px;color:#f0f4f8">{_pct(r["BaselineShare"])}</td>'
            f'<td style="text-align:right;padding:5px 12px;color:#f0f4f8">{_pct(r["OldTariff"])}</td></tr>')
    h+=(f'<tr style="background:#1e2740;font-weight:bold;text-align:right">'
        f'<td colspan="2" style="padding:5px 8px;color:#c8a96e">Total</td>'
        f'<td style="padding:5px 8px;color:#c8a96e">{_pct(df["BaselineShare"].sum())}</td>'
        f'<td style="padding:5px 8px;color:#c8a96e">—</td></tr></table>')
    return h

def render_results(result,label,sigma,total):
    h=(f"<h3 style='margin:20px 0 4px'>📊 Results — {label} &nbsp;|&nbsp; σ = {sigma}</h3>"
       f"<p style='margin:0 0 8px;color:#c0cad6'><b style='color:#f0f4f8'>2025 Total Imports:</b> {_dollar(total)}</p>"
       f'<table style="{TS}"><tr style="background:{THEAD};color:white;text-align:center">')
    for c in ["#","Country","Baseline %","Old Tariff","New Tariff ★","Tariff Δ","New %","Share Δ","Baseline $","New $","Growth"]:
        h+=f'<th style="padding:6px 8px">{c}</th>'
    h+='</tr>'
    for _,r in result.iterrows():
        bg="#1a1d2e" if int(r["Rank"])%2 else "#13161f"
        h+=(f'<tr style="background:{bg};text-align:right">'
            f'<td style="text-align:center;padding:5px 8px;color:#f0f4f8">{int(r["Rank"])}</td>'
            f'<td style="text-align:left;padding:5px 8px;color:#f0f4f8"><b>{r["Country of Origin"]}</b></td>'
            f'<td style="padding:5px 8px;color:#f0f4f8">{_pct(r["BaselineShare"])}</td>'
            f'<td style="padding:5px 8px;color:#f0f4f8">{_pct(r["OldTariff"])}</td>'
            f'<td style="background:#2a2510;font-weight:bold;padding:5px 8px;color:#f0e080">{_pct(r["NewTariff"])}</td>'
            f'<td style="padding:5px 8px;color:#f0f4f8">{_pct(r["TariffChange"])}</td>'
            f'<td style="padding:5px 8px;color:#f0f4f8"><b>{_pct(r["NewShare"])}</b></td>'
            f'<td style="padding:5px 8px">{_badge(r["ShareChangePct"],0.001)}</td>'
            f'<td style="padding:5px 8px;color:#f0f4f8">{_dollar(r["BaselineQty"])}</td>'
            f'<td style="padding:5px 8px;color:#f0f4f8">{_dollar(r["NewQty"])}</td>'
            f'<td style="padding:5px 8px">{_dbadge(r["GrowthValue"])}</td></tr>')
    h+=(f'<tr style="background:#1e2740;font-weight:bold;text-align:right">'
        f'<td colspan="2" style="padding:5px 8px;color:#c8a96e">✓ Checks</td>'
        f'<td style="padding:5px 8px;color:#c8a96e">{_pct(result["BaselineShare"].sum())}</td>'
        f'<td colspan="4" style="padding:5px 8px;color:#c8a96e">—</td>'
        f'<td style="padding:5px 8px;color:#c8a96e">—</td>'
        f'<td style="padding:5px 8px;color:#c8a96e">{_dollar(result["BaselineQty"].sum())}</td>'
        f'<td style="padding:5px 8px;color:#c8a96e">{_dollar(result["NewQty"].sum())}</td>'
        f'<td style="padding:5px 8px">{_dbadge(result["GrowthValue"].sum())}</td></tr></table>')
    return h

def render_port_table(port_df,reg_df,total):
    def _row(name,old,new,chg,is_reg=False):
        cc=_cc(chg); s=_sign(chg); bg="#1e2740" if is_reg else ("#1a1d2e" if hash(name)%2 else "#13161f"); fw="bold" if is_reg else "normal"
        return (f'<tr style="background:{bg};font-weight:{fw};text-align:right">'
                f'<td style="text-align:left;padding:5px 8px;color:#f0f4f8">{name}</td>'
                f'<td style="padding:5px 8px;color:#f0f4f8">{old*100:.2f}%</td>'
                f'<td style="padding:5px 8px;color:#f0f4f8">{new*100:.2f}%</td>'
                f'<td style="padding:5px 8px;color:{cc}">{s}{chg*100:.2f}pp</td>'
                f'<td style="padding:5px 8px;color:#f0f4f8">{_dollar(old*total)}</td>'
                f'<td style="padding:5px 8px;color:#f0f4f8">{_dollar(new*total)}</td>'
                f'<td style="padding:5px 8px;color:{cc}">{s}{_dollar(chg*total)}</td></tr>')
    h=f'<table style="{TS}"><tr style="background:{THEAD};color:white;text-align:center">'
    for c in ["Port / Region","Before","After","Change (pp)","Before ($)","After ($)","Volume Δ"]:
        h+=f'<th style="padding:6px 8px">{c}</th>'
    h+='</tr>'
    pdf2=port_df.copy(); pdf2["Region"]=pdf2["Port of Entry"].apply(assign_region)
    for region in ["West Coast","Northern Border","Mexico Border","Northern Atlantic Coast","Southern Coast"]:
        rr=reg_df[reg_df["Region"]==region]
        if rr.empty: continue
        r=rr.iloc[0]; h+=_row(f"▶ {region}",r["OldShare"],r["NewShare"],r["Change"],True)
        for _,pr in pdf2[pdf2["Region"]==region].sort_values("OldShare",ascending=False).iterrows():
            h+=_row(f"&nbsp;&nbsp;&nbsp;{pr['Port of Entry']}",pr["OldShare"],pr["NewShare"],pr["Change"])
    other=pdf2[pdf2["Region"]=="Other"].sort_values("OldShare",ascending=False)
    if not other.empty:
        h+=_row("▶ Other / Interior",other["OldShare"].sum(),other["NewShare"].sum(),other["Change"].sum(),True)
        for _,pr in other.iterrows():
            h+=_row(f"&nbsp;&nbsp;&nbsp;{pr['Port of Entry']}",pr["OldShare"],pr["NewShare"],pr["Change"])
    h+="</table>"; return h

def build_map(port_df,label,total,mode="individual"):
    m=folium.Map(location=[38.5,-95.0],zoom_start=4,tiles="CartoDB positron",prefer_canvas=True)
    if mode=="individual":
        df=port_df.copy()
        df["lat"]=df["Port of Entry"].map(lambda p: PORT_COORDS.get(p,(None,None))[0])
        df["lon"]=df["Port of Entry"].map(lambda p: PORT_COORDS.get(p,(None,None))[1])
        df=df.dropna(subset=["lat","lon"]); ms=df["OldShare"].max() or 1
        for _,r in df.iterrows():
            chg=r["Change"]; fc="#1a7d3a" if chg>0.0005 else ("#c0392b" if chg<-0.0005 else "#7f8c8d")
            bc="#0f5a29" if chg>0.0005 else ("#8e1a1a" if chg<-0.0005 else "#5a6467")
            ro=max(7,min(50,(r["OldShare"]/ms)**0.5*50)); rn=max(4,min(50,(r["NewShare"]/ms)**0.5*50))
            pop=folium.Popup(f'<b>{r["Port of Entry"]}</b><br>Before:{r["OldShare"]*100:.2f}% → After:{r["NewShare"]*100:.2f}%<br><b style="color:{_cc(chg)}">{_sign(chg)}{chg*100:.2f}pp</b> | {_sign(chg)}{_dollar(chg*total)}',max_width=260)
            folium.CircleMarker([r["lat"],r["lon"]],radius=ro,color=bc,fill=True,fill_color=fc,fill_opacity=0.2,weight=2,popup=pop,tooltip=folium.Tooltip(f"{r['Port of Entry']}: {_sign(chg)}{chg*100:.2f}pp",sticky=True)).add_to(m)
            folium.CircleMarker([r["lat"],r["lon"]],radius=rn,color=bc,fill=True,fill_color=fc,fill_opacity=0.85,weight=1.5).add_to(m)
    else:
        rd=agg_regions(port_df); ms=rd["OldShare"].max() or 1
        for _,r in rd.iterrows():
            reg=r["Region"]; chg=r["Change"]; col=REGION_COLORS.get(reg,"#555")
            fc="#1a7d3a" if chg>0.0005 else ("#c0392b" if chg<-0.0005 else "#7f8c8d")
            bc="#0f5a29" if chg>0.0005 else ("#8e1a1a" if chg<-0.0005 else "#5a6467")
            lat,lon=REGION_CENTROIDS.get(reg,(38,-95))
            ro=max(15,min(70,(r["OldShare"]/ms)**0.5*70)); rn=max(8,min(70,(r["NewShare"]/ms)**0.5*70))
            pi=[p for p in REGION_MAP.get(reg,[]) if p in port_df["Port of Entry"].values]
            pop=folium.Popup(f'<b style="color:{col}">{reg}</b><br>Before:{r["OldShare"]*100:.2f}% → After:{r["NewShare"]*100:.2f}%<br><b style="color:{_cc(chg)}">{_sign(chg)}{chg*100:.2f}pp</b> | {_sign(chg)}{_dollar(chg*total)}<br><small>{"  •  ".join(pi)}</small>',max_width=280)
            folium.CircleMarker([lat,lon],radius=ro,color=bc,fill=True,fill_color=fc,fill_opacity=0.2,weight=2.5,popup=pop,tooltip=folium.Tooltip(f"{reg}: {_sign(chg)}{chg*100:.2f}pp",sticky=True)).add_to(m)
            folium.CircleMarker([lat,lon],radius=rn,color=bc,fill=True,fill_color=fc,fill_opacity=0.85,weight=2).add_to(m)
            folium.Marker([lat+1.8,lon],icon=folium.DivIcon(html=f'<div style="font-family:Arial;font-size:11px;font-weight:bold;color:{col};white-space:nowrap">{reg}</div>',icon_size=(180,20),icon_anchor=(90,10))).add_to(m)
    leg=(f'<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:#1a1d2e;border:1px solid #3a4060;'
         f'padding:14px 18px;border-radius:8px;font-family:Arial;font-size:12px;color:#f0f4f8">'
         f'<b style="color:#c8a96e">{label}</b><br><br>'
         f'<span style="color:#1a7d3a">●</span> Gaining &nbsp;<span style="color:#c0392b">●</span> Losing &nbsp;<span style="color:#7f8c8d">●</span> Flat<br>'
         f'<span style="font-size:11px;color:#a0b0c0">Outer ring = before &nbsp; Filled = after</span></div>')
    m.get_root().html.add_child(folium.Element(leg)); return m

# -----------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------
for k in ["results","port_df","reg_df","result_label","result_total","result_sigma"]:
    if k not in st.session_state: st.session_state[k]=None

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
try:
    df_pc,df_h4cb,df_h2cb,df_h4p,df_h2p,df_elas=load_all_data()
except Exception as e:
    st.error(f"Failed to load data: {e}"); st.stop()
try:
    clustered_sigma=compute_clustered_sigma(df_elas)
except Exception as e:
    clustered_sigma={}; st.warning(f"Clustering failed, using presets: {e}")

# -----------------------------------------------------------
# UI — HEADER
# -----------------------------------------------------------
st.title("🏭 Armington Trade Model")
st.markdown("*Constant Elasticity of Substitution — tariff impact on country & port shares*")
st.markdown("**Baseline shares:** time-weighted 2020–2025 (weights 1–6), normalized to top-10 = 100%")
st.divider()

# -----------------------------------------------------------
# STEP 1
# -----------------------------------------------------------
st.subheader("Step 1 — Select Product, Level & Elasticity")
col1,col2=st.columns([2,1])

with col1:
    avail=sorted(df_pc["HS2"].dropna().unique().astype(int).tolist())
    hs2_opts={f"HS2 {h} — {HS2_LABELS.get(h,'')}":h for h in avail}
    hs2_lbl=st.selectbox("HS2 Category",options=list(hs2_opts.keys()))
    hs2=hs2_opts[hs2_lbl]
    level=st.radio("Level",["HS2 level (full category)","HS4 level (specific product)"],horizontal=True)
    hs4=None
    if "HS4" in level:
        sv=st.checkbox("Sort HS4 by 2025 Customs Value")
        sub=df_pc[df_pc["HS2"]==hs2].copy(); sub["HS4"]=pd.to_numeric(sub["HS4"],errors="coerce")
        v25=sub[sub["Year"]==2025].groupby("HS4")["Customs Value $ (Consumption)"].sum().rename("v25")
        meta=sub.drop_duplicates("HS4").set_index("HS4")[["HS4 Description"]].join(v25,how="left"); meta["v25"]=meta["v25"].fillna(0)
        codes=(meta.sort_values("v25",ascending=False).index.dropna().astype(int).tolist() if sv else sorted(meta.index.dropna().astype(int).tolist()))
        desc=meta["HS4 Description"].to_dict(); val=meta["v25"].to_dict()
        def _fmt(c):
            v=val.get(c,0); vs=f"  ${v/1e6:,.0f}M" if v>=1e6 else (f"  ${v/1e3:,.0f}K" if v>0 else "")
            return f"HS4 {c} — {str(desc.get(c,''))[:45]}{vs}"
        hs4_opts={_fmt(c):c for c in codes}
        hs4=hs4_opts[st.selectbox("HS4 Product",options=list(hs4_opts.keys()))]

with col2:
    sm=st.radio("Elasticity (σ) Mode",["Custom — set σ manually","Claude AI — expert-scored σ","Data-driven — K-means"])
    if "Custom" in sm:
        sigma=st.number_input("σ value",min_value=1.0,max_value=10.0,value=4.0,step=0.1)
        st.caption("Enter any σ between 1.0 and 10.0")
    elif "Claude" in sm:
        sigma=float(PRESET_SIGMA.get(hs2,4.0))
        st.markdown(f"""<div style="background:#1e2235;border:1px solid #3a4060;border-radius:6px;padding:8px 14px;margin:4px 0 2px;">
          <span style="color:#c0cad6;font-size:12px;">σ value</span><br>
          <span style="color:#f0f4f8;font-size:22px;font-weight:700;">{sigma}</span>
        </div>""", unsafe_allow_html=True)
        st.caption(f"Claude AI scoring for HS2 {hs2}: σ = {sigma} — expert-calibrated")
    else:
        sigma=float(clustered_sigma.get(hs2,PRESET_SIGMA.get(hs2,4.0)))
        st.markdown(f"""<div style="background:#1e2235;border:1px solid #3a4060;border-radius:6px;padding:8px 14px;margin:4px 0 2px;">
          <span style="color:#c0cad6;font-size:12px;">σ value</span><br>
          <span style="color:#f0f4f8;font-size:22px;font-weight:700;">{sigma}</span>
        </div>""", unsafe_allow_html=True)
        st.caption(f"K-means cluster for HS2 {hs2}: σ = {sigma} — from trade data")

st.divider()

# -----------------------------------------------------------
# STEP 2
# -----------------------------------------------------------
st.subheader("Step 2 — Review Baseline & Enter New Tariff Rates")
try:
    baseline=get_baseline(df_h4cb,df_h2cb,hs2,hs4)
    total=get_total(df_pc,hs2,hs4)
    label=(f"HS4 {hs4} — {df_pc[df_pc['HS4']==hs4]['HS4 Description'].iloc[0][:50]}" if hs4 else f"HS2 {hs2} — {HS2_LABELS.get(hs2,'')}")
except Exception as e:
    st.error(f"Error loading baseline: {e}"); st.stop()

st.markdown(render_baseline(baseline,label,sigma,total),unsafe_allow_html=True)
st.markdown("**Enter new tariff rates (%) — pre-filled with 2025 effective rates:**")

new_tariffs={}
cols=st.columns(2)
for i,(_,r) in enumerate(baseline.iterrows()):
    country=r["Country of Origin"]; ot=round(float(r["OldTariff"])*100,1)
    with cols[i%2]:
        new_tariffs[country]=st.number_input(
            f"{int(r['Rank'])}. {country}  (Share: {_pct(r['BaselineShare'])} | Old: {ot}%)",
            min_value=0.0,max_value=500.0,value=ot,step=1.0,key=f"t_{country}_{hs2}_{hs4}")

st.divider()

# -----------------------------------------------------------
# STEP 3 — RUN
# -----------------------------------------------------------
if st.button("🚀 Run Armington Model",type="primary",use_container_width=True):
    with st.spinner("Running model..."):
        try:
            result=run_model(baseline,{c:v/100 for c,v in new_tariffs.items()},sigma,total)
            port_df=get_ports(df_h4p,df_h2p,result,hs2,hs4)
            reg_df=agg_regions(port_df) if not port_df.empty else pd.DataFrame()
            st.session_state.results=result; st.session_state.port_df=port_df
            st.session_state.reg_df=reg_df; st.session_state.result_label=label
            st.session_state.result_total=total; st.session_state.result_sigma=sigma
        except Exception as e:
            st.error(f"Model error: {e}")

# -----------------------------------------------------------
# RESULTS
# -----------------------------------------------------------
if st.session_state.results is not None:
    result=st.session_state.results; port_df=st.session_state.port_df
    reg_df=st.session_state.reg_df; label=st.session_state.result_label
    total=st.session_state.result_total; sigma=st.session_state.result_sigma

    st.subheader("📊 Armington Results")
    st.markdown(render_results(result,label,sigma,total),unsafe_allow_html=True)
    st.divider()

    if port_df is not None and not port_df.empty:
        st.subheader("🚢 Port Impact")
        mm=st.radio("Map View",["Individual Ports","Regional View"],horizontal=True,key="map_toggle")
        mk="individual" if "Individual" in mm else "regional"
        st_folium(build_map(port_df,label,total,mk),width=1200,height=550,returned_objects=[])
        st.markdown("**Port Impact Summary**")
        st.markdown(render_port_table(port_df,reg_df,total),unsafe_allow_html=True)
    else:
        st.info("No port data found for this selection.")

    # -----------------------------------------------------------
    # CSV DOWNLOADS
    # -----------------------------------------------------------
    st.divider()
    st.subheader("📥 Download Results")

    def _results_csv(df):
        out=df[["Rank","Country of Origin","BaselineShare","OldTariff","NewTariff","TariffChange","NewShare","ShareChangePct","BaselineQty","NewQty","GrowthValue"]].copy()
        out.columns=["Rank","Country","Baseline Share","Old Tariff","New Tariff","Tariff Change","New Share","Share Change","Baseline Qty ($)","New Qty ($)","Growth ($)"]
        for c in ["Baseline Share","Old Tariff","New Tariff","Tariff Change","New Share","Share Change"]:
            out[c]=out[c].map(lambda v: f"{v*100:.3f}%")
        return out.to_csv(index=False).encode("utf-8")

    def _port_csv(pdf,total):
        out=pdf.copy(); out["Region"]=out["Port of Entry"].apply(assign_region)
        out["Before %"]=out["OldShare"].map(lambda v:f"{v*100:.3f}%")
        out["After %"]=out["NewShare"].map(lambda v:f"{v*100:.3f}%")
        out["Change pp"]=out["Change"].map(lambda v:f"{v*100:.3f}pp")
        out["Before ($)"]=out["OldShare"]*total; out["After ($)"]=out["NewShare"]*total
        out["Volume Delta ($)"]=out["Change"]*total
        return out[["Port of Entry","Region","Before %","After %","Change pp","Before ($)","After ($)","Volume Delta ($)"]].to_csv(index=False).encode("utf-8")

    dl1,dl2,dl3=st.columns(3)
    fname=label[:30].replace(" ","_").replace("/","_")

    with dl1:
        st.download_button("⬇️ Country Results (CSV)",data=_results_csv(result),file_name=f"country_results_{fname}.csv",mime="text/csv",use_container_width=True)

    if port_df is not None and not port_df.empty:
        with dl2:
            st.download_button("⬇️ Port Impact (CSV)",data=_port_csv(port_df,total),file_name=f"port_impact_{fname}.csv",mime="text/csv",use_container_width=True)

    with dl3:
        zb=io.BytesIO()
        with zipfile.ZipFile(zb,"w") as zf:
            zf.writestr("country_results.csv",_results_csv(result).decode("utf-8"))
            if port_df is not None and not port_df.empty:
                zf.writestr("port_impact.csv",_port_csv(port_df,total).decode("utf-8"))
        st.download_button("⬇️ Download All (ZIP)",data=zb.getvalue(),file_name=f"armington_full_{fname}.zip",mime="application/zip",use_container_width=True)
