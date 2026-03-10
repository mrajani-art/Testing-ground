"""
METa QuotR — Intelligent Estimating Platform
Calyx Containers | Streamlit entry point
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ── Page config (must be first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="METa QuotR | Calyx Containers",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from config.settings import (
    APP_NAME, APP_SUBTITLE, COMPANY_NAME, REP_NAMES,
    SUBSTRATE_OPTIONS, FINISH_OPTIONS, SEAL_TYPE_OPTIONS,
    GUSSET_TYPE_OPTIONS, ZIPPER_OPTIONS, TEAR_NOTCH_OPTIONS,
    HOLE_PUNCH_OPTIONS, CORNER_TREATMENT_OPTIONS, EMBELLISHMENT_OPTIONS,
    FILL_STYLE_OPTIONS, PRINT_METHOD_OPTIONS,
    QUANTITY_TIERS, TARIFF_SCENARIOS, DEFAULT_TARIFF_RATE,
    MIN_WIDTH_IN, MIN_HEIGHT_IN, MIN_GUSSET_IN,
    MAX_WIDTH_IN, MAX_HEIGHT_IN, MAX_GUSSET_IN,
)
from src.utils.vendor_routing import route_vendor, calculate_web_width
from src.data.supabase_client import get_supabase_client


# ═══════════════════════════════════════════════════════════════════
# VENDOR COLOR MAP  (used in multiple render functions)
# ═══════════════════════════════════════════════════════════════════

VENDOR_COLORS = {
    "ross":     ("#93c5fd", "#0f2744"),
    "internal": ("#c4b5fd", "#1a0d44"),
    "dazpak":   ("#fbbf24", "#2d1500"),
    "tedpack":  ("#86efac", "#0a2016"),
}


# ═══════════════════════════════════════════════════════════════════
# MODEL LOADER — cached so it only loads once per Streamlit session
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading prediction models...")
def _load_predictor():
    """
    Load QuotePredictor (Dazpak/Ross/Internal) and TedpackPredictor.
    Returns (predictor, tedpack, error_list).
    Fails gracefully — errors are shown as warnings, not crashes.
    """
    predictor = None
    tedpack   = None
    errors    = []

    try:
        from src.ml.prediction import QuotePredictor
        predictor = QuotePredictor()
        predictor.load_models()
    except Exception as e:
        errors.append(f"QuotePredictor: {e}")

    try:
        from src.ml.tedpack_predictor_v2 import TedpackPredictor
        tedpack = TedpackPredictor()
        tedpack.load()
    except Exception as e:
        errors.append(f"TedpackPredictor: {e}")

    return predictor, tedpack, errors


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "rep_name":        "",
        "company_name":    "",
        "width":           5.0,
        "height":          8.0,
        "gusset":          3.0,
        "substrate":       "MET PET",
        "finish":          "Matte",
        "seal_type":       "Stand Up Pouch",
        "gusset_type":     "K-Seal",
        "fill_style":      "Top Fill",
        "zipper":          "No Zipper",
        "tear_notch":      "None",
        "hole_punch":      "None",
        "corner_treatment":"Straight",
        "embellishment":   "None",
        "print_method":    "Digital",
        "quantities":      [5_000, 10_000, 25_000],
        "tariff_rate":     DEFAULT_TARIFF_RATE,
        "vendor_override": None,
        "last_result":     None,
        "last_logged_id":  None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def calc_landed(fob: float, qty: int, tariff_rate: float,
                container_rate: float = 3_000.0) -> float:
    """FOB → DDP using standard freight + tariff model."""
    freight   = max(container_rate / max(qty, 1), 0.08)
    tariff    = fob * tariff_rate
    customs   = 485.0 / max(qty, 1)          # MPF flat
    insurance = (fob + freight) * 0.004
    return round(fob + freight + tariff + customs + insurance, 5)


def log_quote_session(specs: dict, result: dict, rep: str, company: str) -> str | None:
    """
    Insert one row into quote_sessions in Supabase.
    Returns the session UUID on success, None on failure.
    Always fails silently — never blocks the UI.
    """
    try:
        supabase     = get_supabase_client()
        predictions  = result.get("predictions", [])
        base         = predictions[0] if predictions else {}

        payload = {
            "rep_name":         rep or None,
            "company_name":     company or None,
            "vendor":           result.get("vendor"),
            "print_method":     result.get("print_method"),
            "width":            specs.get("width"),
            "height":           specs.get("height"),
            "gusset":           specs.get("gusset"),
            "substrate":        specs.get("substrate"),
            "finish":           specs.get("finish"),
            "seal_type":        specs.get("seal_type"),
            "gusset_type":      specs.get("gusset_type"),
            "zipper":           specs.get("zipper"),
            "tear_notch":       specs.get("tear_notch"),
            "hole_punch":       specs.get("hole_punch"),
            "corner_treatment": specs.get("corner_treatment"),
            "embellishment":    specs.get("embellishment"),
            "quantity_tiers":   specs.get("quantities"),
            "tariff_rate":      specs.get("tariff_rate"),
            "base_unit_price":  base.get("unit_price"),
            "base_total_price": base.get("total_price"),
            "routing_reason":   result.get("routing_reason"),
            "model_mape":       result.get("model_metrics", {}).get("mape"),
            "is_deterministic": result.get("is_deterministic", False),
            "all_predictions":  predictions,
            "cost_factors":     result.get("cost_factors", {}),
        }

        resp = supabase.table("quote_sessions").insert(payload).execute()
        if resp.data:
            return resp.data[0].get("id")
    except Exception:
        pass  # silent fail
    return None


# ═══════════════════════════════════════════════════════════════════
# RENDER — CENTER PRICING PANEL
# ═══════════════════════════════════════════════════════════════════

def render_pricing(result: dict, specs: dict):
    predictions  = result.get("predictions", [])
    vendor_name  = result.get("vendor", "").upper()
    vcolor, vbg  = VENDOR_COLORS.get(result.get("vendor", ""), ("#e8f2e8", "#1a241a"))
    tariff_rate  = specs.get("tariff_rate", DEFAULT_TARIFF_RATE)
    is_det       = result.get("is_deterministic", False)
    metrics      = result.get("model_metrics", {})

    # Warnings
    for w in result.get("warnings", []):
        st.warning(w)

    if not predictions:
        st.error("No predictions returned. Check that .joblib model files are in models/")
        return

    # ── Vendor banner ──────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:#080d08;border:1px solid #1a241a;border-radius:6px;
                padding:8px 14px;margin-bottom:12px;display:flex;
                align-items:center;justify-content:space-between;">
      <span style="background:{vbg};color:{vcolor};padding:3px 12px;border-radius:12px;
                   font-size:11px;font-weight:700;letter-spacing:0.12em;
                   font-family:'Barlow Condensed',sans-serif;">{vendor_name}</span>
      <span style="color:#4a5e4a;font-size:9px;font-style:italic;">
        {result.get('routing_reason','')}
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Primary price block ────────────────────────────────────────
    base      = predictions[0]
    fob       = base["unit_price"]
    qty       = base["quantity"]
    total_fob = base["total_price"]
    landed    = calc_landed(fob, qty, tariff_rate)
    total_lan = round(landed * qty, 2)
    lo        = base.get("lower_bound", fob * 0.90)
    hi        = base.get("upper_bound", fob * 1.10)
    tariff_pct = int(tariff_rate * 100)

    if is_det:
        method_note = f"Deterministic Calculator · MAPE {metrics.get('mape', 7.9):.1f}%"
    else:
        mape = metrics.get("mape", 0) * 100
        r2   = metrics.get("r2", 0)
        n    = metrics.get("n_samples", "?")
        method_note = f"ML Model · MAPE {mape:.1f}% · R² {r2:.3f} · n={n}"

    ci_html = "" if is_det else f"""
      <div style="margin-top:10px;">
        <div style="color:#4a5e4a;font-size:8px;font-family:'Barlow Condensed',sans-serif;
                    margin-bottom:4px;letter-spacing:0.05em;">80% CONFIDENCE INTERVAL</div>
        <div style="background:#1a241a;height:8px;border-radius:4px;">
          <div style="background:#4ade8018;height:100%;border-radius:4px;width:100%;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:3px;
                    font-family:'IBM Plex Mono',monospace;font-size:9px;color:#4a5e4a;">
          <span>${lo:.4f}</span><span>POINT ESTIMATE</span><span>${hi:.4f}</span>
        </div>
      </div>"""

    st.markdown(f"""
    <div style="background:#0e150e;border:1px solid #1a241a;border-radius:8px;
                padding:20px 24px;margin-bottom:12px;">
      <div style="color:#4a5e4a;font-size:9px;letter-spacing:0.1em;
                  font-family:'Barlow Condensed',sans-serif;margin-bottom:4px;">
        UNIT PRICE FOB
      </div>
      <div style="display:flex;align-items:baseline;gap:6px;margin-bottom:2px;">
        <span style="color:#8aa88a;font-size:32px;font-family:'IBM Plex Mono',monospace;">$</span>
        <span style="color:#4ade80;font-size:56px;font-weight:700;
                     font-family:'IBM Plex Mono',monospace;line-height:1;">{fob:.4f}</span>
        <span style="color:#4a5e4a;font-size:12px;font-weight:600;align-self:flex-end;
                     font-family:'Barlow Condensed',sans-serif;">/unit</span>
      </div>
      <div style="color:#e8f2e8;font-size:16px;font-family:'IBM Plex Mono',monospace;
                  margin-bottom:12px;">
        ${total_fob:,.2f} TOTAL FOB AT {qty:,} UNITS
      </div>
      <hr style="border:none;border-top:1px solid #1a241a;margin:12px 0;">
      <div style="display:flex;align-items:baseline;gap:10px;margin-bottom:4px;">
        <span style="color:#4a5e4a;font-size:9px;font-family:'Barlow Condensed',sans-serif;
                     letter-spacing:0.1em;">LANDED COST</span>
        <span style="color:#c8d8c8;font-size:28px;font-weight:600;
                     font-family:'IBM Plex Mono',monospace;">${landed:.4f}</span>
        <span style="background:#2d1500;color:#fbbf24;padding:2px 8px;border-radius:10px;
                     font-size:8px;font-family:'Barlow Condensed',sans-serif;">
          {tariff_pct}% TARIFF
        </span>
      </div>
      <div style="color:#8aa88a;font-size:14px;font-family:'IBM Plex Mono',monospace;
                  margin-bottom:4px;">
        ${total_lan:,.2f} TOTAL LANDED
      </div>
      {ci_html}
      <div style="color:#4a5e4a;font-size:8px;font-family:'IBM Plex Mono',monospace;
                  margin-top:10px;">{method_note}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tier table ─────────────────────────────────────────────────
    st.markdown(
        "<div style='color:#4a5e4a;font-size:9px;letter-spacing:0.12em;margin-bottom:6px;"
        "font-family:Barlow Condensed,sans-serif;'>QUANTITY TIER TABLE</div>",
        unsafe_allow_html=True
    )

    base_landed = calc_landed(predictions[0]["unit_price"],
                              predictions[0]["quantity"], tariff_rate)
    rows = []
    for p in predictions:
        fob_u   = p["unit_price"]
        q       = p["quantity"]
        lan_u   = calc_landed(fob_u, q, tariff_rate)
        total_l = round(lan_u * q, 2)
        delta   = round(lan_u - base_landed, 5)
        if delta > 0:
            vs = f"▲ +${delta:.4f}"
        elif delta < 0:
            vs = f"▼ −${abs(delta):.4f}"
        else:
            vs = "— baseline"
        rows.append({
            "QTY":           f"{q:,}",
            "FOB/unit":      f"${fob_u:.4f}",
            "LANDED/unit":   f"${lan_u:.4f}",
            "TOTAL LANDED":  f"${total_l:,.0f}",
            "vs base":       vs,
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    # ── Internal: component breakdown ─────────────────────────────
    if is_det and result.get("component_costs"):
        with st.expander("📋 HP 6900 Component Cost Breakdown"):
            comp = result["component_costs"][0]
            comp_rows = [
                {"Component": k.replace("_", " ").title(), "Cost": f"${v:,.4f}"}
                for k, v in comp.items()
                if k not in ("quantity", "total") and isinstance(v, (int, float)) and v > 0
            ]
            comp_rows.append({"Component": "TOTAL",
                               "Cost": f"${comp.get('total', 0):,.4f}"})
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

            if result.get("layout"):
                lay = result["layout"]
                st.caption(
                    f"Layout: {lay.get('no_across')}×{lay.get('no_around')} "
                    f"({lay.get('labels_per_cycle')} up) · "
                    f"Gear {lay.get('gear_teeth')} · "
                    f"Repeat {lay.get('repeat_in'):.3f}\" · "
                    f"Stock {lay.get('stock_length_ft'):,.0f} ft"
                )


# ═══════════════════════════════════════════════════════════════════
# RENDER — RIGHT INTELLIGENCE PANEL
# ═══════════════════════════════════════════════════════════════════

def render_intelligence(result: dict, specs: dict):
    cost_factors = result.get("cost_factors", {})
    is_det       = result.get("is_deterministic", False)
    metrics      = result.get("model_metrics", {})
    vendor_name  = result.get("vendor", "")
    vcolor, vbg  = VENDOR_COLORS.get(vendor_name, ("#e8f2e8", "#1a241a"))

    # ── A: Cost drivers ────────────────────────────────────────────
    st.markdown(
        "<div style='color:#4a5e4a;font-size:9px;letter-spacing:0.12em;margin-bottom:8px;"
        "font-family:Barlow Condensed,sans-serif;'>A · COST DRIVERS</div>",
        unsafe_allow_html=True
    )

    if is_det and result.get("component_costs"):
        # Deterministic HP6900: show component percentages
        comp  = result["component_costs"][0]
        total = comp.get("total", 1) or 1
        COMP_LABELS = {
            "substrate":     ("Substrate Film",   "#4ade80"),
            "clicks":        ("HP Click Charges", "#3b82f6"),
            "hp_running":    ("HP Run Time",       "#a78bfa"),
            "laminate":      ("Laminate Film",     "#d97706"),
            "poucher_labor": ("Poucher Labor",     "#f59e0b"),
            "zipper":        ("Zipper Stock",      "#ef4444"),
            "packaging":     ("Packaging",         "#8aa88a"),
            "priming":       ("HP Priming",        "#6ee7b7"),
            "hp_makeready":  ("HP Makeready",      "#c4b5fd"),
            "sealer":        ("Sealer Ink",        "#fcd34d"),
            "thermo_labor":  ("Thermo Labor",      "#fb923c"),
        }
        for key, (label, color) in COMP_LABELS.items():
            val = comp.get(key, 0)
            if val > 0:
                pct = min(round(val / total * 100, 1), 100)
                st.markdown(f"""
                <div style="margin-bottom:9px;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                    <span style="color:#e8f2e8;font-size:9px;
                                 font-family:'Barlow Condensed',sans-serif;">{label}</span>
                    <span style="color:{color};font-size:11px;
                                 font-family:'IBM Plex Mono',monospace;">{pct}%</span>
                  </div>
                  <div style="background:#1a241a;height:6px;border-radius:3px;">
                    <div style="background:{color};opacity:0.75;width:{pct}%;
                                height:100%;border-radius:3px;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    elif cost_factors:
        # ML models: SHAP-style grouped importances
        VOL_FEATS  = {"log_quantity","quantity","inv_quantity","skus_in_quote",
                      "is_gravure","ross_setup_per_unit","area_x_logqty"}
        SIZE_FEATS = {"bag_area_sqin","print_width","width","height","gusset",
                      "print_area_msi","estimated_weight_g","has_gusset"}
        MAT_FEATS  = {"substrate","finish","zipper","zipper_width","zipper_score",
                      "has_zipper","ross_converting_cost"}

        GROUP_COLORS = {
            "ORDER VOLUME": ("#3b82f6", "#1e3a5f"),
            "BAG SIZE":     ("#4ade80", "#1a3a1a"),
            "MATERIAL":     ("#d97706", "#2d2200"),
            "CONFIG":       ("#a78bfa", "#1f1040"),
        }

        groups: dict[str, float] = {}
        for feat, info in cost_factors.items():
            if feat in VOL_FEATS:   g = "ORDER VOLUME"
            elif feat in SIZE_FEATS: g = "BAG SIZE"
            elif feat in MAT_FEATS:  g = "MATERIAL"
            else:                    g = "CONFIG"
            groups[g] = groups.get(g, 0) + info.get("importance", 0)

        total_imp = sum(groups.values()) or 1
        for gname, imp in sorted(groups.items(), key=lambda x: -x[1]):
            pct = round(imp / total_imp * 100, 1)
            accent, fill = GROUP_COLORS[gname]
            st.markdown(f"""
            <div style="margin-bottom:12px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span style="color:#e8f2e8;font-size:9px;font-weight:700;
                             font-family:'Barlow Condensed',sans-serif;">{gname}</span>
                <span style="color:{accent};font-size:12px;
                             font-family:'IBM Plex Mono',monospace;">{pct:.0f}%</span>
              </div>
              <div style="background:#1a241a;height:8px;border-radius:4px;">
                <div style="background:{fill};border-right:3px solid {accent};
                            width:{pct}%;height:100%;border-radius:4px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Feature detail"):
            for feat, info in sorted(cost_factors.items(),
                                     key=lambda x: -x[1].get("importance", 0))[:10]:
                st.markdown(
                    f"<span style='color:#8aa88a;font-size:9px;"
                    f"font-family:IBM Plex Mono,monospace;'>"
                    f"{feat}: <b style='color:#e8f2e8;'>"
                    f"{info.get('importance',0):.1f}%</b> · "
                    f"{info.get('value','—')}</span>",
                    unsafe_allow_html=True
                )
    else:
        st.markdown(
            "<div style='color:#4a5e4a;font-size:10px;'>No cost factor data.</div>",
            unsafe_allow_html=True
        )

    st.divider()

    # ── B: Model metrics ───────────────────────────────────────────
    st.markdown(
        "<div style='color:#4a5e4a;font-size:9px;letter-spacing:0.12em;margin-bottom:6px;"
        "font-family:Barlow Condensed,sans-serif;'>B · MODEL METRICS</div>",
        unsafe_allow_html=True
    )

    if is_det:
        mape = metrics.get("mape", 7.9)
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:10px;color:#8aa88a;'>"
            f"{metrics.get('method','Deterministic Calculator')}<br>"
            f"<span style='color:#4ade80;'>MAPE: {mape:.1f}%</span> · "
            f"n={metrics.get('training_rows', 285)}</div>",
            unsafe_allow_html=True
        )
    elif metrics:
        mape  = metrics.get("mape", 0) * 100
        r2    = metrics.get("r2", 0)
        n     = metrics.get("n_samples", "?")
        color = "#4ade80" if mape < 12 else ("#d97706" if mape < 20 else "#ef4444")
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:11px;'>"
            f"<span style='color:{color};'>MAPE {mape:.1f}%</span>  ·  "
            f"<span style='color:#4ade80;'>R² {r2:.3f}</span>  ·  "
            f"<span style='color:#4a5e4a;'>n={n}</span></div>",
            unsafe_allow_html=True
        )
    else:
        st.caption("No metrics available — train models first.")

    st.divider()

    # ── Routing summary ────────────────────────────────────────────
    st.markdown(
        "<div style='color:#4a5e4a;font-size:9px;letter-spacing:0.12em;margin-bottom:6px;"
        "font-family:Barlow Condensed,sans-serif;'>ROUTING</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<span style='background:{vbg};color:{vcolor};padding:2px 10px;"
        f"border-radius:10px;font-family:Barlow Condensed,sans-serif;"
        f"font-size:11px;font-weight:700;'>{vendor_name.upper()}</span>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='color:#4a5e4a;font-size:9px;font-style:italic;margin-top:6px;'>"
        f"{result.get('routing_reason','')}</div>",
        unsafe_allow_html=True
    )

    # ── Session log badge ──────────────────────────────────────────
    if st.session_state.get("last_logged_id"):
        st.divider()
        sid = st.session_state["last_logged_id"]
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:8px;color:#1a4a2a;'>"
            f"● SESSION LOGGED · SUPABASE<br>{sid[:20]}...</div>",
            unsafe_allow_html=True
        )


# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════

predictor, tedpack_predictor, load_errors = _load_predictor()

models_ok     = bool(predictor or tedpack_predictor)
models_status = "● MODELS LIVE" if models_ok else "○ MODELS OFFLINE"
models_color  = "#4ade80"       if models_ok else "#f59e0b"
now_str       = datetime.now().strftime("%d %b %Y · %H:%M")

st.markdown(f"""
<div style="background:#0e150e;padding:10px 24px;border-bottom:1px solid #1a241a;
            display:flex;align-items:center;justify-content:space-between;margin-bottom:0;">
  <div style="display:flex;align-items:center;gap:20px;">
    <span style="color:#e8f2e8;font-size:18px;font-weight:700;letter-spacing:0.08em;
                 font-family:'Barlow Condensed',sans-serif;">METa QuotR</span>
    <span style="color:#4ade80;font-size:9px;letter-spacing:0.2em;
                 font-family:'Barlow Condensed',sans-serif;">INTELLIGENT ESTIMATING PLATFORM</span>
    <span style="color:#4a5e4a;font-size:9px;">CALYX CONTAINERS</span>
  </div>
  <div style="display:flex;align-items:center;gap:16px;font-size:9px;color:#4a5e4a;
              font-family:'IBM Plex Mono',monospace;">
    <span style="color:{models_color};">{models_status}</span>
    <span>|</span>
    <span>{now_str}</span>
  </div>
</div>
""", unsafe_allow_html=True)

if load_errors:
    with st.expander("⚠ Model load warnings (non-blocking)", expanded=False):
        for e in load_errors:
            st.warning(e)
        st.info(
            "To fix: run `python scripts/train_models.py` and commit "
            "the resulting .joblib files in models/ to GitHub. "
            "Tedpack models come from TEDPACK_ML_V2.ipynb."
        )


# ═══════════════════════════════════════════════════════════════════
# 3-COLUMN LAYOUT
# ═══════════════════════════════════════════════════════════════════

left, center, right = st.columns([2.6, 4.2, 3.2])


# ═══════════════════════════════════════════════════════════════════
# LEFT — SPEC BUILDER
# ═══════════════════════════════════════════════════════════════════

with left:
    st.markdown(
        "<p style='color:#4ade80;font-size:10px;letter-spacing:0.15em;margin:12px 0 4px;"
        "font-family:Barlow Condensed,sans-serif;'>SPEC BUILDER</p>",
        unsafe_allow_html=True
    )

    # Session context
    st.session_state["company_name"] = st.text_input(
        "Account", value=st.session_state["company_name"],
        placeholder="Company name..."
    )
    st.session_state["rep_name"] = st.selectbox(
        "Rep", options=[""] + REP_NAMES, index=0
    )
    st.divider()

    # Dimensions
    st.markdown("**Dimensions (inches)**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state["width"] = st.number_input(
            'W"', min_value=MIN_WIDTH_IN, max_value=MAX_WIDTH_IN,
            value=st.session_state["width"], step=0.25
        )
    with c2:
        st.session_state["height"] = st.number_input(
            'H"', min_value=MIN_HEIGHT_IN, max_value=MAX_HEIGHT_IN,
            value=st.session_state["height"], step=0.25
        )
    with c3:
        st.session_state["gusset"] = st.number_input(
            'G"', min_value=MIN_GUSSET_IN, max_value=MAX_GUSSET_IN,
            value=st.session_state["gusset"], step=0.25
        )

    # Live routing
    web_width = calculate_web_width(st.session_state["height"], st.session_state["gusset"])
    bag_area  = round(st.session_state["width"] * st.session_state["height"], 2)

    routing = route_vendor(
        print_method=st.session_state["print_method"],
        height=st.session_state["height"],
        gusset=st.session_state["gusset"],
        quantities=st.session_state["quantities"],
    )
    vendor = st.session_state.get("vendor_override") or routing["vendor"]
    vcolor, vbg = VENDOR_COLORS.get(vendor, ("#e8f2e8", "#1a241a"))

    st.markdown(f"""
    <div style="background:#0b110b;border:1px solid #1a241a;border-radius:6px;
                padding:10px 14px;margin:8px 0;">
      <div style="display:flex;gap:20px;margin-bottom:8px;">
        <span style="color:#8aa88a;font-size:9px;font-family:'IBM Plex Mono',monospace;">
          WEB WIDTH<br>
          <span style="color:#4ade80;font-size:14px;font-weight:600;">{web_width:.1f}"</span>
        </span>
        <span style="color:#8aa88a;font-size:9px;font-family:'IBM Plex Mono',monospace;">
          BAG AREA<br>
          <span style="color:#4ade80;font-size:14px;font-weight:600;">{bag_area} sq in</span>
        </span>
      </div>
      <span style="background:{vbg};color:{vcolor};padding:4px 12px;border-radius:12px;
                   font-size:11px;font-weight:700;letter-spacing:0.12em;
                   font-family:'Barlow Condensed',sans-serif;">{vendor.upper()}</span>
      <div style="color:#4a5e4a;font-size:9px;margin-top:5px;font-style:italic;">
        {routing['reason']}
      </div>
      {"".join(f'<div style="color:#fbbf24;font-size:9px;margin-top:3px;">⚠ {w}</div>'
               for w in routing.get('warnings', []))}
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Vendor override"):
        ov_opts = ["Auto (recommended)", "dazpak", "ross", "internal", "tedpack"]
        override = st.selectbox("Force vendor", ov_opts, index=0)
        st.session_state["vendor_override"] = None if override == "Auto (recommended)" else override

    st.divider()

    # Spec controls
    st.session_state["print_method"] = st.radio(
        "Print Method", options=PRINT_METHOD_OPTIONS,
        horizontal=True,
        index=PRINT_METHOD_OPTIONS.index(st.session_state["print_method"])
    )
    st.session_state["substrate"] = st.selectbox(
        "Substrate", SUBSTRATE_OPTIONS,
        index=SUBSTRATE_OPTIONS.index(st.session_state["substrate"])
    )
    st.session_state["finish"] = st.selectbox(
        "Finish", FINISH_OPTIONS,
        index=FINISH_OPTIONS.index(st.session_state["finish"])
    )

    ca, cb = st.columns(2)
    with ca:
        st.session_state["seal_type"]   = st.selectbox("Seal Type",   SEAL_TYPE_OPTIONS)
        st.session_state["gusset_type"] = st.selectbox("Gusset Type", GUSSET_TYPE_OPTIONS)
        st.session_state["fill_style"]  = st.selectbox("Fill",        FILL_STYLE_OPTIONS)
    with cb:
        st.session_state["zipper"]      = st.selectbox("Zipper",     ZIPPER_OPTIONS)
        st.session_state["tear_notch"]  = st.selectbox("Tear Notch", TEAR_NOTCH_OPTIONS)
        st.session_state["hole_punch"]  = st.selectbox("Hole Punch", HOLE_PUNCH_OPTIONS)

    cc, cd = st.columns(2)
    with cc:
        st.session_state["corner_treatment"] = st.selectbox("Corners",       CORNER_TREATMENT_OPTIONS)
    with cd:
        st.session_state["embellishment"]    = st.selectbox("Embellishment", EMBELLISHMENT_OPTIONS)

    st.divider()

    # Quantity tiers
    st.markdown("**Quantity Tiers**")
    default_tiers = QUANTITY_TIERS.get(vendor, [5_000, 10_000, 25_000, 50_000])
    selected_qtys = st.multiselect(
        "Select tiers", options=default_tiers,
        default=[q for q in st.session_state["quantities"] if q in default_tiers],
        format_func=lambda x: f"{x:,}"
    )
    if selected_qtys:
        st.session_state["quantities"] = sorted(selected_qtys)

    st.divider()

    # Tariff
    tariff_label = st.radio(
        "Tariff Assumption", options=list(TARIFF_SCENARIOS.keys()),
        horizontal=True,
        index=list(TARIFF_SCENARIOS.values()).index(st.session_state["tariff_rate"])
              if st.session_state["tariff_rate"] in TARIFF_SCENARIOS.values() else 1,
    )
    st.session_state["tariff_rate"] = TARIFF_SCENARIOS[tariff_label]

    st.divider()

    # Generate button
    generate = st.button("⚡ GENERATE QUOTE", use_container_width=True, type="primary")

    # Session log indicator
    if st.session_state.get("last_logged_id"):
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:8px;"
            f"color:#1a4a2a;margin-top:6px;'>"
            f"● SESSION LOGGED · {st.session_state['last_logged_id'][:12]}…</div>",
            unsafe_allow_html=True
        )


# ═══════════════════════════════════════════════════════════════════
# CENTER — PRICING
# ═══════════════════════════════════════════════════════════════════

with center:
    st.markdown(
        "<p style='color:#4ade80;font-size:10px;letter-spacing:0.15em;margin:12px 0 4px;"
        "font-family:Barlow Condensed,sans-serif;'>PRICE INTELLIGENCE</p>",
        unsafe_allow_html=True
    )

    # Build spec dict from session state
    specs = {
        "width":             st.session_state["width"],
        "height":            st.session_state["height"],
        "gusset":            st.session_state["gusset"],
        "substrate":         st.session_state["substrate"],
        "finish":            st.session_state["finish"],
        "seal_type":         st.session_state["seal_type"],
        "gusset_type":       st.session_state["gusset_type"],
        "fill_style":        st.session_state["fill_style"],
        "zipper":            st.session_state["zipper"],
        "tear_notch":        st.session_state["tear_notch"],
        "hole_punch":        st.session_state["hole_punch"],
        "corner_treatment":  st.session_state["corner_treatment"],
        "embellishment":     st.session_state["embellishment"],
        "print_method":      st.session_state["print_method"],
        "quantities":        st.session_state["quantities"],
        "tariff_rate":       st.session_state["tariff_rate"],
    }

    if generate:
        qtys = st.session_state["quantities"]
        if not qtys:
            st.warning("Select at least one quantity tier.")
        else:
            with st.spinner("Computing..."):
                result   = None
                err_msg  = None
                active_v = st.session_state.get("vendor_override") or routing["vendor"]

                try:
                    if active_v == "tedpack":
                        if tedpack_predictor:
                            result = tedpack_predictor.predict(specs, qtys)
                        else:
                            err_msg = (
                                "Tedpack model not loaded. "
                                "Run TEDPACK_ML_V2.ipynb and commit the 4 .joblib files to models/."
                            )
                    else:
                        if predictor:
                            result = predictor.predict(
                                specs, qtys,
                                vendor_override=st.session_state.get("vendor_override")
                            )
                        else:
                            err_msg = (
                                "Models not loaded. "
                                "Run `python scripts/train_models.py` and commit "
                                ".joblib files to models/."
                            )
                except Exception as e:
                    err_msg = f"Prediction error: {e}"

                if result:
                    st.session_state["last_result"] = result
                    # ── LOG SESSION TO SUPABASE ────────────────────
                    session_id = log_quote_session(
                        specs=specs, result=result,
                        rep=st.session_state["rep_name"],
                        company=st.session_state["company_name"],
                    )
                    if session_id:
                        st.session_state["last_logged_id"] = session_id
                elif err_msg:
                    st.error(err_msg)

    result = st.session_state.get("last_result")
    if result:
        render_pricing(result, specs)
    else:
        st.markdown(
            "<div style='color:#4a5e4a;text-align:center;margin-top:80px;font-size:13px;'>"
            "Configure spec in the left panel<br>and press GENERATE QUOTE"
            "</div>",
            unsafe_allow_html=True
        )


# ═══════════════════════════════════════════════════════════════════
# RIGHT — INTELLIGENCE PANEL
# ═══════════════════════════════════════════════════════════════════

with right:
    st.markdown(
        "<p style='color:#4ade80;font-size:10px;letter-spacing:0.15em;margin:12px 0 4px;"
        "font-family:Barlow Condensed,sans-serif;'>INTELLIGENCE PANEL</p>",
        unsafe_allow_html=True
    )

    result = st.session_state.get("last_result")
    if result:
        render_intelligence(result, specs)
    else:
        st.markdown(
            "<div style='color:#4a5e4a;font-size:11px;margin-top:16px;'>"
            "SHAP cost drivers and model metrics appear here after GENERATE QUOTE."
            "</div>",
            unsafe_allow_html=True
        )
        st.divider()
        st.markdown(
            f"<div style='font-size:10px;color:#8aa88a;font-family:IBM Plex Mono,monospace;'>"
            f"<b style='color:#e8f2e8;'>ROUTING</b><br>"
            f"<span style='background:{vbg};color:{vcolor};padding:2px 8px;"
            f"border-radius:10px;font-family:Barlow Condensed,sans-serif;"
            f"font-size:10px;'>{vendor.upper()}</span><br><br>"
            f"<span style='color:#4a5e4a;font-style:italic;font-size:9px;'>"
            f"{routing['reason']}</span></div>",
            unsafe_allow_html=True
        )
        for w in routing.get("warnings", []):
            st.warning(w)
