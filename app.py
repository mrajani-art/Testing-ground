"""
METa QuotR — Intelligent Estimating Platform
Calyx Containers | Streamlit entry point
"""
import streamlit as st
import pandas as pd

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


# ── Session state init ─────────────────────────────────────────────
def init_state():
    defaults = {
        "rep_name": "",
        "company_name": "",
        "width": 5.0,
        "height": 8.0,
        "gusset": 3.0,
        "substrate": "MET PET",
        "finish": "Matte",
        "seal_type": "Stand Up Pouch",
        "gusset_type": "K-Seal",
        "fill_style": "Top Fill",
        "zipper": "No Zipper",
        "tear_notch": "None",
        "hole_punch": "None",
        "corner_treatment": "Straight",
        "embellishment": "None",
        "print_method": "Digital",
        "quantities": [5_000, 10_000, 25_000],
        "tariff_rate": DEFAULT_TARIFF_RATE,
        "last_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Header ─────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:#0e150e; padding:12px 24px; border-bottom:1px solid #1a241a;
            display:flex; align-items:center; justify-content:space-between; margin-bottom:16px;">
  <div>
    <span style="color:#e8f2e8; font-size:20px; font-weight:700; letter-spacing:0.05em;">
      {APP_NAME}
    </span>
    <span style="color:#4a5e4a; font-size:11px; margin-left:12px;">
      {APP_SUBTITLE} · {COMPANY_NAME}
    </span>
  </div>
  <div style="color:#4ade80; font-size:10px;">● MODELS LIVE</div>
</div>
""", unsafe_allow_html=True)


# ── 3-column layout ────────────────────────────────────────────────
left, center, right = st.columns([2.6, 4.2, 3.2])


# ═══════════════════════════════════════════════════════════════════
# LEFT — SPEC BUILDER
# ═══════════════════════════════════════════════════════════════════
with left:
    st.markdown("#### 🏗 Spec Builder")

    # Session context
    with st.container():
        st.session_state["company_name"] = st.text_input(
            "Account", value=st.session_state["company_name"],
            placeholder="Company name..."
        )
        st.session_state["rep_name"] = st.selectbox(
            "Rep", options=[""] + REP_NAMES,
            index=0
        )

    st.divider()

    # Dimensions
    st.markdown("**Dimensions (inches)**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state["width"] = st.number_input(
            "Width", min_value=MIN_WIDTH_IN, max_value=MAX_WIDTH_IN,
            value=st.session_state["width"], step=0.25
        )
    with c2:
        st.session_state["height"] = st.number_input(
            "Height", min_value=MIN_HEIGHT_IN, max_value=MAX_HEIGHT_IN,
            value=st.session_state["height"], step=0.25
        )
    with c3:
        st.session_state["gusset"] = st.number_input(
            "Gusset", min_value=MIN_GUSSET_IN, max_value=MAX_GUSSET_IN,
            value=st.session_state["gusset"], step=0.25
        )

    # Live routing display
    web_width = calculate_web_width(st.session_state["height"], st.session_state["gusset"])
    bag_area = round(st.session_state["width"] * st.session_state["height"], 2)

    routing = route_vendor(
        print_method=st.session_state["print_method"],
        height=st.session_state["height"],
        gusset=st.session_state["gusset"],
        quantities=st.session_state["quantities"],
    )
    vendor = routing["vendor"]

    vendor_colors = {
        "ross":     ("#93c5fd", "#0f2744"),
        "internal": ("#c4b5fd", "#1a0d44"),
        "dazpak":   ("#fbbf24", "#2d1500"),
        "tedpack":  ("#86efac", "#0a2016"),
    }
    vcolor, vbg = vendor_colors.get(vendor, ("#e8f2e8", "#1a241a"))

    st.markdown(f"""
    <div style="background:#0b110b; border:1px solid #1a241a; border-radius:6px;
                padding:10px 14px; margin:8px 0;">
      <div style="display:flex; gap:16px; margin-bottom:6px;">
        <span style="color:#8aa88a; font-size:10px;">WEB WIDTH<br>
          <span style="color:#4ade80; font-size:14px; font-weight:600;">{web_width:.1f}"</span>
        </span>
        <span style="color:#8aa88a; font-size:10px;">BAG AREA<br>
          <span style="color:#4ade80; font-size:14px; font-weight:600;">{bag_area} sq in</span>
        </span>
      </div>
      <span style="background:{vbg}; color:{vcolor}; padding:3px 10px; border-radius:12px;
                   font-size:11px; font-weight:700; letter-spacing:0.1em;">
        {vendor.upper()}
      </span>
      <div style="color:#4a5e4a; font-size:9px; margin-top:4px;">{routing['reason']}</div>
      {"".join(f'<div style="color:#fbbf24; font-size:9px;">⚠ {w}</div>' for w in routing.get('warnings', []))}
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Print method
    st.session_state["print_method"] = st.radio(
        "Print Method", options=PRINT_METHOD_OPTIONS,
        horizontal=True, index=PRINT_METHOD_OPTIONS.index(st.session_state["print_method"])
    )

    # Substrate
    st.session_state["substrate"] = st.selectbox(
        "Substrate", options=SUBSTRATE_OPTIONS,
        index=SUBSTRATE_OPTIONS.index(st.session_state["substrate"])
    )

    # Finish
    st.session_state["finish"] = st.selectbox(
        "Finish", options=FINISH_OPTIONS,
        index=FINISH_OPTIONS.index(st.session_state["finish"])
    )

    # Seal & structure
    col_a, col_b = st.columns(2)
    with col_a:
        st.session_state["seal_type"] = st.selectbox("Seal Type", SEAL_TYPE_OPTIONS)
        st.session_state["gusset_type"] = st.selectbox("Gusset Type", GUSSET_TYPE_OPTIONS)
        st.session_state["fill_style"] = st.selectbox("Fill Style", FILL_STYLE_OPTIONS)
    with col_b:
        st.session_state["zipper"] = st.selectbox("Zipper", ZIPPER_OPTIONS)
        st.session_state["tear_notch"] = st.selectbox("Tear Notch", TEAR_NOTCH_OPTIONS)
        st.session_state["hole_punch"] = st.selectbox("Hole Punch", HOLE_PUNCH_OPTIONS)

    col_c, col_d = st.columns(2)
    with col_c:
        st.session_state["corner_treatment"] = st.selectbox("Corners", CORNER_TREATMENT_OPTIONS)
    with col_d:
        st.session_state["embellishment"] = st.selectbox("Embellishment", EMBELLISHMENT_OPTIONS)

    st.divider()

    # Quantity chips
    st.markdown("**Quantities**")
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
        "Tariff Assumption",
        options=list(TARIFF_SCENARIOS.keys()),
        horizontal=True,
        index=list(TARIFF_SCENARIOS.values()).index(st.session_state["tariff_rate"])
        if st.session_state["tariff_rate"] in TARIFF_SCENARIOS.values() else 1,
    )
    st.session_state["tariff_rate"] = TARIFF_SCENARIOS[tariff_label]

    st.divider()

    # Generate button
    generate = st.button("⚡ GENERATE QUOTE", use_container_width=True, type="primary")


# ═══════════════════════════════════════════════════════════════════
# CENTER — PRICING
# ═══════════════════════════════════════════════════════════════════
with center:
    st.markdown("#### 💰 Pricing")

    if generate or st.session_state["last_result"]:
        if generate:
            # Placeholder until real ML models are wired in
            st.info(
                "🔧 **Model not yet connected.** "
                "Once `src/ml/prediction.py` is populated with your trained models, "
                "real predictions will appear here.",
                icon="⚙️"
            )

            # Show what WOULD be sent to the model
            spec_preview = {
                "vendor": vendor,
                "width": st.session_state["width"],
                "height": st.session_state["height"],
                "gusset": st.session_state["gusset"],
                "web_width": web_width,
                "substrate": st.session_state["substrate"],
                "finish": st.session_state["finish"],
                "seal_type": st.session_state["seal_type"],
                "zipper": st.session_state["zipper"],
                "print_method": st.session_state["print_method"],
                "quantities": st.session_state["quantities"],
                "tariff_rate": st.session_state["tariff_rate"],
            }
            st.markdown("**Spec being quoted:**")
            st.json(spec_preview)
    else:
        st.markdown(
            "<div style='color:#4a5e4a; text-align:center; margin-top:80px; font-size:13px;'>"
            "Configure spec in the left panel<br>and press GENERATE QUOTE"
            "</div>",
            unsafe_allow_html=True
        )


# ═══════════════════════════════════════════════════════════════════
# RIGHT — INTELLIGENCE PANEL
# ═══════════════════════════════════════════════════════════════════
with right:
    st.markdown("#### 🧠 Intelligence")

    st.markdown(
        "<div style='color:#4a5e4a; font-size:11px;'>"
        "SHAP cost drivers, counterfactual optimizations, and customer stickiness "
        "will appear here once models are connected."
        "</div>",
        unsafe_allow_html=True
    )

    # Show routing summary always
    st.markdown("---")
    st.markdown("**Current Routing**")
    st.markdown(f"Vendor: **{vendor.upper()}**")
    st.markdown(f"Web width: **{web_width:.2f}\"**")
    st.markdown(f"Reason: *{routing['reason']}*")
    if routing.get("warnings"):
        for w in routing["warnings"]:
            st.warning(w)
