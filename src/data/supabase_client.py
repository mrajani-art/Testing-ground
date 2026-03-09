"""
METa QuotR — Supabase Client
Handles all database reads and writes.
"""
import streamlit as st
from supabase import create_client, Client
import pandas as pd
from datetime import datetime
import uuid


@st.cache_resource
def get_supabase() -> Client:
    """
    Get a cached Supabase client.
    Reads credentials from .streamlit/secrets.toml (local)
    or Streamlit Cloud secrets (production).
    """
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def fetch_training_data(vendor: str = None) -> pd.DataFrame:
    """
    Fetch training data from Supabase.
    Joins quotes + quote_prices into one flat DataFrame.

    Args:
        vendor: "dazpak" | "ross" | "internal" | None (all vendors)

    Returns:
        DataFrame with one row per (quote × quantity tier)
    """
    supabase = get_supabase()

    # Fetch quotes
    query = supabase.table("quotes").select("*")
    if vendor:
        query = query.eq("vendor", vendor)
    quotes_resp = query.execute()
    quotes = pd.DataFrame(quotes_resp.data)

    if quotes.empty:
        return pd.DataFrame()

    # Fetch all price tiers for those quotes
    quote_ids = quotes["id"].tolist()
    prices_resp = (
        supabase.table("quote_prices")
        .select("*")
        .in_("quote_id", quote_ids)
        .execute()
    )
    prices = pd.DataFrame(prices_resp.data)

    if prices.empty:
        return pd.DataFrame()

    # Merge
    df = prices.merge(quotes, left_on="quote_id", right_on="id", suffixes=("_price", "_quote"))

    # Keep and rename key columns
    keep_cols = {
        "vendor": "vendor",
        "print_method": "print_method",
        "width": "width",
        "height": "height",
        "gusset": "gusset",
        "print_width": "print_width",
        "bag_area_sqin": "bag_area_sqin",
        "substrate": "substrate",
        "finish": "finish",
        "seal_type": "seal_type",
        "gusset_type": "gusset_type",
        "fill_style": "fill_style",
        "zipper": "zipper",
        "tear_notch": "tear_notch",
        "hole_punch": "hole_punch",
        "corner_treatment": "corner_treatment",
        "embellishment": "embellishment",
        "quantity": "quantity",
        "unit_price": "unit_price",
        "fl_number": "fl_number",
    }

    available = {k: v for k, v in keep_cols.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)

    # Cast numerics
    for col in ["width", "height", "gusset", "print_width", "bag_area_sqin", "unit_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["unit_price", "quantity", "width", "height"])
    df = df[df["unit_price"] > 0]
    df = df[df["quantity"] > 0]

    return df.reset_index(drop=True)


def log_quote_session(
    rep_name: str,
    company_name: str,
    spec: dict,
    quantity: int,
    vendor_routed: str,
    predicted_fob: float,
    predicted_landed: float,
    confidence_lo: float,
    confidence_hi: float,
    tariff_rate: float,
) -> str:
    """
    Log a quote generation event to Supabase.
    Called every time the rep presses Generate Quote.

    Returns: session_id (UUID string)
    """
    supabase = get_supabase()
    session_id = str(uuid.uuid4())

    row = {
        "session_id": session_id,
        "rep_name": rep_name or "Unknown",
        "company_name": company_name or "",
        "spec_json": spec,
        "quantity": quantity,
        "vendor_routed": vendor_routed,
        "predicted_fob": round(predicted_fob, 5) if predicted_fob else None,
        "predicted_landed": round(predicted_landed, 5) if predicted_landed else None,
        "confidence_lo": round(confidence_lo, 5) if confidence_lo else None,
        "confidence_hi": round(confidence_hi, 5) if confidence_hi else None,
        "tariff_rate": tariff_rate,
        "generated_at": datetime.utcnow().isoformat(),
    }

    try:
        supabase.table("quote_sessions").insert(row).execute()
    except Exception as e:
        # Non-fatal — log to console, don't crash the app
        print(f"[quote_sessions] Failed to log: {e}")

    return session_id
