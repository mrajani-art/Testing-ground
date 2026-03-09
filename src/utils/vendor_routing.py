"""
Vendor Routing — METa QuotR
Based on actual Calyx quote data (1,253 quotes, Supabase).

WHAT THE DATA SHOWS:
  - Dazpak: always flexographic, any web width (8.5"–29")
  - Internal: digital, dominates small runs (5K–50K), any web width
  - Ross: digital, dominates larger runs + wide format (up to 46")
  - Web width does NOT hard-separate Internal from Ross
    (both quote the same widths — routing is driven by volume + width together)

ROUTING LOGIC:
  Flexographic → Dazpak
  Digital + web_width > 29"  → Ross (only Ross goes this wide)
  Digital + max_qty <= 15000 → Internal (small-run sweet spot)
  Digital + everything else  → Ross
"""


def calculate_web_width(height: float, gusset: float = 0.0) -> float:
    """Web width = Height × 2 + Gusset (inches)."""
    return round((height * 2) + gusset, 3)


def route_vendor(
    print_method: str,
    height: float,
    gusset: float,
    quantities: list[int],
) -> dict:
    """
    Determine vendor routing from print method, dimensions, and quantities.

    Returns:
        vendor: "dazpak" | "ross" | "internal"
        web_width: float
        reason: str
        warnings: list[str]
    """
    web_width = calculate_web_width(height, gusset)
    max_qty = max(quantities) if quantities else 0
    warnings = []

    # ── Flexographic → always Dazpak ──────────────────────────────
    if print_method.lower() == "flexographic":
        vendor = "dazpak"
        reason = f"Print method: Flexographic → Dazpak"

        # Soft warning — Dazpak tiers typically start at 50K
        if max_qty < 25_000:
            warnings.append(
                f"⚠ Dazpak typical minimum tier is 50K units. "
                f"Max quantity {max_qty:,} is unusually low for flexographic. "
                f"Confirm with Dazpak before quoting."
            )

    # ── Digital routing ───────────────────────────────────────────
    elif print_method.lower() == "digital":

        if web_width > 29.0:
            # Only Ross quotes above 29" in our data
            vendor = "ross"
            reason = f"Digital + web width {web_width:.1f}\" > 29\" → Ross (only Ross handles this width)"

        elif max_qty <= 15_000:
            # Internal dominates small digital runs
            vendor = "internal"
            reason = f"Digital + max qty {max_qty:,} ≤ 15K → Internal HP 6900 (small-run sweet spot)"

        else:
            # Ross handles larger digital volumes
            vendor = "ross"
            reason = f"Digital + max qty {max_qty:,} > 15K → Ross HP 200K"

    else:
        # Unknown print method — default to internal with warning
        vendor = "internal"
        reason = f"Unknown print method '{print_method}' — defaulting to Internal"
        warnings.append(f"⚠ Unrecognized print method: '{print_method}'. Defaulted to Internal.")

    return {
        "vendor": vendor,
        "web_width": web_width,
        "reason": reason,
        "warnings": warnings,
    }


def get_default_quantity_tiers(vendor: str) -> list[int]:
    """
    Default quantity tiers shown in UI for each vendor.
    Based on actual tier distributions in Supabase data.
    """
    tiers = {
        "dazpak":   [50_000, 100_000, 250_000, 500_000],
        "ross":     [5_000, 10_000, 25_000, 50_000, 100_000, 250_000],
        "internal": [5_000, 10_000, 25_000, 50_000, 100_000, 250_000],
        "tedpack":  [20_000, 50_000, 100_000, 200_000, 300_000],
    }
    return tiers.get(vendor, [10_000, 25_000, 50_000, 100_000])
