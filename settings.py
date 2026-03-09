"""
METa QuotR — Configuration & Constants
All vendor rules, spec options, and app settings live here.
Based on actual Supabase data (1,253 quotes analysed March 2026).
"""

# ── App Identity ───────────────────────────────────────────────────
APP_NAME = "METa QuotR"
APP_SUBTITLE = "Intelligent Estimating Platform"
COMPANY_NAME = "Calyx Containers"
VERSION = "0.1.0"

# ── Rep Names ──────────────────────────────────────────────────────
REP_NAMES = ["Dave", "Jake", "Kyle", "Miirhaan", "Xander", "Other"]

# ── Vendor Routing Rules (derived from actual data) ────────────────
#
# Dazpak:   always flexographic, any web width
# Internal: digital, sweet spot 5K–50K, any web width up to ~29"
# Ross:     digital, sweet spot 25K–500K, handles up to 46" web width
# Tedpack:  digital, large format / high volume (synthetic data only for now)
#
# Web width = (height × 2) + gusset
# Routing threshold: if digital + max_qty <= 15K → Internal, else → Ross
# Override: if web_width > 29" → Ross (only Ross quotes above 29" in data)

INTERNAL_SMALL_RUN_MAX_QTY = 15_000   # digital jobs at or below this → Internal
ROSS_MAX_WEB_WIDTH_IN_DATA = 46.0     # widest Ross quote observed
WIDE_FORMAT_THRESHOLD = 29.0          # above this → Ross only

# Dazpak typical tiers (from actual data: 50K, 100K, 250K, 500K dominate)
DAZPAK_TYPICAL_MIN_TIER = 50_000      # soft warning below this

# ── Default Quantity Tiers per Vendor ─────────────────────────────
QUANTITY_TIERS = {
    "dazpak":   [50_000, 100_000, 250_000, 500_000],
    "ross":     [5_000, 10_000, 25_000, 50_000, 100_000, 250_000],
    "internal": [5_000, 10_000, 25_000, 50_000, 100_000, 250_000],
    "tedpack":  [20_000, 50_000, 100_000, 200_000, 300_000],
}

# ── Spec Options ───────────────────────────────────────────────────

SUBSTRATE_OPTIONS = [
    "MET PET",
    "CLR PET",
    "WHT MET PET",
    "ALOX PET",
    "HB CLR PET",
    "Kraft",
    "Custom",
]

FINISH_OPTIONS = [
    "None",
    "Matte",
    "Gloss",
    "Soft Touch",
    "Holographic",
]

SEAL_TYPE_OPTIONS = [
    "Stand Up Pouch",
    "3-Side Seal",
    "Back Seal",
    "Flat Bottom",
    "Quad Seal",
]

GUSSET_TYPE_OPTIONS = [
    "None",
    "K-Seal",
    "K-Seal + Skirt",
    "Plow Bottom",
]

FILL_STYLE_OPTIONS = [
    "Top Fill",
    "Bottom Fill",
    "Side Fill",
]

ZIPPER_OPTIONS = [
    "No Zipper",
    "Press to Close",
    "Slider",
    "Velcro",
    "Child Resistant",
    "Standard CR",
    "Presto CR",
]

TEAR_NOTCH_OPTIONS = [
    "None",
    "Standard",
    "Double",
]

HOLE_PUNCH_OPTIONS = [
    "None",
    "Standard",
    "Butterfly",
    "Euro Slot",
    "Sombrero",
]

CORNER_TREATMENT_OPTIONS = [
    "Straight",
    "Rounded",
]

EMBELLISHMENT_OPTIONS = [
    "None",
    "Hot Stamp Gold",
    "Hot Stamp Silver",
    "Embossing",
    "Spot UV",
]

PRINT_METHOD_OPTIONS = [
    "Digital",
    "Flexographic",
]

# ── Tariff Scenarios ───────────────────────────────────────────────
TARIFF_SCENARIOS = {
    "25%": 0.25,
    "35%": 0.35,
    "45%": 0.45,
    "54%": 0.54,
}
DEFAULT_TARIFF_RATE = 0.35

# ── Landed Cost Defaults ───────────────────────────────────────────
DEFAULT_CONTAINER_RATE = 3_000.0   # USD per 40ft container
DEFAULT_FX_RATE = 0.1376           # CNY → USD

# ── Dimension Constraints ─────────────────────────────────────────
MIN_WIDTH_IN  = 3.0
MIN_HEIGHT_IN = 4.0
MIN_GUSSET_IN = 0.0
MAX_WIDTH_IN  = 20.0
MAX_HEIGHT_IN = 24.0
MAX_GUSSET_IN = 8.0
