"""
Quote Prediction Engine.

Takes user specifications, routes to the correct vendor,
and generates price predictions with confidence intervals
across multiple quantity tiers.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    DAZPAK_MIN_ORDER_QTY, ROSS_MIN_PRINT_WIDTH_INCHES,
    INTERNAL_MAX_WEB_WIDTH,
    DEFAULT_QTY_TIERS, DAZPAK_DEFAULT_TIERS, ROSS_DEFAULT_TIERS,
)
from src.ml.feature_engineering import prepare_features
from src.ml.model_training import QuoteModelTrainer
from src.ml.internal_calculator import calculate_internal_quote

logger = logging.getLogger(__name__)


class QuotePredictor:
    """Generate price predictions using trained models."""

    def __init__(self):
        self.models = {}
        self._loaded = False

    def load_models(self):
        """Load trained models for both vendors."""
        for vendor in ["dazpak", "ross", "internal"]:
            try:
                self.models[vendor] = QuoteModelTrainer.load(vendor)
                logger.info(f"Loaded {vendor} model")
            except FileNotFoundError:
                logger.warning(f"No trained model found for {vendor}")
        self._loaded = True

    def predict(self, specs: dict, quantity_tiers: list[int],
                vendor_override: Optional[str] = None) -> dict:
        """
        Generate price predictions for the given specs across quantity tiers.

        Args:
            specs: Dict with keys matching the input parameters
                   (width, height, gusset, substrate, finish, etc.)
            quantity_tiers: List of quantities to quote
            vendor_override: Force a specific vendor (skip routing)

        Returns:
            {
                "vendor": "dazpak" | "ross",
                "print_method": "flexographic" | "digital",
                "routing_reason": str,
                "predictions": [
                    {
                        "quantity": int,
                        "unit_price": float,
                        "total_price": float,
                        "lower_bound": float,
                        "upper_bound": float,
                    }, ...
                ],
                "cost_factors": {feature: importance, ...},
                "model_metrics": {...},
                "warnings": [str, ...],
            }
        """
        if not self._loaded:
            self.load_models()

        warnings = []

        # ── Route to vendor ─────────────────────────────────────────
        width = float(specs.get("width", 0))
        height = float(specs.get("height", 0))
        gusset = float(specs.get("gusset", 0))
        print_width = height * 2 + gusset

        if vendor_override:
            vendor = vendor_override
            routing_reason = f"Manually selected: {vendor}"
        else:
            vendor, routing_reason = self._route_vendor(
                specs.get("print_method", ""),
                print_width, quantity_tiers
            )

        # Validate vendor constraints
        vendor_warnings = self._validate_vendor_constraints(
            vendor, print_width, quantity_tiers
        )
        warnings.extend(vendor_warnings)

        # ── Internal vendor: use deterministic calculator ─────────
        if vendor == "internal":
            result = calculate_internal_quote(specs, quantity_tiers)
            result["warnings"] = warnings + result.get("warnings", [])
            result["routing_reason"] = routing_reason
            return result

        # ── Dazpak / Ross: use ML model ──────────────────────────
        if vendor not in self.models:
            return {
                "vendor": vendor,
                "print_method": "flexographic" if vendor == "dazpak" else "digital",
                "routing_reason": routing_reason,
                "predictions": [],
                "cost_factors": {},
                "model_metrics": {},
                "warnings": warnings + [f"No trained model available for {vendor}"],
                "error": f"Model for {vendor} not found. Train models first.",
            }

        model = self.models[vendor]

        # ── Generate predictions per tier ───────────────────────────
        predictions = []
        for qty in quantity_tiers:
            row = {**specs, "quantity": qty}
            pred = self._predict_single(model, row)
            predictions.append({
                "quantity": qty,
                "unit_price": round(pred["point"], 5),
                "total_price": round(pred["point"] * qty, 2),
                "lower_bound": round(pred["lower"], 5),
                "upper_bound": round(pred["upper"], 5),
                "confidence_range": round(pred["upper"] - pred["lower"], 5),
            })

        # ── Cost factor breakdown ───────────────────────────────────
        cost_factors = self._compute_cost_factors(model, specs, quantity_tiers[0])

        return {
            "vendor": vendor,
            "print_method": "flexographic" if vendor == "dazpak" else "digital",
            "routing_reason": routing_reason,
            "print_width": round(print_width, 3),
            "bag_area": round(width * height, 3),
            "predictions": predictions,
            "cost_factors": cost_factors,
            "model_metrics": model.metrics,
            "warnings": warnings,
        }

    def _route_vendor(self, print_method: str, print_width: float,
                      qtys: list[int]) -> tuple[str, str]:
        """
        Determine which vendor should handle this quote.

        Rules:
        1. If user selects Flexographic → Dazpak
        2. If user selects Digital:
           - web width < 12" → Internal (HP 6900)
           - web width ≥ 12" → Ross
        3. Auto-routing based on quantity and print width
        """
        max_qty = max(qtys) if qtys else 0

        if print_method.lower() == "flexographic":
            return "dazpak", "Print method: Flexographic → Dazpak"

        if print_method.lower() == "digital":
            if print_width <= INTERNAL_MAX_WEB_WIDTH:
                return "internal", f"Print method: Digital, web width {print_width:.1f}\" ≤ 12\" → Internal (HP 6900)"
            else:
                return "ross", f"Print method: Digital, web width {print_width:.1f}\" > 12\" → Ross"

        # Auto-route
        if max_qty >= DAZPAK_MIN_ORDER_QTY:
            return "dazpak", f"Auto-route: max qty {max_qty:,} ≥ {DAZPAK_MIN_ORDER_QTY:,} → Dazpak (Flexographic)"

        if print_width <= INTERNAL_MAX_WEB_WIDTH:
            return "internal", f"Auto-route: web width {print_width:.1f}\" ≤ 12\" → Internal (HP 6900)"

        if print_width > ROSS_MIN_PRINT_WIDTH_INCHES:
            return "ross", f"Auto-route: web width {print_width:.1f}\" > 12\" → Ross (Digital)"

        # Default to internal for smaller runs
        return "internal", "Auto-route: smaller quantities → Internal (HP 6900)"

    def _validate_vendor_constraints(self, vendor: str, print_width: float,
                                     qtys: list[int]) -> list[str]:
        """Check vendor business rules and return warnings."""
        warnings = []

        if vendor == "dazpak":
            below_moq = [q for q in qtys if q < DAZPAK_MIN_ORDER_QTY]
            if below_moq:
                warnings.append(
                    f"⚠ Dazpak MOQ is {DAZPAK_MIN_ORDER_QTY:,} units. "
                    f"Quantities {below_moq} are below minimum."
                )

        if vendor == "ross":
            if print_width <= ROSS_MIN_PRINT_WIDTH_INCHES:
                warnings.append(
                    f"⚠ Ross requires web width > 12\". "
                    f"Current web width: {print_width:.2f}\" "
                    f"(Height×2 + Gusset). Job may be rejected."
                )

        if vendor == "internal":
            if print_width > INTERNAL_MAX_WEB_WIDTH:
                warnings.append(
                    f"⚠ Internal (HP 6900) handles web width ≤ 12\". "
                    f"Current web width: {print_width:.2f}\". "
                    f"Consider routing to Ross instead."
                )

        return warnings

    def _predict_single(self, model: QuoteModelTrainer, row: dict) -> dict:
        """Predict price for a single spec+quantity combination."""
        df = pd.DataFrame([row])
        df = prepare_features(df)

        X = model.preprocessor.transform(df)
        point_raw = float(model.model_point.predict(X)[0])
        lower_raw = float(model.model_lower.predict(X)[0])
        upper_raw = float(model.model_upper.predict(X)[0])

        # Back-transform if model was trained on log(price)
        if model.use_log_target:
            point = np.exp(point_raw)
            lower = np.exp(lower_raw)
            upper = np.exp(upper_raw)
        else:
            point = point_raw
            lower = lower_raw
            upper = upper_raw

        # Ensure bounds are sensible
        point = max(point, 0.001)
        lower = max(lower, 0.001)
        upper = max(upper, point)
        if lower > point:
            lower = point * 0.85

        return {"point": point, "lower": lower, "upper": upper}

    def _compute_cost_factors(self, model: QuoteModelTrainer,
                              specs: dict, base_qty: int) -> dict:
        """
        Compute a breakdown of which features most influence the price.
        Uses feature importances + marginal effect estimation.
        """
        # Start with model's global feature importances
        importances = model.feature_importances

        # Annotate with the user's actual spec values for context
        cost_factors = {}
        for feature, importance in importances.items():
            if importance < 0.01:  # Skip negligible factors
                continue

            value = specs.get(feature, "—")
            if feature == "log_quantity":
                value = f"log₁₀({base_qty:,}) = {np.log10(base_qty):.2f}"
            elif feature == "bag_area_sqin":
                value = f"{float(specs.get('width', 0)) * float(specs.get('height', 0)):.1f} sq in"
            elif feature == "print_width":
                h = float(specs.get("height", 0))
                g = float(specs.get("gusset", 0))
                value = f"{h * 2 + g:.2f}\" (H×2+G)"
            elif feature == "area_x_logqty":
                value = "area × volume interaction"
            elif feature == "quantity":
                value = f"{base_qty:,}"
            elif feature == "zipper_width":
                w = float(specs.get("width", 0))
                has_z = specs.get("zipper", "No Zipper") != "No Zipper"
                value = f"{w:.1f}\" × {'zipper' if has_z else 'no zipper'}"
            elif feature == "ross_converting_cost":
                w = float(specs.get("width", 0))
                has_z = specs.get("zipper", "No Zipper") != "No Zipper"
                from src.ml.feature_engineering import _ross_converting_cost
                value = f"${_ross_converting_cost(w, has_z):.4f}/unit"
            elif feature == "ross_setup_per_unit":
                value = f"${102.25 / base_qty:.5f}/unit (setup amortized)"
            elif feature == "print_area_msi":
                h = float(specs.get("height", 0))
                g = float(specs.get("gusset", 0))
                pw = h * 2 + g
                value = f"{pw * h / 1000:.4f} MSI"

            cost_factors[feature] = {
                "importance": round(importance * 100, 1),
                "value": str(value),
            }

        return cost_factors
