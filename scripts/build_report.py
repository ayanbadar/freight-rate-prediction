"""
Builds reports/Freight_Rate_Prediction_Report.pdf — the submission report
covering: data exploration findings, data-quality issues & fixes, the
train/validation split approach, model choice & performance, and the
required fixed December prediction chart.

Run with:
    python scripts/build_report.py

Requires the notebook to have already been run (so that
reports/eda_plots/*.png and scorer_results/candidate_december.png exist).
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "reports" / "eda_plots"
SCORER = ROOT / "scorer_results"
OUT_PATH = ROOT / "reports" / "Freight_Rate_Prediction_Report.pdf"

NAVY = (6, 74, 86)
GREY = (90, 90, 90)
LIGHT = (240, 245, 246)


class Report(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_y(10)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.cell(self.epw / 2, 6, "Freight Rate Prediction - Technical Report", align="L")
        self.cell(self.epw / 2, 6, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*NAVY)
        line_y = self.get_y() + 1
        self.line(self.l_margin, line_y, self.l_margin + self.epw, line_y)
        self.set_y(line_y + 4)

    def footer(self) -> None:
        pass


def h1(pdf: Report, text: str) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(*NAVY)
    pdf.cell(pdf.epw, 10, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*NAVY)
    pdf.set_line_width(0.6)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + pdf.epw, pdf.get_y())
    pdf.ln(4)


def h2(pdf: Report, text: str) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*NAVY)
    pdf.cell(pdf.epw, 8, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(20, 20, 20)
    pdf.ln(1)


def body(pdf: Report, text: str, size: int = 10.5) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", size)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(pdf.epw, 5.6, text)
    pdf.ln(1)


def bullets(pdf: Report, items: list[str], size: int = 10.5) -> None:
    pdf.set_font("Helvetica", "", size)
    pdf.set_text_color(20, 20, 20)
    indent = 8
    for item in items:
        pdf.set_x(pdf.l_margin + 4)
        pdf.cell(4, 5.6, "-")
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(pdf.epw - indent, 5.6, item)
        pdf.ln(0.5)
    pdf.ln(1)


def metric_table(pdf: Report, headers: list[str], rows: list[list[str]]) -> None:
    pdf.set_x(pdf.l_margin)
    col_w = [55, 33.75, 33.75, 33.75, 33.75]
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(*NAVY)
    pdf.set_text_color(255, 255, 255)
    for w, htext in zip(col_w, headers):
        pdf.cell(w, 8, htext, border=0, align="C", fill=True)
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(20, 20, 20)
    for i, row in enumerate(rows):
        fill = i % 2 == 0
        pdf.set_fill_color(*LIGHT)
        for w, val in zip(col_w, row):
            pdf.cell(w, 8, val, border=0, align="C", fill=fill)
        pdf.ln(8)
    pdf.ln(3)


def image_block(pdf: Report, path: Path, caption: str, width: float = 190) -> None:
    if pdf.get_y() > 190:
        pdf.add_page()
    pdf.set_x((210 - width) / 2)
    pdf.image(str(path), x=(210 - width) / 2, w=width)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*GREY)
    pdf.cell(pdf.epw, 6, caption, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(20, 20, 20)
    pdf.ln(2)


def main() -> None:
    pdf = Report(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(10, 12, 10)

    # ------------------------------------------------------------- Cover
    pdf.add_page()
    pdf.ln(50)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(*NAVY)
    pdf.multi_cell(pdf.epw, 12, "Freight Rate Prediction", align="C")
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 15)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(pdf.epw, 9, "Machine Learning Engineer Assessment - Technical Report", align="C")
    pdf.ln(10)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        pdf.epw, 6,
        "Validation approach, train/test split methodology, data-quality findings,\n"
        "modeling decisions, and the fixed December 2025 prediction chart.",
        align="C",
    )
    pdf.ln(20)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(*GREY)
    pdf.multi_cell(pdf.epw, 6, "Model: XGBoost (native categorical support, time-based validation)", align="C")

    # ------------------------------------------------------- 1. Overview
    pdf.add_page()
    h1(pdf, "1. Overview")
    body(
        pdf,
        "The task: predict freight posted_rate for loads using train_test.csv (48,000 labeled "
        "loads, Jan 1 - Oct 31, 2025), then generate predictions for validation.csv (12,000 "
        "unlabeled loads spanning Nov 1 - Dec 31, 2025) and a fixed December 2025 scenario "
        "(december_chart_inputs.csv) used to produce a required prediction chart via the "
        "provided score.py.",
    )
    body(
        pdf,
        "This report covers: (1) key EDA findings, (2) data-quality issues identified and how "
        "they were addressed, (3) the train/validation split methodology, (4) the modeling "
        "approach and results, and (5) the required December prediction chart, including an "
        "important extrapolation issue that was found and fixed along the way.",
    )

    # ----------------------------------------------------- 2. EDA Findings
    h1(pdf, "2. Key EDA Findings")
    h2(pdf, "2.1 Structure")
    bullets(pdf, [
        "48,000 rows x 14 columns; date range 2025-01-01 to 2025-10-31, 304 continuous days "
        "(~158 loads/day, no gaps).",
        "64 unique pickup cities, 64 unique delivery cities, 3 equipment types: "
        "Dry Van (56.7%), Reefer (25.1%), Flatbed (18.2%).",
        "No duplicate load_id values or fully duplicated rows; all ids match the expected "
        "TR-###### format.",
    ])
    h2(pdf, "2.2 Target & Correlations")
    bullets(pdf, [
        "posted_rate is right-skewed (skew ~1.90); applying log1p reduces skew to ~-0.49, "
        "making it a good target transform for training.",
        "distance dominates the linear relationship with posted_rate (corr ~0.91). "
        "weight, market_index, and quote_signal show weak individual linear correlation "
        "(0.03 to -0.04) - likely contributing via nonlinear interactions rather than "
        "additive effects.",
    ])
    image_block(pdf, PLOTS / "overview.png",
                "Figure 1. posted_rate distribution, log-transform, distance relationship, and rate over time.")
    image_block(pdf, PLOTS / "rate_per_mile_by_equipment.png",
                "Figure 2. Mean rate-per-mile by equipment type.")

    # -------------------------------------------- 3. Data Quality Issues
    pdf.add_page()
    h1(pdf, "3. Data-Quality Issues Identified & How They Were Addressed")
    h2(pdf, "3.1 weight: sign-flip errors and missing values")
    body(
        pdf,
        "292 rows had negative weight values. Investigation showed the absolute value of "
        "these negatives (mean ~31,724 lb, range 5,000-47,500 lb) closely matches the "
        "plausible overall weight distribution (mean ~31,029 lb, same range) - strongly "
        "indicating a sign-flip data-entry error rather than genuinely invalid data. "
        "Fix: apply abs(weight). Separately, 300 rows had missing weight; these were imputed "
        "with the equipment-specific median weight, and a weight_was_missing indicator flag "
        "was retained as a model feature.",
    )
    h2(pdf, "3.2 market_index: missing values")
    body(
        pdf,
        "374 rows had missing market_index. Investigation showed market_index is nearly "
        "constant across loads on the same date (mean within-date std ~0.025) but varies "
        "meaningfully month to month (0.89 in September to 1.30 in May) - i.e. it behaves as "
        "a shared, date-level macro signal rather than per-load noise. Fix: impute missing "
        "values using the mean market_index of other loads on the same date, with a global "
        "median fallback for any date with no observed value at all, and a "
        "market_index_was_missing indicator flag retained as a feature.",
    )
    h2(pdf, "3.3 distance vs. coordinates inconsistency")
    body(
        pdf,
        "22 rows had a stated distance that was more than 2x or less than 0.5x the "
        "great-circle (haversine) distance implied by the pickup/delivery coordinates. "
        "Given the small count, these were not dropped (they may reflect legitimate indirect "
        "or multi-stop routing) - instead, haversine_distance and distance_ratio were added "
        "as model features so the model can use route-circuity information directly.",
    )

    # ----------------------------------------- 4. Train/Validation Split
    h1(pdf, "4. Train / Validation Split Approach")
    body(
        pdf,
        "The task is fundamentally forward-in-time forecasting: the labeled data covers "
        "Jan-Oct 2025, and predictions are required for Nov-Dec 2025. A random split would "
        "let the model 'see the future' relative to some validation rows (e.g. training on a "
        "September load while validating on a June load), overstating real-world "
        "performance and failing to reflect the actual prediction task.",
    )
    body(
        pdf,
        "Instead, a time-based split was used: train on 2025-01-01 to 2025-08-31 "
        "(38,477 rows, ~80%), validate on the most recent two months, 2025-09-01 to "
        "2025-10-31 (9,523 rows, ~20%). This holdout mimics predicting into unseen future "
        "dates, matching how the final model is actually used against validation.csv and the "
        "December scenario. The final production model is then refit on the full labeled "
        "dataset (all 48,000 rows) using the number of trees selected via early stopping on "
        "this time-based holdout.",
    )

    # ------------------------------------------------- 5. Feature Engineering
    h1(pdf, "5. Feature Engineering")
    bullets(pdf, [
        "Geo: raw distance, haversine_distance (straight-line), and distance_ratio "
        "(circuity proxy).",
        "Calendar: month, day_of_week, is_weekend, plus sin/cos cyclical encodings of month "
        "and day-of-week to capture seasonality without unbounded extrapolation risk "
        "(see Section 7).",
        "Categorical: pickup, delivery, equipment kept as native pandas categoricals for "
        "XGBoost's built-in categorical split handling (enable_categorical=True), avoiding "
        "manual target-encoding leakage risk. A combined lane feature (pickup + delivery) "
        "lets the model learn lane-pair-specific pricing beyond additive per-city effects.",
        "Target: trained on log1p(posted_rate), inverted with expm1 at prediction time.",
        "Explicitly excluded rate_per_mile (posted_rate / distance) from the feature set, "
        "since it is derived directly from the target and would leak it.",
    ])

    # --------------------------------------------------------- 6. Modeling
    pdf.add_page()
    h1(pdf, "6. Model Choice & Performance")
    body(
        pdf,
        "XGBoost (gradient-boosted trees) was chosen for this tabular regression problem: it "
        "handles nonlinear feature interactions well, supports native categorical features "
        "(avoiding leakage-prone target encoding), trains quickly on this dataset size, and "
        "is robust to the outliers and mixed feature types identified during EDA.",
    )
    h2(pdf, "6.1 Baseline vs. Final Model (time-based holdout, Sep-Oct 2025)")
    metric_table(
        pdf,
        ["Model", "MAE ($)", "RMSE ($)", "MAPE", "R2"],
        [
            ["Linear (distance only)", "494.11", "990.30", "23.12%", "0.579"],
            ["XGBoost (final features)", "153.94", "651.09", "6.46%", "0.818"],
        ],
    )
    body(
        pdf,
        "XGBoost reduces MAE by ~69% and MAPE from 23.1% to 6.5% relative to the distance-only "
        "baseline, confirming that weight, market_index, quote_signal, lane, and seasonality "
        "carry real, nonlinear signal beyond distance alone. The final production model was "
        "refit on all 48,000 labeled rows using 142 trees (the count selected via early "
        "stopping on the time-based holdout).",
    )
    image_block(pdf, PLOTS / "modeling_diagnostics.png",
                "Figure 3. Predicted vs. actual, residuals over time, and feature importance (validation holdout).")
    body(
        pdf,
        "Diagnostics: predicted-vs-actual points hug the diagonal closely, with more spread "
        "at higher rates (longer-haul loads). Residuals over time show no systematic drift "
        "into the holdout period. Feature importance confirms distance/haversine_distance "
        "dominate, with lane, market_index, and weight contributing meaningfully.",
    )

    # ------------------------------------------- 7. December chart & limitation
    pdf.add_page()
    h1(pdf, "7. Fixed December 2025 Prediction Chart")
    body(
        pdf,
        "december_chart_inputs.csv is a fixed synthetic scenario (Lexington -> Fort Wayne, "
        "360 miles, Dry Van, 32,000 lb) repeated for every day of December 2025, with only "
        "the date changing. Critically, the file is missing pickup/delivery coordinates, "
        "market_index, and quote_signal entirely - these had to be reconstructed from the "
        "training data before the model could score it:",
    )
    bullets(pdf, [
        "Coordinates: every city in train_test.csv maps to one consistent (lat, lon) pair "
        "(verified across all 64 cities); looked up directly for Lexington and Fort Wayne.",
        "market_index: modeled as a smooth annual seasonal signal via harmonic regression "
        "(sin/cos of day-of-year plus a linear trend, R^2 ~0.70 on observed daily means) and "
        "extrapolated to December, since market_index is a shared date-level macro signal.",
        "quote_signal: looked up as the historical mean for the same (pickup, delivery, "
        "equipment) lane - the Lexington -> Fort Wayne / Dry Van lane has 21 historical "
        "observations in train_test.csv - falling back to equipment-level, then global, "
        "means if no lane history exists.",
    ])
    h2(pdf, "7.1 An extrapolation issue found and fixed")
    body(
        pdf,
        "Tree-based models cannot extrapolate a trend past the range of values seen in "
        "training: a split such as 'day_of_year <= 250' routes every out-of-range value into "
        "the same branch. Training data's day_of_year tops out at 304 (Oct 31), while "
        "December is day_of_year 335-365 - entirely unobserved. Including raw day_of_year and "
        "days_since_start features caused all 31 December predictions to come out "
        "bit-for-bit identical, a clear symptom of this clamping behavior.",
    )
    body(
        pdf,
        "Fix: day_of_year and days_since_start were removed from the model's feature set, "
        "keeping only bounded, recurring calendar signals (month, day_of_week, is_weekend, "
        "and their sin/cos encodings). Removing these features had virtually no effect on "
        "held-out validation accuracy (MAE moved from $151.80 to $153.94), confirming they "
        "carried little real signal even within the training date range.",
    )
    h2(pdf, "7.2 Why the resulting chart is (honestly) nearly flat")
    body(
        pdf,
        "After the fix, the December predictions are effectively flat at ~$942.81/day. This "
        "is not a remaining bug - it reflects real properties of the data: market_index (corr "
        "~0.03) and quote_signal (corr ~-0.04) have essentially no linear relationship with "
        "posted_rate in this dataset, and the Lexington -> Fort Wayne / Dry Van lane has only "
        "21-32 historical observations, too few for the model to justify further splitting on "
        "day-of-week or market_index for this specific lane/equipment/weight combination. A "
        "flat, stable estimate is the statistically honest output of the model given the "
        "evidence, rather than fabricated day-to-day variation.",
    )
    image_block(pdf, SCORER / "candidate_december.png",
                "Figure 4. Required December 2025 prediction chart, produced by the provided score.py.")

    # --------------------------------------------------------- 8. Summary
    h1(pdf, "8. Summary")
    bullets(pdf, [
        "Data cleaned: weight sign-flip + missing-value imputation, market_index same-date "
        "imputation, geo-consistency features added instead of dropping suspect rows.",
        "Time-based train/validation split (80/20 by date) used throughout, matching the "
        "real forward-in-time forecasting task.",
        "XGBoost with native categorical support cuts MAE by ~69% vs. a distance-only "
        "baseline (MAE $153.94, MAPE 6.46%, R2 0.818 on the time-based holdout).",
        "validation_predictions.csv (12,000 rows) and december_predictions.csv (31 rows) "
        "both validated successfully by the provided score.py, which produced the required "
        "chart shown in Figure 4.",
        "A real tree-extrapolation bug (out-of-range day_of_year/days_since_start) was found "
        "and fixed; the resulting near-flat December chart is an honest reflection of weak "
        "market_index/quote_signal signal and limited lane-specific history, not a modeling "
        "error.",
    ])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT_PATH))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
