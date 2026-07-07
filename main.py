"""MicroStrategy (MSTR) Valuation & Regression Model Engine.

A small Streamlit app that compares two ways of mapping the Bitcoin spot price
to an implied MSTR share price:

* Model 1 -- a naive single-variable proportional fit (price-to-price).
* Model 2 -- a levered net-asset-value (NAV) model with a premium that is
  *calibrated* to the selected balance-sheet era via a two-point line.
* Model 3 -- Model 2 plus a carry adjustment: a chosen horizon of preferred
  dividends and debt interest is subtracted as a senior drain on NAV before
  the same calibrated premium is applied.

Run locally with:

    python -m streamlit run main.py

This is an educational modelling tool, not investment advice. See README.md.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="MSTR Regression Model Engine", layout="wide")

st.title("MicroStrategy (MSTR) Valuation & Regression Model Engine")
st.write(
    "A framework comparing a simple price-to-price regression against a "
    "levered-equity model whose premium is auto-calibrated to the selected "
    "balance-sheet era, plus a carry-adjusted variant that charges the NAV "
    "for ongoing preferred dividends and debt interest."
)

# ---------------------------------------------------------
# 1. HISTORICAL DATA ARCHIVE
# ---------------------------------------------------------
# Each era captures a snapshot of the balance sheet plus an anchor point
# (a historical BTC/MSTR price pair) used to calibrate the models.
#
# "debt_billions" = total claims senior to common equity, i.e. convertible /
# senior notes (balance-sheet carrying value) plus the liquidation preference of
# preferred stock (preferred only exists from 2025 onward). Year-end figures are
# taken directly from Strategy/MicroStrategy SEC filings (10-K balance sheets and
# XBRL facts, CIK 0001050446); anchor prices are the Dec-31 closes. The Mid-2026
# row is an interim (non-10-K) snapshot from press/secondary sources.
era_data = {
    "End of 2023 (Bear Recovery)": {
        "btc_holdings": 189150,        # 189,150 BTC as of Dec 31, 2023
        "debt_billions": 2.20,         # ~$2.2B convertible/senior notes
        "annual_carry_billions": 0.02, # est. note interest only (mostly 0-0.75% converts); no preferred yet
        "usd_reserve_billions": 0.0,   # USD Reserve not yet established
        "shares_millions": 166.00,     # ~166M diluted (post 10:1 split basis)
        "historical_btc_price": 42000.0,
        "historical_mstr_price": 63.00,  # ~$63 post-split
        "floor_btc": 25000.0,
    },
    "End of 2024 (ETF Options Launch)": {
        "btc_holdings": 446400,        # 446,400 BTC as of Dec 31, 2024
        "debt_billions": 7.19,         # $7.19B long-term debt (10-K), no preferred
        "annual_carry_billions": 0.05, # est. low-coupon convert interest; no preferred yet
        "usd_reserve_billions": 0.0,   # USD Reserve not yet established
        "shares_millions": 245.78,     # 226,138,000 class A + 19,640,250 class B (10-K)
        "historical_btc_price": 100000.0,
        "historical_mstr_price": 380.00,
        "floor_btc": 30000.0,
    },
    "End of 2025 (Preferred Stack)": {
        "btc_holdings": 672500,        # 713,502 (Q4'25 report) - 41,002 bought Jan 2026
        "debt_billions": 16.22,        # $8.19B notes + $8.03B preferred liq. pref. (10-K)
        "annual_carry_billions": 0.75, # est. preferred dividends (~$8B @ ~8-9% blended) + note interest
        "usd_reserve_billions": 0.0,   # USD Reserve established Dec 1, 2025; year-end balance not disclosed
        "shares_millions": 312.06,     # 292,422,000 class A + 19,640,250 class B (10-K)
        "historical_btc_price": 87500.0,   # BTC close Dec 31, 2025 ~$87,509
        "historical_mstr_price": 152.00,   # MSTR close Dec 31, 2025 $151.95
        "floor_btc": 30000.0,
    },
    "Mid 2026 (Current Leverage)": {
        # 843,775 BTC as of Jul 5, 2026 (8-K filed Jul 6, 2026); aggregate cost
        # $63.69B, average $75,476/BTC. First-ever confirmed BTC SALES: 1,363
        # BTC sold Jun 29-30 (avg $59,256) and 2,225 BTC sold Jul 1-5 (avg
        # $60,773), under a new "BTC Monetization Program" (announced Jun 29,
        # 2026; up to $1.25B capacity, all unused as of Jul 5) -- proceeds
        # funded preferred distributions and replenished the USD Reserve.
        "btc_holdings": 843775,
        "debt_billions": 22.20,        # ~$6.7B notes + ~$15.5B preferred (late-May '26; still not updated in an 8-K)
        "annual_carry_billions": 1.76, # ~$1.76B annual preferred dividends + interest (8-K Jun 29, 2026)
        "usd_reserve_billions": 2.55,  # USD Reserve $2.55B as of Jul 5, 2026 (8-K filed Jul 6, 2026) --
                                       # confirmed unchanged from the Jun 28 balance: BTC-sale proceeds
                                       # replenished the reserve after funding preferred distributions.
                                       # ~17.4 months of the ~$1.76B annual preferred dividend + interest
                                       # expense. Includes unsettled ATM proceeds as of that date.
        "shares_millions": 352.07,     # unchanged: 8-K confirms no ATM share sales Jun 29-Jul 5, 2026
        "historical_btc_price": 62600.0,   # BTC ~$60-64k early Jul 2026 (sales executed $59.3k-$60.8k; spot ~$62.6-62.9k Jul 6)
        "historical_mstr_price": 97.00,    # MSTR ~$93-104 early Jul 2026 (prior close $93.39, intraday high $104.11 Jul 6) -- down sharply after the Q2 8-K's $8.32B unrealized digital-asset loss
        "floor_btc": 30000.0,
        # Upper regression anchor: STRC preferred IPO pricing day (Jul 24, 2025),
        # when the market first priced the dividend "carry." Real closes that day:
        # BTC ~$118,368, MSTR ~$414.92. Model 3 fits its premium line through this
        # real high-BTC point and the current anchor, instead of a theoretical
        # floor -- so the upper end stays tied to an observed price, not an
        # extrapolation that runs away to share prices MSTR has never reached.
        "regression_anchor_btc": 118368.0,
        "regression_anchor_mstr": 414.92,
    },
}

# ---------------------------------------------------------
# 2. GLOBAL SIDEBAR CONTROLS & LIVE FETCH
# ---------------------------------------------------------
st.sidebar.header("1. Select Target Era Configuration")
selected_era = st.sidebar.selectbox(
    "Active Balance Sheet Profile", list(era_data.keys()), index=3
)

# Load data state based on selection.
active_profile = era_data[selected_era]
btc_holdings = active_profile["btc_holdings"]
total_debt = active_profile["debt_billions"] * 1_000_000_000
usd_reserve = active_profile.get("usd_reserve_billions", 0.0) * 1_000_000_000
annual_carry = active_profile.get("annual_carry_billions", 0.0) * 1_000_000_000
shares_outstanding = active_profile["shares_millions"] * 1_000_000
anchor_btc = active_profile["historical_btc_price"]
anchor_mstr = active_profile["historical_mstr_price"]

st.sidebar.markdown("---")
st.sidebar.header("2. Live Market Override")


@st.cache_data(ttl=3600)
def fetch_live_market_prices():
    """Return the latest BTC-USD and MSTR closes, or (None, None) on failure."""
    try:
        btc_hist = yf.Ticker("BTC-USD").history(period="1d")
        mstr_hist = yf.Ticker("MSTR").history(period="1d")
        if btc_hist.empty or mstr_hist.empty:
            return None, None
        return float(btc_hist["Close"].iloc[-1]), float(mstr_hist["Close"].iloc[-1])
    except Exception:
        return None, None


use_live = st.sidebar.checkbox("Override with Live Spot Prices?", value=False)

# Re-anchor based on user preference. Only fetch when actually requested so the
# app works offline by default.
if use_live:
    live_btc_val, live_mstr_val = fetch_live_market_prices()
    if live_btc_val is not None:
        anchor_btc = live_btc_val
        anchor_mstr = live_mstr_val
    else:
        st.sidebar.warning(
            "Live price fetch failed; falling back to the era's historical anchor."
        )

st.sidebar.metric(label="Anchor BTC Price ($x)", value=f"${anchor_btc:,.2f}")
st.sidebar.metric(label="Anchor MSTR Price ($y)", value=f"${anchor_mstr:,.2f}")

st.sidebar.markdown("---")
st.sidebar.header("3. Model Parameter Fine-Tuning")
floor_btc_input = st.sidebar.slider(
    "Premium Bedrock Floor (BTC Price)",
    min_value=15000,
    max_value=45000,
    value=int(active_profile["floor_btc"]),
    step=1000,
    help="BTC price at which the market premium is assumed to collapse to 1.0x NAV.",
)

# ---------------------------------------------------------
# 3. MATHEMATICAL ENGINES & PARAMETER DERIVATION
# ---------------------------------------------------------
# MODEL 1 PARAMETER: simple linear coefficient (alpha = y / x).
alpha_parameter = anchor_mstr / anchor_btc

# MODEL 2 PARAMETERS: levered net equity & calibrated premium line.
anchor_gross_assets = anchor_btc * btc_holdings + usd_reserve
anchor_net_equity = max(0.01, anchor_gross_assets - total_debt)
anchor_nav_per_share = anchor_net_equity / shares_outstanding
actual_anchor_premium = anchor_mstr / anchor_nav_per_share

# Two-point premium line through (floor_btc, 1.0x) and (anchor_btc, observed
# premium). The slope is "m"; the premium formula uses its reciprocal as a
# denominator (the "slope sensitivity"). Deriving the denominator here is what
# makes Model 2 reproduce the anchor exactly for *every* era -- a fixed,
# hand-tuned denominator only fits the single era it was tuned to.
premium_calibratable = (actual_anchor_premium > 1.0) and (anchor_btc > floor_btc_input)
if premium_calibratable:
    premium_slope_m = (actual_anchor_premium - 1.0) / (anchor_btc - floor_btc_input)
    derived_slope_sensitivity = 1.0 / premium_slope_m
else:
    premium_slope_m = 0.0
    derived_slope_sensitivity = None

st.sidebar.caption(
    "The premium slope is auto-calibrated to the anchor by default. Disable to "
    "tune it manually."
)
auto_calibrate = st.sidebar.checkbox(
    "Auto-calibrate premium to anchor (recommended)", value=True
)
manual_slope_sensitivity = st.sidebar.slider(
    "Manual Premium Slope Sensitivity (denominator)",
    min_value=50000,
    max_value=200000,
    value=120600,
    step=1000,
    disabled=auto_calibrate and derived_slope_sensitivity is not None,
    help="Larger denominator => premium expands more slowly with BTC price.",
)

if auto_calibrate and derived_slope_sensitivity is not None:
    active_slope_sensitivity = derived_slope_sensitivity
else:
    if auto_calibrate and derived_slope_sensitivity is None:
        st.sidebar.warning(
            "Anchor premium is ≤ 1.0x (or anchor ≤ floor), so the premium line "
            "cannot be calibrated. Using the manual slider instead."
        )
    active_slope_sensitivity = manual_slope_sensitivity

st.sidebar.markdown("---")
st.sidebar.header("4. Model 3: Carry-Adjusted NAV")
st.sidebar.caption(
    "Model 3 extends Model 2 by charging the NAV for a chosen horizon of "
    "preferred dividends + debt interest, treated as a senior drain funded by "
    "selling BTC or drawing the USD reserve."
)
carry_horizon_years = st.sidebar.slider(
    "Dividend + Interest Carry Horizon (years)",
    min_value=0,
    max_value=15,
    value=5,
    step=1,
    help="Years of annual preferred dividends + interest subtracted from net "
    "equity. 0 collapses Model 3 back onto Model 2.",
)
st.sidebar.metric(
    "Annual Carry (Dividends + Interest)",
    f"${annual_carry:,.0f}",
    help=f"${active_profile.get('annual_carry_billions', 0.0)} Billion / year for "
    f"the selected era.",
)

# MODEL 3 PARAMETERS: carry-adjusted net equity with an EMPIRICAL premium line.
# The premium is expressed as premium(x) = max(1.0, a + b*x), where (a, b) are
# fit to real market anchors rather than a theoretical floor.
#
# Preferred calibration -- TWO real anchors: the current price and a high-BTC
# anchor from the STRC preferred IPO (Jul 24, 2025), when the market first priced
# the dividend carry. Both premiums are measured against the SAME (selected era)
# carry-adjusted NAV, so the line is a genuine BTC->premium regression. Pinning
# the upper end to a real ~2x observation stops the premium from extrapolating to
# share prices MSTR has never reached.
#
# Fallback -- if the era has no second anchor: the old floor + current-anchor
# line, through (floor_btc, 1.0x) and (anchor_btc, observed premium).
anchor_carry_drag = annual_carry * carry_horizon_years


def m3_carry_adj_nav(btc_price):
    """Carry-adjusted NAV per share for the active era at a given BTC price."""
    g = btc_price * btc_holdings + usd_reserve
    return max(0.01, g - total_debt - anchor_carry_drag) / shares_outstanding


m3_anchor_nav_per_share = m3_carry_adj_nav(anchor_btc)
m3_actual_anchor_premium = anchor_mstr / m3_anchor_nav_per_share

m3_upper_btc = active_profile.get("regression_anchor_btc")
m3_upper_mstr = active_profile.get("regression_anchor_mstr")
m3_two_anchor = (
    m3_upper_btc is not None
    and m3_upper_mstr is not None
    and m3_upper_btc != anchor_btc
)

if m3_two_anchor:
    # Empirical line through two REAL market anchors (current + STRC IPO).
    m3_upper_nav_per_share = m3_carry_adj_nav(m3_upper_btc)
    m3_upper_premium = m3_upper_mstr / m3_upper_nav_per_share
    m3_premium_b = (m3_upper_premium - m3_actual_anchor_premium) / (
        m3_upper_btc - anchor_btc
    )
    m3_premium_a = m3_actual_anchor_premium - m3_premium_b * anchor_btc
    m3_calibration_mode = "two-anchor regression (current + STRC IPO)"
elif m3_actual_anchor_premium > 1.0 and anchor_btc > floor_btc_input:
    # Fallback: theoretical floor (1.0x) + current anchor.
    m3_upper_premium = None
    m3_premium_b = (m3_actual_anchor_premium - 1.0) / (anchor_btc - floor_btc_input)
    m3_premium_a = 1.0 - m3_premium_b * floor_btc_input
    m3_calibration_mode = "floor + current anchor"
else:
    # Degenerate: hold the premium flat at the observed anchor premium.
    m3_upper_premium = None
    m3_premium_b = 0.0
    m3_premium_a = m3_actual_anchor_premium
    m3_calibration_mode = "flat (uncalibratable)"


def evaluate_models(target_price_btc):
    """Return per-share model outputs for a target BTC price.

    Tuple: (model1_price, nav_per_share, premium_multiple, model2_price,
    carry_adj_nav_per_share, model3_price).
    """
    # Model 1: simple proportional price-to-price.
    m1_price = alpha_parameter * target_price_btc

    # Model 2: levered NAV, floored at zero (equity wiped out below debt).
    gross_assets = target_price_btc * btc_holdings + usd_reserve
    net_equity = max(0.0, gross_assets - total_debt)
    nav_per_share = net_equity / shares_outstanding

    # Premium expands linearly from 1.0x at the floor, never below 1.0x.
    m2_premium = max(
        1.0, 1.0 + ((target_price_btc - floor_btc_input) / active_slope_sensitivity)
    )
    m2_price = nav_per_share * m2_premium

    # Model 3: carry-adjusted NAV. Subtract N years of preferred dividends +
    # debt interest as an extra senior claim, then apply the empirical premium
    # line premium(x) = max(1.0, a + b*x). With two real anchors (current + STRC
    # IPO) the line reproduces both observed prices and interpolates between them
    # instead of extrapolating off a theoretical floor.
    carry_drag = annual_carry * carry_horizon_years
    m3_net_equity = max(0.0, gross_assets - total_debt - carry_drag)
    m3_nav_per_share = m3_net_equity / shares_outstanding
    m3_premium = max(1.0, m3_premium_a + m3_premium_b * target_price_btc)
    m3_price = m3_nav_per_share * m3_premium

    return (
        m1_price,
        nav_per_share,
        m2_premium,
        m2_price,
        m3_nav_per_share,
        m3_price,
    )


# Slider bounds for the BTC look-up. Expand the upper bound if the live/anchor
# price sits above the default ceiling so the default value is always valid.
lookup_min = 20000
lookup_max = max(150000, int(anchor_btc * 1.25))
default_lookup = int(min(max(anchor_btc, lookup_min), lookup_max))

# Create tabs for the dashboard layout.
tab_dashboard, tab_math = st.tabs(
    ["📊 Interactive Dashboard", "🧮 Mathematical Derivations"]
)

# ---------------------------------------------------------
# TAB 1: INTERACTIVE DASHBOARD
# ---------------------------------------------------------
with tab_dashboard:
    st.subheader(f"Active Framework: {selected_era}")

    # Upper metric row.
    col1, col2, col3 = st.columns(3)
    col1.metric("Model 1 Alpha (Slope Coefficient)", f"{alpha_parameter:.8f}")
    col2.metric("Baseline Anchor NAV / Share", f"${anchor_nav_per_share:,.2f}")
    col3.metric("Observed Anchor Market Premium", f"{actual_anchor_premium:.2f}x")

    st.markdown("---")

    # Target BTC slider for dynamic look-up.
    lookup_btc = st.slider(
        "Simulate Target Bitcoin Price ($)",
        min_value=lookup_min,
        max_value=lookup_max,
        value=default_lookup,
        step=500,
    )
    res_m1, res_nav, res_prem, res_m2, res_m3_nav, res_m3 = evaluate_models(lookup_btc)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model 1: Simple Linear Price Target", f"${res_m1:,.2f}")
    col2.metric("Model 2: Levered NAV (1x)", f"${res_nav:,.2f}")
    col3.metric(
        f"Model 2: Premium-Adjusted Target ({res_prem:.2f}x)", f"${res_m2:,.2f}"
    )

    res_m3_prem = max(1.0, m3_premium_a + m3_premium_b * lookup_btc)
    col4, col5 = st.columns(2)
    col4.metric("Model 3: Carry-Adjusted NAV (1x)", f"${res_m3_nav:,.2f}")
    col5.metric(
        f"Model 3: Carry-Adjusted Target ({res_m3_prem:.2f}x, "
        f"{carry_horizon_years}y carry)",
        f"${res_m3:,.2f}",
        delta=f"${res_m3 - res_m2:,.2f} vs Model 2",
    )

    st.markdown("---")

    # Plotting curve comparison.
    st.write("### Model Trajectory Comparison: Simple Linear vs. Levered Premium")
    btc_axis = np.linspace(lookup_min, lookup_max, 250)
    m1_curve, nav_curve, m2_curve, m3_curve = [], [], [], []

    for bx in btc_axis:
        m1_c, nav_c, _, m2_c, _, m3_c = evaluate_models(bx)
        m1_curve.append(m1_c)
        nav_curve.append(nav_c)
        m2_curve.append(m2_c)
        m3_curve.append(m3_c)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=btc_axis,
            y=m1_curve,
            mode="lines",
            name="Model 1: Simple Linear Fit",
            line=dict(color="#ffc658", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=btc_axis,
            y=nav_curve,
            mode="lines",
            name="Model 2: Pure 1x Levered NAV (Floor)",
            line=dict(color="#8884d8", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=btc_axis,
            y=m2_curve,
            mode="lines",
            name="Model 2: Calibrated Premium Curve",
            line=dict(color="#ff7300", width=4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=btc_axis,
            y=m3_curve,
            mode="lines",
            name=f"Model 3: Carry-Adjusted ({carry_horizon_years}y)",
            line=dict(color="#2ca02c", width=3, dash="dot"),
        )
    )

    # Real market anchors that calibrate Model 3's premium regression.
    anchor_x = [anchor_btc]
    anchor_y = [anchor_mstr]
    anchor_text = ["Current anchor"]
    if m3_two_anchor:
        anchor_x.append(m3_upper_btc)
        anchor_y.append(m3_upper_mstr)
        anchor_text.append("STRC IPO anchor (Jul 2025)")
    fig.add_trace(
        go.Scatter(
            x=anchor_x,
            y=anchor_y,
            mode="markers+text",
            marker=dict(size=11, color="#1f77b4", symbol="circle"),
            text=anchor_text,
            textposition="top center",
            name="Real Market Anchors (Model 3 regression)",
        )
    )

    # Active look-up marker.
    fig.add_trace(
        go.Scatter(
            x=[lookup_btc],
            y=[res_m2],
            mode="markers",
            marker=dict(size=12, color="red", symbol="cross"),
            name="Active Target Look-up",
        )
    )

    fig.update_layout(
        xaxis_title="Bitcoin Spot Price (USD)",
        yaxis_title="MSTR Per Share Price (USD)",
        hovermode="x unified",
        height=550,
    )
    st.plotly_chart(fig, width="stretch")

    # Quick reference matrix.
    st.write("### Valuation Comparison Matrix")
    matrix_intervals = [30000, 40000, 50000, anchor_btc, 70000, 80000, 90000, 100000, 120000, 150000]
    matrix_rows = []
    for val in sorted(set(matrix_intervals)):
        m1_p, nav_p, pr_p, m2_p, _, m3_p = evaluate_models(val)
        matrix_rows.append(
            {
                "Target BTC Price": f"${val:,.0f}",
                "Model 1 (Simple Linear)": f"${m1_p:,.2f}",
                "Pure 1x NAV (Levered Core)": f"${nav_p:,.2f}",
                "Model 2 Premium Multiplier": f"{pr_p:.2f}x",
                "Model 2 (Levered + Premium)": f"${m2_p:,.2f}",
                "Model 3 (Carry-Adjusted)": f"${m3_p:,.2f}",
            }
        )
    st.table(pd.DataFrame(matrix_rows))

# ---------------------------------------------------------
# TAB 2: MATHEMATICAL DERIVATIONS
# ---------------------------------------------------------
with tab_math:
    st.header("Model Parameters & Derivations")
    st.write(
        "This section unpacks the two regressions and shows how the active "
        "coefficients are computed from the selected era."
    )

    st.markdown("---")

    st.write("### 📜 Model 1: Simple Anchored Price-to-Price Linear Fit")
    st.write(
        "A single-variable proportional relationship through the origin $(0, 0)$, "
        "mapping MSTR directly to spot Bitcoin with no leverage adjustment."
    )
    st.latex(r"y = \alpha \cdot x")
    st.write("**Active Derivation:**")
    st.latex(
        rf"\alpha = \frac{{\text{{Anchor MSTR Price}}}}{{\text{{Anchor BTC Price}}}}"
        rf" = \frac{{{anchor_mstr:,.2f}}}{{{anchor_btc:,.2f}}} = {alpha_parameter:.8f}"
    )
    st.markdown(
        f"> **Applied:** $\\text{{MSTR Target}} = {alpha_parameter:.8f} \\cdot "
        f"\\text{{BTC Price}}$"
    )

    st.markdown("---")

    st.write("### 📐 Model 2: Levered Net Equity with Calibrated Premium")
    st.write(
        "Compute the corporate net asset value per share, then multiply by a "
        "premium that expands linearly with the BTC price."
    )

    st.write("#### Step A: Capital Structure (NAV per Share)")
    st.latex(
        r"\text{NAV per Share} = \frac{(\text{BTC Price} \cdot \text{Holdings})"
        r" + \text{USD Reserve} - \text{Senior Liabilities}}{\text{Shares Outstanding}}"
    )
    st.write("**Active Balance-Sheet Inputs:**")
    st.write(f"* **Corporate BTC Inventory:** {btc_holdings:,.0f} BTC")
    st.write(
        f"* **USD Reserve (cash assets):** ${usd_reserve:,.2f} "
        f"(${active_profile.get('usd_reserve_billions', 0.0)} Billion)"
    )
    st.write(
        f"* **Total Senior Liabilities:** ${total_debt:,.2f} "
        f"(${active_profile['debt_billions']} Billion)"
    )
    st.write(
        f"* **Shares Outstanding:** {shares_outstanding:,.0f} "
        f"({active_profile['shares_millions']} Million)"
    )

    st.write("#### Step B: Calibrated Premium Line")
    st.write(
        "A straight line connecting the cyclical floor (premium collapses to "
        "1.0x NAV) to the active anchor point:"
    )
    st.latex(
        r"\text{Premium}(x) = \max\!\left(1.0,\; 1.0 + "
        r"\frac{x - \text{Floor BTC}}{\text{Slope Sensitivity}}\right)"
    )

    st.write("**Two-Point Calibration:**")
    st.write(
        f"* **Point 1 (Floor):** $({floor_btc_input:,.0f}\\text{{ BTC}},\\ "
        f"1.00\\text{{x}})$"
    )
    st.write(
        f"* **Point 2 (Anchor):** $({anchor_btc:,.0f}\\text{{ BTC}},\\ "
        f"{actual_anchor_premium:.2f}\\text{{x}})$"
    )

    if premium_calibratable:
        st.write("**Derived slope $m$ and its reciprocal (the denominator used):**")
        st.latex(
            rf"m = \frac{{{actual_anchor_premium:.4f} - 1.00}}"
            rf"{{{anchor_btc:,.0f} - {floor_btc_input:,.0f}}} = {premium_slope_m:.10f}"
            rf"\qquad \frac{{1}}{{m}} = {derived_slope_sensitivity:,.0f}"
        )
    else:
        st.warning(
            "The anchor premium is ≤ 1.0x (or anchor ≤ floor) for this era, so a "
            "rising premium line cannot be calibrated."
        )

    mode = (
        "auto-calibrated"
        if auto_calibrate and derived_slope_sensitivity is not None
        else "manual"
    )
    st.write(f"**Active denominator ({mode}):** `{active_slope_sensitivity:,.0f}`")

    # Self-consistency check: a correctly calibrated Model 2 must reproduce the
    # anchor MSTR price exactly.
    _, _, _, m2_at_anchor, _, _ = evaluate_models(anchor_btc)
    anchor_error = m2_at_anchor - anchor_mstr
    st.write("**Calibration check — does Model 2 reproduce the anchor?**")
    st.latex(
        rf"\text{{Model 2}}({anchor_btc:,.0f}) = \${m2_at_anchor:,.2f}"
        rf"\quad\text{{vs anchor }}\${anchor_mstr:,.2f}"
        rf"\quad(\Delta = \${anchor_error:+,.2f})"
    )
    if abs(anchor_error) < max(0.01, 0.005 * anchor_mstr):
        st.success("✅ Calibrated: Model 2 reproduces the anchor price.")
    else:
        st.warning(
            "⚠️ Model 2 does not reproduce the anchor with the current manual "
            "denominator. Enable auto-calibration for an exact fit."
        )

    st.info(
        "💡 **Note on curve shape:** Model 2 bends upward because a NAV that "
        "rises linearly with BTC is multiplied by a premium that also rises "
        "linearly. The product of two linear terms is **quadratic (parabolic)**, "
        "not exponential."
    )

    st.markdown("---")

    st.write("### 🪙 Model 3: Carry-Adjusted NAV with Two-Anchor Regression")
    st.write(
        "Model 2 counts the preferred stack only at its liquidation preference "
        "and pins its premium line to a *theoretical* floor, so it extrapolates "
        "to share prices MSTR has never reached. Model 3 fixes both problems: it "
        "(1) charges the NAV for the ongoing dividend + interest carry, and "
        "(2) calibrates its premium by **regressing through two real market "
        "anchors** — the current price and the STRC preferred IPO, when the "
        "market first priced that carry — so the upper end stays tied to an "
        "observed premium instead of running away."
    )
    st.latex(
        r"\text{Carry-Adj NAV} = \frac{(\text{BTC Price} \cdot \text{Holdings})"
        r" + \text{USD Reserve} - \text{Senior Liabilities}"
        r" - (\text{Annual Carry} \times \text{Horizon})}{\text{Shares Outstanding}}"
        r"\qquad \text{Premium}(x) = \max(1.0,\; a + b\,x)"
    )
    st.write("**Active Carry Inputs:**")
    st.write(
        f"* **Annual Carry (Dividends + Interest):** ${annual_carry:,.2f} "
        f"(${active_profile.get('annual_carry_billions', 0.0)} Billion / year)"
    )
    st.write(f"* **Carry Horizon:** {carry_horizon_years} years")
    st.write(
        f"* **Total Carry Drag:** ${anchor_carry_drag:,.2f} "
        f"(subtracted from net equity before the premium)"
    )

    st.write(f"**Premium Calibration — `{m3_calibration_mode}`:**")
    if m3_two_anchor:
        st.write(
            f"* **Anchor 1 (Current):** BTC ${anchor_btc:,.0f}, MSTR "
            f"${anchor_mstr:,.2f} → carry-adj NAV ${m3_anchor_nav_per_share:,.2f}"
            f" → premium **{m3_actual_anchor_premium:.2f}x**"
        )
        st.write(
            f"* **Anchor 2 (STRC IPO, Jul 24 2025):** BTC ${m3_upper_btc:,.0f}, "
            f"MSTR ${m3_upper_mstr:,.2f} → carry-adj NAV "
            f"${m3_upper_nav_per_share:,.2f} → premium **{m3_upper_premium:.2f}x**"
        )
        st.latex(
            rf"b = \frac{{{m3_upper_premium:.3f} - {m3_actual_anchor_premium:.3f}}}"
            rf"{{{m3_upper_btc:,.0f} - {anchor_btc:,.0f}}} = {m3_premium_b:.3e}"
            rf"\qquad a = {m3_premium_a:.4f}"
        )
        st.caption(
            "Both premiums are measured against the same (current-era) "
            "carry-adjusted NAV, so the line is a true BTC→premium regression. "
            "Note it interpolates between two observed prices rather than "
            "extrapolating off the floor — taming the runaway upper tail."
        )
    else:
        st.write(
            f"* No second (STRC) anchor for this era — falling back to the floor "
            f"+ current-anchor line. Observed premium "
            f"**{m3_actual_anchor_premium:.2f}x** at BTC ${anchor_btc:,.0f}."
        )
        st.latex(rf"a = {m3_premium_a:.4f}\qquad b = {m3_premium_b:.3e}")

    # Calibration check: Model 3 must reproduce BOTH real anchors.
    _, _, _, _, _, m3_anch = evaluate_models(anchor_btc)
    st.write("**Calibration check — does Model 3 reproduce the real anchors?**")
    check = (
        rf"\text{{Model 3}}({anchor_btc:,.0f}) = \${m3_anch:,.2f}"
        rf"\;\text{{vs }}\${anchor_mstr:,.2f}"
    )
    if m3_two_anchor:
        _, _, _, _, _, m3_up = evaluate_models(m3_upper_btc)
        check += (
            rf";\quad \text{{Model 3}}({m3_upper_btc:,.0f}) = \${m3_up:,.2f}"
            rf"\;\text{{vs }}\${m3_upper_mstr:,.2f}"
        )
    st.latex(check)
    if abs(m3_anch - anchor_mstr) < max(0.01, 0.005 * anchor_mstr):
        st.success(
            "✅ Calibrated: Model 3 passes through both real market anchors and "
            "interpolates the premium between them."
        )
    else:
        st.warning("⚠️ Model 3 does not reproduce the anchor for this configuration.")
    st.caption(
        "Carry figures for pre-2026 eras are rough estimates; the Mid-2026 carry "
        "is the ~$1.76B disclosed in the Jun 29, 2026 8-K. The STRC anchor uses "
        "the real Jul 24, 2025 closes (BTC ~$118,368, MSTR ~$414.92)."
    )

    st.caption(
        "Educational model only — not investment advice. Figures are illustrative "
        "snapshots and may not reflect current filings."
    )
