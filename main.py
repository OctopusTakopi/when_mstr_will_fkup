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
from plotly.subplots import make_subplots

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
# XBRL facts, CIK 0001050446); anchor prices are the year-end closes -- the last
# trading day before the date when it falls on a weekend or holiday. The
# Mid-2026 rows are interim (non-10-K) snapshots from press/secondary sources.
era_data = {
    "End of 2023 (Bear Recovery)": {
        "as_of": "2023-12-31",
        "atm_btc_fraction": None,      # weekly ATM split not tracked for the 10-K snapshots
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
        "as_of": "2024-12-31",
        "atm_btc_fraction": None,      # weekly ATM split not tracked for the 10-K snapshots
        "btc_holdings": 446400,        # 446,400 BTC as of Dec 31, 2024
        "debt_billions": 7.19,         # $7.19B long-term debt (10-K), no preferred
        "annual_carry_billions": 0.05, # est. low-coupon convert interest; no preferred yet
        "usd_reserve_billions": 0.0,   # USD Reserve not yet established
        "shares_millions": 245.78,     # 226,138,000 class A + 19,640,250 class B (10-K)
        # Corrected pair: this era previously anchored to a round-number
        # (BTC $100,000 / MSTR $380) that in fact describes mid-December 2024,
        # not the year-end closes -- worth ~0.4x of premium on the history
        # chart. Dec 31, 2024 was a trading day.
        "historical_btc_price": 93429.2,   # BTC close Dec 31, 2024 $93,429.20
        "historical_mstr_price": 289.62,   # MSTR close Dec 31, 2024 $289.62
        "floor_btc": 30000.0,
    },
    "End of 2025 (Preferred Stack)": {
        "as_of": "2025-12-31",
        "atm_btc_fraction": None,      # weekly ATM split not tracked for the 10-K snapshots
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
        # 842,138 BTC as of Aug 2, 2026 (8-K filed Aug 3, 2026); aggregate cost
        # $63.51B, average $75,419/BTC. First bitcoin sale in three weeks: 1,638
        # BTC sold Jul 27-Aug 2 @ avg $63,957 (the BTC Monetization Program,
        # announced Jun 29 with up to $1.25B capacity; proceeds funded preferred
        # dividends and STRC repurchases). This week's capital activity was
        # 3,011,361 MSTR shares sold via ATM for $290.6M net proceeds ($250.0M to
        # USD Reserve, $28.9M to STRC repurchases, $11.7M to cash) and 912,143
        # STRC shares repurchased for $81.2M; no preferred shares were sold.
        "as_of": "2026-08-02",
        "atm_btc_fraction": 0.0,       # ATM: $250.0M to USD Reserve, $28.9M to STRC, $11.7M cash -- none to BTC
        "btc_holdings": 842138,
        "debt_billions": 22.08,        # prior ~$22.17B senior claims less $91.2M STRC liquidation preference repurchased Jul 27-Aug 2
        "annual_carry_billions": 1.75, # $1.76B (8-K Jun 29) less ~$11M/yr from 912,143 STRC retired @ $12/share
        "usd_reserve_billions": 4.0,   # USD Reserve as of Aug 2, 2026 (8-K filed Aug 3, 2026); includes unsettled ATM proceeds
        "shares_millions": 368.06,     # prior ~365.05M + 3,011,361 Class A sold via ATM Jul 27-Aug 2, 2026 (8-K)
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
    "Mid 2026 (Active BTC Selling)": {
        # 840,447 BTC as of Aug 9, 2026 (8-K filed Aug 10, 2026); aggregate cost
        # $63.36B, average $75,385/BTC. Second bitcoin sale in the program: 1,690
        # BTC sold Aug 3-9 @ avg $64,262 (the BTC Monetization Program,
        # announced Jun 29 with up to $1.25B capacity; proceeds funded STRC
        # repurchases). This week's capital activity was 6,585,682 MSTR shares
        # sold via ATM for $653.1M net proceeds ($650.0M to USD Reserve,
        # $3.1M to cash) and 1,152,020 STRC shares repurchased for $108.6M;
        # no preferred shares were sold.
        "as_of": "2026-08-09",
        "atm_btc_fraction": 0.0,       # ATM: $650.0M to USD Reserve, $3.1M cash -- none to BTC
        "btc_holdings": 840447,
        "debt_billions": 21.96,        # prior ~$22.08B senior claims less $115.2M STRC liquidation preference repurchased Aug 3-9
        "annual_carry_billions": 1.74, # $1.76B (8-K Jun 29) less ~$25M/yr from 2.06M STRC retired cumulatively @ $12/share
        "usd_reserve_billions": 4.65,  # USD Reserve as of Aug 9, 2026 (8-K filed Aug 10, 2026); includes unsettled ATM proceeds
        "shares_millions": 374.65,     # prior ~368.06M + 6,585,682 Class A sold via ATM Aug 3-9, 2026 (8-K)
        "historical_btc_price": 62600.0,   # BTC ~$60-64k early Jul 2026 (sales executed $59.3k-$60.8k; spot ~$62.6-62.9k Jul 6)
        "historical_mstr_price": 97.00,    # MSTR ~$93-104 early Jul 2026 (prior close $93.39, intraday high $104.11 Jul 6) -- down sharply after the Q2 8-K's $8.32B unrealized digital-asset loss
        "floor_btc": 30000.0,
        "regression_anchor_btc": 118368.0,
        "regression_anchor_mstr": 414.92,
    },
    "Mid 2026 (BTC Sales Paused)": {
        # 840,447 BTC as of Aug 16, 2026 (8-K filed Aug 17, 2026); aggregate
        # cost $63.36B, average $75,385/BTC. First week with no bitcoin
        # purchases or sales since the BTC Monetization Program began
        # (announced Jun 29 with up to $1.25B capacity). This week's capital
        # activity was 3,458,866 MSTR shares sold via ATM for $333.7M net
        # proceeds ($149.1M to USD Reserve, $132.2M to STRC repurchases,
        # $52.4M to STRC dividends) and 1,388,720 STRC shares repurchased for
        # $132.2M; no preferred shares were sold.
        "as_of": "2026-08-16",
        "atm_btc_fraction": 0.0,       # ATM: $149.1M to USD Reserve, $132.2M STRC, $52.4M dividends -- none to BTC
        "btc_holdings": 840447,
        "debt_billions": 21.82,        # prior ~$21.96B senior claims less $138.9M STRC liquidation preference repurchased Aug 10-16
        "annual_carry_billions": 1.72, # $1.76B (8-K Jun 29) less ~$41M/yr from 3.45M STRC retired cumulatively @ $12/share
        "usd_reserve_billions": 4.80,  # USD Reserve as of Aug 16, 2026 (8-K filed Aug 17, 2026); includes unsettled ATM proceeds
        "shares_millions": 378.11,     # prior ~374.65M + 3,458,866 Class A sold via ATM Aug 10-16, 2026 (8-K)
        "historical_btc_price": 62800.0,   # BTC close Aug 16, 2026 $62,818.65
        "historical_mstr_price": 93.04,    # MSTR close Aug 14, 2026 $93.04 (last trading day before the Aug 16 snapshot)
        "floor_btc": 30000.0,
        "regression_anchor_btc": 118368.0,
        "regression_anchor_mstr": 414.92,
    },
    "Mid 2026 (USD Cash Launch)": {
        # 840,447 BTC as of Aug 23, 2026 (8-K filed Aug 24, 2026); aggregate
        # cost $63.36B, average $75,385/BTC. Second consecutive week with no
        # bitcoin purchases or sales since the BTC Monetization Program began
        # (announced Jun 29 with up to $1.25B capacity). This 8-K also
        # established "USD Cash," a new component of the Digital Credit
        # Capital Framework: a separately designated pool of USD liquidity
        # for general Bitcoin Treasury Company purposes (bitcoin purchases,
        # preferred dividends, interest, MSTR/preferred/convert repurchases,
        # USD Reserve top-ups), complementing the dividend/interest-earmarked
        # USD Reserve. This week's capital activity was 18,261,118 MSTR
        # shares sold via ATM for $2,006.5M net proceeds ($300.0M to USD
        # Reserve, $136.4M to STRC repurchases, remaining $1,570.1M to USD
        # Cash) and 1,431,212 STRC shares repurchased for $136.4M; no
        # preferred shares were sold.
        "as_of": "2026-08-23",
        "atm_btc_fraction": 0.0,       # ATM: $300.0M to USD Reserve, $136.4M STRC, $1,570.1M USD Cash -- none to BTC
        "btc_holdings": 840447,
        "debt_billions": 21.68,        # prior ~$21.82B senior claims less $143.1M STRC liquidation preference repurchased Aug 17-23
        "annual_carry_billions": 1.70, # $1.76B (8-K Jun 29) less ~$59M/yr from 4.88M STRC retired cumulatively @ $12/share
        "usd_reserve_billions": 5.10,  # USD Reserve as of Aug 23, 2026 (8-K filed Aug 24, 2026); includes unsettled ATM proceeds
        "usd_cash_billions": 1.59,     # USD Cash as of Aug 23, 2026 (8-K filed Aug 24, 2026); includes unsettled ATM proceeds
        "shares_millions": 396.37,     # prior ~378.11M + 18,261,118 Class A sold via ATM Aug 17-23, 2026 (8-K)
        "historical_btc_price": 77755.0,   # BTC close Aug 23, 2026 $77,755.27
        "historical_mstr_price": 119.25,   # MSTR close Aug 21, 2026 $119.25 (last trading day before the Aug 23 snapshot)
        "floor_btc": 30000.0,
        "regression_anchor_btc": 118368.0,
        "regression_anchor_mstr": 414.92,
    },
    "Mid 2026 (BTC Buying Resumed)": {
        # 845,050 BTC as of Aug 30, 2026 (8-K filed Aug 31, 2026); aggregate
        # cost $63.73B, average $75,412/BTC. First bitcoin purchase since the
        # BTC Monetization Program sales (announced Jun 29): 4,603 BTC bought
        # Aug 24-30 for $369.7M at an average $80,318/BTC, funded entirely by
        # MSTR ATM proceeds. This week's capital activity was 4,531,421 MSTR
        # shares sold via ATM for $602.8M net proceeds ($369.7M to bitcoin
        # purchases, $151.8M to STRC repurchases, $50.7M to STRC dividends,
        # $30.0M to USD Cash) and 1,557,177 STRC shares repurchased for
        # $151.8M; no preferred shares were sold. $364.8M remains under the
        # Digital Credit Securities Repurchase Program and $1.0B under the
        # MSTR repurchase program.
        "as_of": "2026-08-30",
        "atm_btc_fraction": 0.613,     # $369.7M of $602.8M ATM net proceeds bought BTC
        "btc_holdings": 845050,
        "debt_billions": 21.52,        # prior ~$21.68B senior claims less $155.7M STRC liquidation preference repurchased Aug 24-30
        "annual_carry_billions": 1.68, # $1.76B (8-K Jun 29) less ~$77M/yr from 6.44M STRC retired cumulatively @ $12/share
        "usd_reserve_billions": 5.10,  # USD Reserve as of Aug 30, 2026 (8-K filed Aug 31, 2026); includes unsettled ATM proceeds
        "usd_cash_billions": 1.61,     # USD Cash as of Aug 30, 2026 (8-K filed Aug 31, 2026); includes unsettled ATM proceeds
        "shares_millions": 400.90,     # prior ~396.37M + 4,531,421 Class A sold via ATM Aug 24-30, 2026 (8-K)
        "historical_btc_price": 77658.0,   # BTC close Aug 30, 2026 ~$77,658 (CoinGecko 00:00 UTC Aug 31; Yahoo daily bar unavailable)
        "historical_mstr_price": 127.31,   # MSTR close Aug 28, 2026 $127.31 (last trading day before the Aug 30 snapshot)
        "floor_btc": 30000.0,
        "regression_anchor_btc": 118368.0,
        "regression_anchor_mstr": 414.92,
    },
    "Mid 2026 (STRC Buyback Scale-Up)": {
        # 845,050 BTC as of Sep 7, 2026 (8-K filed Sep 8, 2026); aggregate
        # cost $63.73B, average $75,412/BTC. No bitcoin purchases or sales
        # and no ATM share sales during Aug 31-Sep 7. Board increased the
        # Digital Credit Securities Repurchase Program from $1.0B to $2.0B;
        # $1.19B remained available as of Sep 7. This week's capital activity
        # was 1,810,885 STRC shares repurchased for $176.3M, funded entirely
        # from USD Cash; no STRF/STRK/STRD/MSTR shares were sold or
        # repurchased. $1.0B remains under the MSTR repurchase program.
        "as_of": "2026-09-07",
        "atm_btc_fraction": None,      # no MSTR ATM sales during the week
        "btc_holdings": 845050,
        "debt_billions": 21.34,        # prior ~$21.52B senior claims less $181.1M STRC liquidation preference repurchased Aug 31-Sep 7
        "annual_carry_billions": 1.66, # $1.76B (8-K Jun 29) less ~$99M/yr from 8.25M STRC retired cumulatively @ $12/share
        "usd_reserve_billions": 5.10,  # USD Reserve as of Sep 7, 2026 (8-K filed Sep 8, 2026)
        "usd_cash_billions": 1.44,     # USD Cash as of Sep 7, 2026 (8-K filed Sep 8, 2026); $176.3M used for STRC repurchases
        "shares_millions": 400.90,     # unchanged; no MSTR ATM sales Aug 31-Sep 7, 2026 (8-K)
        "historical_btc_price": 79116.0,   # BTC close Sep 7, 2026 $79,115.85
        "historical_mstr_price": 142.80,   # MSTR close Sep 4, 2026 $142.80 (last trading day before the Sep 7 snapshot; Labor Day)
        "floor_btc": 30000.0,
        "regression_anchor_btc": 118368.0,
        "regression_anchor_mstr": 414.92,
    },
    "Mid 2026 (USD Cash Drawdown)": {
        # 845,050 BTC as of Sep 13, 2026 (8-K filed Sep 14, 2026); aggregate
        # cost $63.73B, average $75,412/BTC. Second consecutive week with no
        # bitcoin purchases or sales and no MSTR ATM share sales. This week's
        # only capital activity was 1,420,467 STRC shares repurchased for
        # $139.3M, funded entirely from USD Cash ($1.44B -> $1.30B); no
        # STRF/STRK/STRD/MSTR shares were sold or repurchased. $1.05B remains
        # under the Digital Credit Securities Repurchase Program (down from
        # $1.19B) and $1.0B under the MSTR repurchase program.
        "as_of": "2026-09-13",
        "atm_btc_fraction": None,      # no MSTR ATM sales during the week
        "btc_holdings": 845050,
        "debt_billions": 21.20,        # prior ~$21.34B senior claims less $142.0M STRC liquidation preference repurchased Sep 8-13
        "annual_carry_billions": 1.64, # $1.76B (8-K Jun 29) less ~$116M/yr from 9.67M STRC retired cumulatively @ $12/share
        "usd_reserve_billions": 5.10,  # USD Reserve as of Sep 13, 2026 (8-K filed Sep 14, 2026)
        "usd_cash_billions": 1.30,     # USD Cash as of Sep 13, 2026 (8-K filed Sep 14, 2026); $139.3M used for STRC repurchases
        "shares_millions": 400.90,     # unchanged; no MSTR ATM sales Sep 8-13, 2026 (8-K)
        "historical_btc_price": 76838.0,   # BTC close Sep 13, 2026 $76,838.16
        "historical_mstr_price": 130.97,   # MSTR close Sep 11, 2026 $130.97 (last trading day before the Sep 13 snapshot)
        "floor_btc": 30000.0,
        "regression_anchor_btc": 118368.0,
        "regression_anchor_mstr": 414.92,
    },
    "Mid 2026 (USD Cash Deployed)": {
        # 846,000 BTC as of Sep 20, 2026 (8-K filed Sep 21, 2026); aggregate
        # cost $63.80B, average $75,416/BTC. Third consecutive week with no
        # MSTR ATM sales; the week's only bitcoin activity was a 950 BTC
        # purchase for $75.7M at an average $79,670/BTC, funded entirely from
        # USD Cash -- the first BTC buy drawn from that pool rather than from
        # ATM proceeds. Capital activity was 1,771,238 STRC shares repurchased
        # for $174.0M (USD Cash); no STRF/STRK/STRD/MSTR shares were sold or
        # repurchased. $174.0M of USD Cash went to STRC buybacks and $75.7M to
        # bitcoin, while $57.4M of the USD Reserve funded preferred dividends
        # and debt interest. $875.1M remains under the Digital Credit
        # Securities Repurchase Program and $1.0B under the MSTR repurchase
        # program.
        "as_of": "2026-09-20",
        "atm_btc_fraction": None,      # no MSTR ATM sales during the week
        "btc_holdings": 846000,
        "debt_billions": 21.02,        # prior ~$21.20B senior claims less $177.1M STRC liquidation preference repurchased Sep 14-20
        "annual_carry_billions": 1.62, # $1.76B (8-K Jun 29) less ~$137M/yr from 11.44M STRC retired cumulatively @ $12/share
        "usd_reserve_billions": 5.04,  # USD Reserve as of Sep 20, 2026 (8-K filed Sep 21, 2026); $57.4M used for dividends/interest
        "usd_cash_billions": 1.05,     # USD Cash as of Sep 20, 2026 (8-K filed Sep 21, 2026); $174.0M STRC + $75.7M BTC deployed
        "shares_millions": 400.90,     # unchanged; no MSTR ATM sales Sep 14-20, 2026 (8-K)
        "historical_btc_price": 81143.0,    # BTC close Sep 20, 2026 $81,142.61
        "historical_mstr_price": 153.92,    # MSTR close Sep 18, 2026 $153.92 (last trading day before the Sep 20 snapshot)
        "floor_btc": 30000.0,
        "regression_anchor_btc": 118368.0,
        "regression_anchor_mstr": 414.92,
    },
    "Mid 2026 (ATM Reopened)": {
        # 847,666 BTC as of Sep 27, 2026 (8-K filed Sep 28, 2026); aggregate
        # cost $63.95B, average $75,437/BTC. Bitcoin purchases continued off
        # the prior week's USD-Cash-funded buy: 1,665 BTC bought Sep 21-27
        # for $142.7M at an average $85,681/BTC -- the highest purchase price
        # of the cycle through Sep 27 -- funded entirely by MSTR ATM proceeds.
        # ATM sales restarted after three idle weeks: 1,469,165 Class A shares
        # for $246.2M net proceeds (~$167.6/share, far above the Sep 18 close of
        # $153.92). Proceeds were split $142.7M to bitcoin and $103.5M to
        # STRC, with a further $48.1M of USD Cash also funding STRC: the
        # week's 1,534,530 STRC shares repurchased for $151.7M cost ~$98.86,
        # a discount to the $100 liquidation preference. $22.1M of the USD
        # Reserve funded preferred dividends. $723.5M remains under the
        # Digital Credit Securities Repurchase Program and $1.0B under the
        # MSTR repurchase program.
        # NOTE: issuing at ~$167.6/share to buy BTC at $85,681 is NAV-accretive
        # but BTC-per-share dilutive: 2,110 -> 2,107 BTC per million shares
        # (~-0.17%), because the 1.12x premium sits far below the ~2.18x
        # BTC-yield break-even once 42% of the proceeds fund STRC buybacks.
        # The issuance flywheel is stalled at these prices -- coin per share
        # only grows above that break-even -- and the NAV math is dominated by
        # the dividend drag rather than by issuance.
        "as_of": "2026-09-27",
        "atm_btc_fraction": 0.580,     # $142.7M of $246.2M ATM net proceeds bought BTC
        "btc_holdings": 847666,
        "debt_billions": 20.87,        # prior ~$21.02B senior claims less $153.5M STRC liquidation preference repurchased Sep 21-27
        "annual_carry_billions": 1.60, # $1.76B (8-K Jun 29) less ~$156M/yr from 12.97M STRC retired cumulatively @ $12/share
        "usd_reserve_billions": 5.02,  # USD Reserve as of Sep 27, 2026 (8-K filed Sep 28, 2026); $22.1M used for preferred dividends
        "usd_cash_billions": 1.00,     # USD Cash as of Sep 27, 2026 (8-K filed Sep 28, 2026); $48.1M used for STRC repurchases
        "shares_millions": 402.37,     # prior ~400.90M + 1,469,165 Class A sold via ATM Sep 21-27, 2026 (8-K)
        "historical_btc_price": 84458.0,    # BTC close Sep 27, 2026 $84,458.09
        "historical_mstr_price": 158.61,    # MSTR close Sep 25, 2026 $158.61 (last trading day before the Sep 27 snapshot)
        "floor_btc": 30000.0,
        "regression_anchor_btc": 118368.0,
        "regression_anchor_mstr": 414.92,
    },
    "Mid 2026 (Issuance Trickle)": {
        # 848,000 BTC as of Oct 4, 2026 (8-K filed Oct 5, 2026); aggregate
        # cost $63.97B, average $75,441/BTC. All of the week's (Sep 28-Oct 4)
        # activity fell in the Oct 1-4 stub: a 334 BTC purchase for $28.7M at
        # an average $85,839/BTC -- the smallest buy of the cycle -- funded
        # $15.7M from MSTR ATM proceeds and $13.0M from USD Cash. ATM sales
        # shrank to a trickle: 92,894 Class A shares for $15.7M net proceeds,
        # all deployed into bitcoin (f = 1.0), and no preferred stock was
        # issued. Capital activity was entirely buybacks again: 1,773,802
        # STRC shares repurchased for $176.3M ($154.1M from USD Cash plus
        # $22.2M of interest earned), draining USD Cash to $833.4M, while
        # $142.5M of the USD Reserve funded preferred dividends and debt
        # interest ($5.02B -> $4.88B). $547.2M remains under the Digital
        # Credit Securities Repurchase Program and $1.0B under the MSTR
        # repurchase program. The Q3 update in the same filing reports a
        # $20.91B digital-asset gain, a $70.82B carrying value and a $1.88B
        # net deferred tax liability.
        # NOTE: this is the first week whose ATM proceeds went 100% to BTC
        # (f = 1.0), pulling the coin-per-share break-even down to ~1.26x.
        # The raise itself was still coin-dilutive: shares went out at ~$169
        # (~1.17x the anchor NAV, below that break-even), so the ~183
        # ATM-funded BTC added fewer coins per share than the 92,894 new
        # shares consumed (~-0.002%). Coin per share rose 2,106.7 -> 2,107.0
        # (+0.02%) only because the other ~151 BTC came from the USD Cash
        # pool -- cash buys add coin without issuing shares, and the f-based
        # break-even does not cover them. The dividend and buyback drain
        # (~$319M this week) still dominates the balance-sheet flow, not the
        # issuance flywheel.
        "as_of": "2026-10-04",
        "atm_btc_fraction": 1.0,       # all $15.7M of ATM net proceeds (92,894 MSTR shares sold Oct 1-4) bought BTC
        "btc_holdings": 848000,
        "debt_billions": 20.69,        # prior ~$20.87B senior claims less $177.4M STRC liquidation preference repurchased Sep 28-Oct 4
        "annual_carry_billions": 1.58, # $1.76B (8-K Jun 29) less ~$177M/yr from 14.74M STRC retired cumulatively @ $12/share
        "usd_reserve_billions": 4.88,  # USD Reserve as of Oct 4, 2026 (8-K filed Oct 5, 2026); $142.5M used for preferred dividends and debt interest
        "usd_cash_billions": 0.83,     # USD Cash as of Oct 4, 2026 (8-K filed Oct 5, 2026); $154.1M STRC buybacks + $13.0M BTC
        "shares_millions": 402.46,     # prior ~402.37M + 92,894 Class A sold via ATM Oct 1-4, 2026 (8-K)
        "historical_btc_price": 86480.0,    # BTC close Oct 4, 2026 $86,480.30
        "historical_mstr_price": 160.01,    # MSTR close Oct 2, 2026 $160.01 (last trading day before the Oct 4 snapshot)
        "floor_btc": 30000.0,
        "regression_anchor_btc": 118368.0,
        "regression_anchor_mstr": 414.92,
    },
    "Mid 2026 (Largest STRC Week)": {
        # 848,250 BTC as of Oct 11, 2026 (8-K filed Oct 12, 2026); aggregate
        # cost $64.00B, average $75,444/BTC. Smallest buy of the cycle: 250
        # BTC for $21.5M at an average $86,194/BTC, funded $21.0M from MSTR
        # ATM proceeds and $0.5M from USD Cash. ATM sales stayed a trickle:
        # 125,183 Class A shares for $21.0M net proceeds (~$167.8/share), all
        # deployed into bitcoin (f = 1.0); no preferred stock was issued. The
        # week's capital activity was the buyback: 1,932,145 STRC shares
        # repurchased for $192.4M (~$99.58, a discount to the $100 liquidation
        # preference), the largest weekly STRC cash outlay of the program,
        # funded entirely from USD Cash ($833.4M -> $640.5M). The USD Reserve
        # was unchanged at $4.88B -- no dividend or interest draw this week.
        # $354.8M remains under the Digital Credit Securities Repurchase
        # Program (down from $547.2M by the $192.4M spent) and $1.0B under
        # the MSTR repurchase program.
        # NOTE: second week in a row with ATM proceeds 100% to BTC (f = 1.0),
        # so the coin-per-share break-even stays on the all-proceeds line
        # (~1.27x at this anchor). The raise was still coin-dilutive: shares
        # went out at ~$167.8 (~1.21x the anchor NAV, below that break-even),
        # and the ~244 ATM-funded BTC did not cover the 125,183 new shares.
        # The ~6 BTC bought with USD Cash were too few to offset that, so
        # coin per share is flat at ~2,107 (~-0.003%). The $192.4M STRC
        # repurchase, not the issuance, is the week's balance-sheet flow.
        "as_of": "2026-10-11",
        "atm_btc_fraction": 1.0,       # all $21.0M of ATM net proceeds (125,183 MSTR shares sold Oct 5-11) bought BTC
        "btc_holdings": 848250,
        "debt_billions": 20.50,        # prior ~$20.69B senior claims less $193.2M STRC liquidation preference repurchased Oct 5-11
        "annual_carry_billions": 1.56, # $1.76B (8-K Jun 29) less ~$200M/yr from 16.68M STRC retired cumulatively @ $12/share
        "usd_reserve_billions": 4.88,  # USD Reserve as of Oct 11, 2026 (8-K filed Oct 12, 2026); unchanged, no dividend/interest draw
        "usd_cash_billions": 0.64,     # USD Cash as of Oct 11, 2026 (8-K filed Oct 12, 2026); $640.5M after $192.4M STRC + $0.5M BTC
        "shares_millions": 402.59,     # prior ~402.46M + 125,183 Class A sold via ATM Oct 5-11, 2026 (8-K)
        "historical_btc_price": 83679.0,    # BTC close Oct 11, 2026 $83,679.47
        "historical_mstr_price": 154.34,    # MSTR close Oct 9, 2026 $154.34 (last trading day before the Oct 11 snapshot)
        "floor_btc": 30000.0,
        "regression_anchor_btc": 118368.0,
        "regression_anchor_mstr": 414.92,
    },
    "Mid 2026 (Buyback Program Upsized)": {
        # 848,517 BTC as of Oct 18, 2026 (8-K filed Oct 19, 2026); aggregate
        # cost $64.02B, average $75,446/BTC. Third consecutive ATM-funded BTC
        # buy: 267 BTC for $22.1M at an average $82,900/BTC, funded $22.0M
        # from MSTR ATM proceeds and $0.1M from USD Cash. ATM sales stayed a
        # trickle: 140,872 Class A shares for $22.0M net proceeds
        # (~$156.17/share), all deployed into bitcoin (f = 1.0); no preferred
        # stock was issued. The week's dominant capital flow was the buyback:
        # 1,938,780 STRC shares repurchased for $193.1M (~$99.60, a discount
        # to the $100 liquidation preference), funded entirely from USD Cash
        # ($640.5M -> $447.3M). The board raised the Digital Credit Securities
        # Repurchase Program from $2.0B to $3.0B; $1.16B remained available
        # after this week's purchases, with $1.0B under the MSTR repurchase
        # program. $45.5M of the USD Reserve funded preferred dividends
        # ($4.88B -> $4.83B).
        # NOTE: at the Oct 16 close, MSTR traded at $136.83, only ~1.01x the
        # ~$135.45 implied NAV/share -- down from 1.11x a week earlier and
        # essentially at the 1.0x issuance break-even. The $156.17 ATM price
        # was ~1.15x NAV, below the ~1.28x all-proceeds BTC-yield break-even.
        # The 267 BTC bought this week only just covered the 0.035% share
        # increase (coin per share ~2,106.98 -> 2,106.91); the larger
        # balance-sheet flow remains the $193.1M STRC repurchase.
        "as_of": "2026-10-18",
        "atm_btc_fraction": 1.0,       # all $22.0M of ATM net proceeds (140,872 MSTR shares sold Oct 12-18) bought BTC
        "btc_holdings": 848517,
        "debt_billions": 20.31,        # prior ~$20.50B senior claims less $193.9M STRC liquidation preference repurchased Oct 12-18
        "annual_carry_billions": 1.54, # $1.76B (8-K Jun 29) less ~$223M/yr from 18.62M STRC retired cumulatively @ $12/share
        "usd_reserve_billions": 4.83,  # USD Reserve as of Oct 18, 2026 (8-K filed Oct 19, 2026); $45.5M used for preferred dividends
        "usd_cash_billions": 0.4473,   # USD Cash as of Oct 18, 2026 (8-K filed Oct 19, 2026); $193.1M STRC + $0.1M BTC
        "shares_millions": 402.73,     # prior ~402.59M + 140,872 Class A sold via ATM Oct 12-18, 2026 (8-K)
        "historical_btc_price": 82001.58,   # BTC close Oct 18, 2026 $82,001.58 (Coinbase/Bitstamp)
        "historical_mstr_price": 136.83,    # MSTR close Oct 16, 2026 $136.83 (last trading day before the Oct 18 snapshot)
        "floor_btc": 30000.0,
        "regression_anchor_btc": 118368.0,
        "regression_anchor_mstr": 414.92,
    },
}

# ---------------------------------------------------------
# 2. GLOBAL SIDEBAR CONTROLS & LIVE FETCH
# ---------------------------------------------------------
st.sidebar.header("1. Select Target Era Configuration")
selected_era = st.sidebar.selectbox(
    "Active Balance Sheet Profile", list(era_data.keys()), index=14
)

# Load data state based on selection.
active_profile = era_data[selected_era]
btc_holdings = active_profile["btc_holdings"]
total_debt = active_profile["debt_billions"] * 1_000_000_000
usd_reserve = active_profile.get("usd_reserve_billions", 0.0) * 1_000_000_000
usd_cash = active_profile.get("usd_cash_billions", 0.0) * 1_000_000_000
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
anchor_gross_assets = anchor_btc * btc_holdings + usd_reserve + usd_cash
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
    g = btc_price * btc_holdings + usd_reserve + usd_cash
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
    gross_assets = target_price_btc * btc_holdings + usd_reserve + usd_cash
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

    # ---------------------------------------------------------
    # PREMIUM HISTORY: what the market paid over NAV, era by era
    # ---------------------------------------------------------
    st.markdown("---")
    st.write("### Premium History: What the Market Paid Over NAV")
    st.write(
        "Each point is one era's anchor observation -- the anchor MSTR price "
        "divided by that era's **own** NAV per share, so the series stays "
        "comparable across different balance sheets. The BTC-per-share line is "
        "the issuance accretion test: ATM-funded coin only grows while the "
        "observed premium clears the BTC-yield break-even; coin bought with "
        "USD Cash adds without issuing shares."
    )

    # One observation per era: premium = anchor MSTR / that era's NAV/share.
    #
    # BTC-YIELD BREAK-EVEN -- the premium at which issuing stock leaves BTC per
    # share unchanged. Issuing at m times NAV/share adds
    # (m * NAV/share * f) / BTC_price coin per new share, while the shares
    # already outstanding carry (holdings / shares) coin each, so BTC/share is
    # flat at
    #     m = (BTC value per share / NAV per share) / f
    # where f is the share of the week's ATM proceeds actually spent on BTC.
    # f = 0 (proceeds to reserves, preferred or dividends) has no finite
    # break-even, and weeks with no issuance have none either; both plot as
    # gaps.
    hist_rows = []
    for era_name, prof in era_data.items():
        anchor_btc_i = prof["historical_btc_price"]
        shares_i = prof["shares_millions"] * 1_000_000
        nav_i = (
            anchor_btc_i * prof["btc_holdings"]
            + prof.get("usd_reserve_billions", 0.0) * 1_000_000_000
            + prof.get("usd_cash_billions", 0.0) * 1_000_000_000
            - prof["debt_billions"] * 1_000_000_000
        ) / shares_i
        btc_backing_multiple = anchor_btc_i * prof["btc_holdings"] / shares_i / nav_i
        atm_btc_fraction = prof.get("atm_btc_fraction") or 0.0
        hist_rows.append(
            {
                "Era": era_name,
                "As of": pd.Timestamp(prof["as_of"]),
                "Premium": prof["historical_mstr_price"] / nav_i,
                "NAV / Share": nav_i,
                "Anchor MSTR": prof["historical_mstr_price"],
                "BTC per 1M Shares": prof["btc_holdings"] / prof["shares_millions"],
                "Break-even (f=1)": btc_backing_multiple,
                "Break-even (actual f)": (
                    btc_backing_multiple / atm_btc_fraction
                    if atm_btc_fraction > 0
                    else float("nan")
                ),
            }
        )
    premium_history = pd.DataFrame(hist_rows).sort_values("As of")

    fig_prem = make_subplots(specs=[[{"secondary_y": True}]])
    fig_prem.add_trace(
        go.Scatter(
            x=premium_history["As of"],
            y=premium_history["Premium"],
            mode="lines+markers",
            name="Observed premium (mNAV)",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=8),
            text=premium_history["Era"],
            customdata=np.stack(
                [premium_history["NAV / Share"], premium_history["Anchor MSTR"]],
                axis=-1,
            ),
            hovertemplate=(
                "%{text}<br>%{x|%Y-%m-%d} -- premium %{y:.2f}x"
                "<br>NAV/sh $%{customdata[0]:,.2f} -- anchor MSTR "
                "$%{customdata[1]:,.2f}<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    fig_prem.add_trace(
        go.Scatter(
            x=premium_history["As of"],
            y=premium_history["Break-even (f=1)"],
            mode="lines+markers",
            name="BTC-yield break-even (all proceeds to BTC)",
            line=dict(color="#9467bd", width=1.5, dash="dot"),
            marker=dict(size=5),
            hovertemplate="all-proceeds break-even %{y:.2f}x<extra></extra>",
        ),
        secondary_y=False,
    )
    fig_prem.add_trace(
        go.Scatter(
            x=premium_history["As of"],
            y=premium_history["Break-even (actual f)"],
            mode="markers",
            name="BTC-yield break-even (actual deployment)",
            marker=dict(size=9, symbol="diamond", color="#d62728"),
            hovertemplate="break-even at the 8-K's own split %{y:.2f}x<extra></extra>",
        ),
        secondary_y=False,
    )
    fig_prem.add_trace(
        go.Scatter(
            x=premium_history["As of"],
            y=premium_history["BTC per 1M Shares"],
            mode="lines+markers",
            name="BTC per 1M shares (right)",
            line=dict(color="#ff7300", width=1.5, dash="dot"),
            marker=dict(size=6, symbol="diamond"),
        ),
        secondary_y=True,
    )
    fig_prem.add_hline(
        y=1.0,
        line=dict(color="#2ca02c", dash="dash", width=1.5),
        annotation_text="1.0x NAV -- issuance break-even (ATM adds NAV only above this)",
        annotation_position="bottom right",
    )

    # Flag the era currently loaded in the sidebar.
    active_hist = premium_history[premium_history["Era"] == selected_era].iloc[0]
    fig_prem.add_trace(
        go.Scatter(
            x=[active_hist["As of"]],
            y=[active_hist["Premium"]],
            mode="markers+text",
            text=[f"selected era: {active_hist['Premium']:.2f}x"],
            textposition="top center",
            marker=dict(size=16, color="red", symbol="circle-open", line=dict(width=3)),
            name="Selected era",
        ),
        secondary_y=False,
    )

    fig_prem.update_layout(
        height=470,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig_prem.update_xaxes(title_text="Era snapshot date (8-K as-of)")
    fig_prem.update_yaxes(
        title_text="Premium over NAV (mNAV, x)", secondary_y=False, tickformat=".2f"
    )
    fig_prem.update_yaxes(
        title_text="BTC per 1M shares", secondary_y=True, tickformat=",.0f"
    )
    st.plotly_chart(fig_prem, width="stretch")
    st.caption(
        "Break-even = (BTC value per share / NAV per share) / f, with f the "
        "share of that week's ATM proceeds spent on BTC. Weeks with no ATM "
        "sales (Sep 7-20) or whose proceeds went entirely to reserves, "
        "preferred or dividends (Aug 2-23, f = 0) have no finite break-even, so "
        "the red series only carries the weeks whose 8-K documents a BTC cut: "
        "61.3% on Aug 30 and 58.0% on Sep 27, plus 100% on Oct 4, Oct 11 and "
        "Oct 18 -- three small raises ($15.7M, $21.0M, then $22.0M) whose ATM "
        "proceeds went entirely into BTC, so those diamonds sit on the "
        "all-proceeds line. Each point is dated by its balance sheet and "
        "priced at the close on, or the last trading day before, that date -- "
        "except the two early-August snapshots, which still carry a stale "
        "early-July anchor (BTC $62.6k / MSTR $97). Every premium uses its "
        "era's own holdings, cash and claims."
    )

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
        r" + \text{USD Reserve} + \text{USD Cash} - \text{Senior Liabilities}}{\text{Shares Outstanding}}"
    )
    st.write("**Active Balance-Sheet Inputs:**")
    st.write(f"* **Corporate BTC Inventory:** {btc_holdings:,.0f} BTC")
    st.write(
        f"* **USD Reserve (cash assets):** ${usd_reserve:,.2f} "
        f"(${active_profile.get('usd_reserve_billions', 0.0)} Billion)"
    )
    st.write(
        f"* **USD Cash (liquidity pool):** ${usd_cash:,.2f} "
        f"(${active_profile.get('usd_cash_billions', 0.0)} Billion)"
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
        "Carry figures for pre-2026 eras are rough estimates; Mid-2026 starts from "
        "the ~$1.76B disclosed in the Jun 29, 2026 8-K and is stepped down as STRC "
        "is retired (~$12/share/yr at the 12% stated rate). The STRC anchor uses "
        "the real Jul 24, 2025 closes (BTC ~$118,368, MSTR ~$414.92)."
    )

    st.caption(
        "Educational model only — not investment advice. Figures are illustrative "
        "snapshots and may not reflect current filings."
    )
