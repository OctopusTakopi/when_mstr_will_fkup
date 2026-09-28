# when mstr will fkup

A small Streamlit app that plays with how the **MicroStrategy (MSTR)** share
price tracks the **Bitcoin (BTC)** spot price. It compares a naive
price-to-price line against a levered net-asset-value model whose premium is
calibrated to a chosen balance-sheet era.

Educational toy, not investment advice. The numbers are illustrative snapshots
and may be out of date.

![App screenshot](docs/screenshot.png)

## Run it

```bash
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

Opens at http://localhost:8501.

## Models

- **Model 1** — a plain proportional line, `MSTR = alpha * BTC`.
- **Model 2** — levered NAV per share times a premium that's auto-calibrated to
  reproduce the era's anchor exactly.

The dashboard also charts the **premium history**: every era's anchor premium
(MSTR price over its own NAV per share) next to BTC per share, against three
reference levels — the 1.0x NAV issuance break-even, the BTC-yield break-even
when every ATM dollar buys BTC (~1.3x NAV in mid-2026), and the BTC-yield
break-even at the split each 8-K actually reports (~2.1–2.2x, since part of the
proceeds funds preferred buybacks). Coin per share only grows while the premium
clears that last line.

The "Mathematical Derivations" tab in the app walks through the full
derivation and shows a live calibration check.

## Files

- `main.py` — the whole app
- `requirements.txt` — dependencies

## License

MIT — see [LICENSE](LICENSE).
