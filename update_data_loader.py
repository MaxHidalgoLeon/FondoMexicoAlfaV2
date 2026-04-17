import re

with open('src/data_loader.py', 'r') as f:
    content = f.read()

# Replace the giant try/except block for fundamentals with a much simpler one that doesn't fallback to mock
fundamentals_replacement = """
    mock_fallbacks_used = []
    
    try:
        fundamentals = provider.get_fundamentals(
            [t for t in equity_tickers if t not in fibra_tickers], start_date, end_date, allow_defaults=not strict_data_mode
        )
    except Exception as e:
        logger.error("Fundamentals load failed from %s (%s). No mock fallback will be used.", source, e)
        fundamentals = pd.DataFrame(columns=["date", "ticker", "pe_ratio", "pb_ratio", "roe",
                                              "profit_margin", "net_debt_to_ebitda", "ebitda_growth", "capex_to_sales"])

    try:
        fibra_fundamentals = provider.get_fibra_fundamentals(fibra_tickers, start_date, end_date, allow_defaults=not strict_data_mode)
    except Exception as e:
        logger.error("FIBRA fundamentals load failed from %s (%s). No mock fallback will be used.", source, e)
        fibra_fundamentals = pd.DataFrame(columns=["date", "ticker", "cap_rate", "ffo_yield",
                                                    "dividend_yield", "ltv", "vacancy_rate"])

    try:
        bonds = provider.get_bonds(bond_tickers, start_date, end_date)
    except Exception as e:
        logger.error("Bond data load failed from %s (%s). No mock fallback will be used.", source, e)
        bonds = pd.DataFrame(columns=["date", "ticker", "asset_class", "price", "ytm", "duration", "credit_spread"])

    try:
        macro = provider.get_macro(start_date, end_date)
    except Exception as e:
        logger.error("Macro load failed from %s (%s). No mock fallback will be used.", source, e)
        macro = pd.DataFrame(columns=["date", "IMAI", "industrial_production_yoy", "exports_yoy",
                                      "usd_mxn", "banxico_rate", "inflation_yoy", "us_ip_yoy", "us_fed_rate"])
                                      
    if fundamentals_lag_days > 0:
        if not fundamentals.empty and "date" in fundamentals.columns:
            fundamentals["date"] = pd.to_datetime(fundamentals["date"]) + pd.Timedelta(days=fundamentals_lag_days)
        if not fibra_fundamentals.empty and "date" in fibra_fundamentals.columns:
            fibra_fundamentals["date"] = pd.to_datetime(fibra_fundamentals["date"]) + pd.Timedelta(days=fundamentals_lag_days)
            
    data_integrity = {
        "source": source,
        "strict_data_mode": strict_data_mode,
        "dropped_tickers": dropped_tickers if 'dropped_tickers' in locals() else [],
        "mock_fallbacks_used": mock_fallbacks_used,
        "fundamentals_lag_days": fundamentals_lag_days
    }

    return {
        "universe": universe,
        "prices": prices,
        "fundamentals": fundamentals,
        "fibra_fundamentals": fibra_fundamentals,
        "bonds": bonds,
        "macro": macro,
        "data_integrity": data_integrity,
    }
"""

# Find the start of the try block for fundamentals
start_idx = content.find("    try:\n        fundamentals = provider.get_fundamentals(")

if start_idx != -1:
    # Find the end of the load_data function
    end_idx = content.find("    return {\n        \"universe\": universe,", start_idx)
    end_idx = content.find("    }", end_idx) + 5
    
    new_content = content[:start_idx] + fundamentals_replacement.lstrip() + content[end_idx:]
    with open('src/data_loader.py', 'w') as f:
        f.write(new_content)
    print("Replaced content successfully.")
else:
    print("Could not find start index.")

