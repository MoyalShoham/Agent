#!/usr/bin/env python3
"""
Symbol Blacklist - Temporarily reduced for training data
"""

# Temporarily allow XRPUSDT, keep ADAUSDT blocked (worst performer)
BLACKLISTED_SYMBOLS = ['ADAUSDT']  # Removed XRPUSDT temporarily

def is_symbol_allowed(symbol):
    """Check if symbol is allowed for trading"""
    return symbol not in BLACKLISTED_SYMBOLS

def get_blacklisted_symbols():
    """Get list of blacklisted symbols"""
    return BLACKLISTED_SYMBOLS.copy()

print(f"Blacklisted symbols (temporary): {BLACKLISTED_SYMBOLS}")
