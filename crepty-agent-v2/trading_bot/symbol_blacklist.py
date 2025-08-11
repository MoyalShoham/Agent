#!/usr/bin/env python3
"""
Symbol Blacklist - Temporarily disable worst performing symbols
"""

# Symbols that are currently losing money - temporarily disabled
BLACKLISTED_SYMBOLS = ['ADAUSDT', 'XRPUSDT']

def is_symbol_allowed(symbol):
    """Check if symbol is allowed for trading"""
    return symbol not in BLACKLISTED_SYMBOLS

def get_blacklisted_symbols():
    """Get list of blacklisted symbols"""
    return BLACKLISTED_SYMBOLS.copy()

print(f"Blacklisted symbols: {BLACKLISTED_SYMBOLS}")
