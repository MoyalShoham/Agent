"""
Alpha Signal Stacker Module
Blends technical, ML, sentiment, and on-chain signals for a composite alpha score.
"""
def stack_alpha_signals(signals_dict):
    # signals_dict: {'technical': x, 'ml': y, 'sentiment': z, ...}
    score = 0
    for v in signals_dict.values():
        if v == 'buy':
            score += 1
        elif v == 'sell':
            score -= 1
    return score
