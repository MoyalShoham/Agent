#!/usr/bin/env python3
"""
Symbol Analysis Script for Trading Bot Research
Analyzes current market conditions and provides symbol recommendations
"""

from trading_bot.utils.binance_client import BinanceClient
import pandas as pd
from datetime import datetime

def analyze_symbols():
    try:
        binance = BinanceClient()
        tickers = binance.client.get_ticker()
        
        # Filter and sort by volume
        usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt_pairs.sort(key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
        
        print('=== TOP 20 HIGHEST VOLUME USDT PAIRS ===')
        print(f"{'Rank':<4} {'Symbol':<15} {'Volume (24h)':<15} {'Change %':<10} {'Price':<12}")
        print("-" * 70)
        
        for i, ticker in enumerate(usdt_pairs[:20]):
            symbol = ticker['symbol']
            volume = float(ticker.get('quoteVolume', 0))
            change = float(ticker.get('priceChangePercent', 0))
            price = float(ticker.get('lastPrice', 0))
            
            print(f"{i+1:<4} {symbol:<15} ${volume:>13,.0f} {change:>+8.2f}% ${price:>10.6f}")
        
        print('\n=== CURRENT TRADING SYMBOLS ANALYSIS ===')
        current_symbols = ['BTCUSDT','ETHUSDT','XRPUSDT','SOLUSDT','ADAUSDT','BNBUSDT','TRXUSDT','ARBUSDT','OPUSDT','AVAXUSDT','DOGEUSDT','LTCUSDT','SPELLUSDT']
        
        print(f"{'Symbol':<12} {'Volume Rank':<12} {'24h Volume':<15} {'Change %':<10} {'Price':<12}")
        print("-" * 70)
        
        for symbol in current_symbols:
            ticker_data = next((t for t in tickers if t['symbol'] == symbol), None)
            if ticker_data:
                volume = float(ticker_data.get('quoteVolume', 0))
                change = float(ticker_data.get('priceChangePercent', 0))
                price = float(ticker_data.get('lastPrice', 0))
                rank = next((i+1 for i, t in enumerate(usdt_pairs) if t['symbol'] == symbol), 999)
                print(f"{symbol:<12} #{rank:<11} ${volume:>13,.0f} {change:>+8.2f}% ${price:>10.6f}")
            else:
                print(f"{symbol:<12} NOT FOUND")
        
        print('\n=== SYMBOL RECOMMENDATIONS BASED ON PROVIDED DATA ===')
        
        # Analyze the provided symbol data for recommendations
        provided_symbols = [
            # High volume, established coins
            ('BTCUSDT', 'Already included - Top crypto by market cap'),
            ('ETHUSDT', 'Already included - Top altcoin'), 
            ('BCHUSDT', 'Bitcoin Cash - Major fork with good liquidity'),
            ('LTCUSDT', 'Already included - Silver to Bitcoin gold'),
            ('LINKUSDT', 'Chainlink - Major DeFi oracle provider'),
            ('UNIUSDT', 'Uniswap - Leading DEX token'),
            ('AAVEUSDT', 'Aave - Major DeFi lending protocol'),
            ('MATICUSDT', 'Polygon - Layer 2 scaling solution'),
            
            # High volume newer coins
            ('NEARUSDT', 'NEAR Protocol - High-performance blockchain'),
            ('FILUSDT', 'Filecoin - Decentralized storage leader'),
            ('APTUSDT', 'Aptos - High-performance Layer 1'),
            ('SUIUSDT', 'Sui - Fast Layer 1 blockchain'),
            ('INJUSDT', 'Injective - DeFi-focused blockchain'),
            
            # Meme/retail favorites with high volume
            ('1000PEPEUSDT', 'PEPE - Popular meme coin with high volume'),
            ('1000FLOKIUSDT', 'FLOKI - Established meme coin'),
            ('1000SHIBUSDT', 'SHIB - Major meme coin ecosystem'),
            
            # AI/Gaming tokens
            ('FETUSDT', 'Fetch.ai - AI and machine learning'),
            ('RENDERUSDT', 'Render - GPU rendering network'),
            ('SANDUSDT', 'Sandbox - Metaverse and gaming'),
            ('MANAUSDT', 'Decentraland - Virtual world platform'),
        ]
        
        print("\nRECOMMENDED ADDITIONS (in priority order):")
        print("=" * 50)
        
        priority_adds = [
            'LINKUSDT', 'UNIUSDT', 'AAVEUSDT', 'NEARUSDT', 'FILUSDT',
            'APTUSDT', 'SUIUSDT', 'RENDERUSDT', 'FETUSDT', '1000PEPEUSDT'
        ]
        
        for i, symbol in enumerate(priority_adds):
            ticker_data = next((t for t in tickers if t['symbol'] == symbol), None)
            if ticker_data:
                volume = float(ticker_data.get('quoteVolume', 0))
                change = float(ticker_data.get('priceChangePercent', 0))
                rank = next((i+1 for i, t in enumerate(usdt_pairs) if t['symbol'] == symbol), 999)
                print(f"{i+1:2d}. {symbol:<15} - Rank #{rank:<3d} - Vol: ${volume:>12,.0f} - Change: {change:>+6.2f}%")
        
        print(f"\n=== SUMMARY & RECOMMENDATIONS ===")
        print(f"Current symbols: {len(current_symbols)}")
        print(f"Recommended additions: {len(priority_adds)}")
        print(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return priority_adds
        
    except Exception as e:
        print(f'Error in analysis: {e}')
        return []

if __name__ == "__main__":
    recommendations = analyze_symbols()
