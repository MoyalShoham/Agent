import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field
load_dotenv()

class Settings(BaseSettings):
    BINANCE_API_KEY: str = Field(..., env='BINANCE_API_KEY')
    BINANCE_API_SECRET: str = Field(..., env='BINANCE_API_SECRET')
    OPENAI_API_KEY: str = Field(..., env='OPENAI_API_KEY')
    PAPER_TRADING: bool = Field(True, env='PAPER_TRADING')
    MAX_DAILY_LOSS: float = Field(1000, env='MAX_DAILY_LOSS')
    MAX_POSITION_SIZE: float = Field(0.05, env='MAX_POSITION_SIZE')
    EMERGENCY_STOP: bool = Field(False, env='EMERGENCY_STOP')
    LOG_LEVEL: str = Field('INFO', env='LOG_LEVEL')
    # Futures / leverage trading
    FUTURES_ENABLED: bool = Field(False, env='FUTURES_ENABLED')
    MAX_LEVERAGE: float = Field(1.5, env='MAX_LEVERAGE')
    RISK_PER_TRADE: float = Field(0.04, env='RISK_PER_TRADE')  # 4% default (was 40 -> unsafe)
    FUTURES_SYMBOLS: str = Field('BTCUSDT,ETHUSDT,XRPUSDT,SOLUSDT,SHIBUSDT', env='FUTURES_SYMBOLS')  # comma separated
    # Spot microcap toggle
    MICROCAP_ENABLED: bool = Field(True, env='MICROCAP_ENABLED')
    BYPASS_META: bool = Field(False, env='BYPASS_META')
    # --- New research / external data settings ---
    FUTURES_RESEARCH_ENABLED: bool = Field(True, env='FUTURES_RESEARCH_ENABLED')
    COINGECKO_API_BASE: str = Field('https://api.coingecko.com/api/v3', env='COINGECKO_API_BASE')
    FUNDING_POSITIVE_THRESHOLD: float = Field(0.0005, env='FUNDING_POSITIVE_THRESHOLD')  # 0.05%
    FUNDING_NEGATIVE_THRESHOLD: float = Field(-0.0005, env='FUNDING_NEGATIVE_THRESHOLD')  # -0.05%
    FUNDING_SIZE_REDUCTION: float = Field(0.7, env='FUNDING_SIZE_REDUCTION')  # scale if expensive to hold
    FUNDING_SIZE_BONUS: float = Field(1.15, env='FUNDING_SIZE_BONUS')  # scale if you are paid to hold
    # --- New futures metrics thresholds ---
    OPEN_INTEREST_RISE_THRESHOLD: float = Field(0.02, env='OPEN_INTEREST_RISE_THRESHOLD')  # 2% rise
    BASIS_POSITIVE_THRESHOLD: float = Field(0.002, env='BASIS_POSITIVE_THRESHOLD')  # 0.2%
    BASIS_NEGATIVE_THRESHOLD: float = Field(-0.002, env='BASIS_NEGATIVE_THRESHOLD')
    LONG_SHORT_EXTREME: float = Field(1.25, env='LONG_SHORT_EXTREME')  # >1.25 bullish / <0.8 bearish
    OI_METRICS_INTERVAL: int = Field(60, env='OI_METRICS_INTERVAL')  # seconds
    RISK_OVERLAY_REDUCTION: float = Field(0.7, env='RISK_OVERLAY_REDUCTION')
    RISK_OVERLAY_EXPANSION: float = Field(1.15, env='RISK_OVERLAY_EXPANSION')

settings = Settings()
