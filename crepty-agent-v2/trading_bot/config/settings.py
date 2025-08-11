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

settings = Settings()
