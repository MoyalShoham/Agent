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

settings = Settings()
