"""
AI Agents System - Specialized AI agents for different trading functions
Uses OpenAI API with agentic method and structured outputs
"""

import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import asyncio
import json
from datetime import datetime, timedelta

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import BaseTool
from loguru import logger

# Pydantic Models for Structured Outputs

class MarketRegime(str, Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class ActionType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    REDUCE = "reduce"
    INCREASE = "increase"

# Financial Analysis Agent Response
class FinancialAnalysisResponse(BaseModel):
    analysis_summary: str = Field(description="Comprehensive market analysis summary")
    market_regime: MarketRegime = Field(description="Current market regime classification")
    key_levels: Dict[str, float] = Field(description="Important support/resistance levels")
    risk_factors: List[str] = Field(description="Identified risk factors")
    opportunities: List[str] = Field(description="Trading opportunities")
    confidence_score: float = Field(description="Analysis confidence (0-1)", ge=0, le=1)
    timeframe: str = Field(description="Analysis timeframe")
    next_update_time: datetime = Field(description="When to update analysis")

# Broker Agent Response
class BrokerResponse(BaseModel):
    action: ActionType = Field(description="Recommended trading action")
    symbol: str = Field(description="Trading symbol")
    quantity: float = Field(description="Position size/quantity")
    entry_price: Optional[float] = Field(description="Suggested entry price")
    stop_loss: Optional[float] = Field(description="Stop loss level")
    take_profit: Optional[float] = Field(description="Take profit level")
    reasoning: str = Field(description="Reasoning behind the decision")
    urgency: str = Field(description="Trade urgency (low/medium/high)")
    risk_level: RiskLevel = Field(description="Risk assessment")
    execution_notes: List[str] = Field(description="Special execution instructions")

# Risk Management Agent Response
class RiskManagementResponse(BaseModel):
    overall_risk_level: RiskLevel = Field(description="Overall portfolio risk level")
    position_recommendations: Dict[str, Dict] = Field(description="Per-position risk adjustments")
    portfolio_adjustments: List[str] = Field(description="Portfolio-level adjustments needed")
    risk_alerts: List[str] = Field(description="Active risk alerts")
    max_position_size: Dict[str, float] = Field(description="Maximum position sizes by symbol")
    stop_loss_adjustments: Dict[str, float] = Field(description="Stop loss adjustments")
    emergency_actions: List[str] = Field(description="Emergency actions if needed")
    confidence: float = Field(description="Risk assessment confidence", ge=0, le=1)

# Market Sentiment Agent Response
class MarketSentimentResponse(BaseModel):
    sentiment_score: float = Field(description="Overall sentiment (-1 to 1)", ge=-1, le=1)
    sentiment_trend: str = Field(description="Sentiment trend direction")
    key_sentiment_drivers: List[str] = Field(description="Main factors driving sentiment")
    social_signals: Dict[str, Any] = Field(description="Social media sentiment signals")
    news_impact: Dict[str, str] = Field(description="Impact of recent news")
    fear_greed_index: float = Field(description="Fear & Greed index (0-100)", ge=0, le=100)
    contrarian_signals: List[str] = Field(description="Contrarian opportunity signals")
    sentiment_reliability: float = Field(description="Reliability of sentiment data", ge=0, le=1)

# Portfolio Optimization Agent Response
class PortfolioOptimizationResponse(BaseModel):
    recommended_allocation: Dict[str, float] = Field(description="Recommended portfolio allocation")
    rebalancing_actions: List[Dict] = Field(description="Specific rebalancing actions")
    diversification_score: float = Field(description="Portfolio diversification score", ge=0, le=1)
    correlation_warnings: List[str] = Field(description="High correlation warnings")
    optimization_objective: str = Field(description="Current optimization objective")
    expected_return: float = Field(description="Expected portfolio return")
    expected_volatility: float = Field(description="Expected portfolio volatility")
    sharpe_ratio: float = Field(description="Expected Sharpe ratio")

# Accountant Expert Response
class AccountantResponse(BaseModel):
    report_period: str = Field(description="Report period (monthly/yearly)")
    total_pnl: float = Field(description="Total profit/loss for the period")
    realized_gains: float = Field(description="Realized capital gains")
    realized_losses: float = Field(description="Realized capital losses")
    unrealized_gains: float = Field(description="Unrealized capital gains")
    unrealized_losses: float = Field(description="Unrealized capital losses")
    total_fees: float = Field(description="Total trading fees paid")
    total_volume: float = Field(description="Total trading volume")
    trade_count: int = Field(description="Number of trades executed")
    winning_trades: int = Field(description="Number of profitable trades")
    losing_trades: int = Field(description="Number of losing trades")
    win_rate: float = Field(description="Win rate percentage", ge=0, le=100)
    average_trade_pnl: float = Field(description="Average P&L per trade")
    best_trade: Dict[str, Any] = Field(description="Best performing trade")
    worst_trade: Dict[str, Any] = Field(description="Worst performing trade")
    positions_summary: List[Dict] = Field(description="Summary of all positions")
    tax_summary: Dict[str, float] = Field(description="Tax liability summary")
    performance_metrics: Dict[str, float] = Field(description="Performance metrics")
    file_path: str = Field(description="Generated report file path")
    recommendations: List[str] = Field(description="Accounting recommendations")

class BaseAIAgent(ABC):
    """Base class for all AI agents"""
    
    def __init__(self, name: str, model: str = "gpt-4o", temperature: float = 0.3):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.conversation_history = []
        
    @abstractmethod
    def create_parser(self) -> PydanticOutputParser:
        """Create the output parser for this agent"""
        pass
        
    @abstractmethod
    def create_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for this agent"""
        pass
        
    @abstractmethod
    def create_tools(self) -> List[BaseTool]:
        """Create tools for this agent"""
        pass
        
    async def process_request(self, query: str, context: Dict[str, Any]) -> Any:
        """Process a request and return structured response"""
        try:
            parser = self.create_parser()
            prompt = self.create_prompt()
            tools = self.create_tools()
            
            # Create agent
            agent = create_tool_calling_agent(
                llm=self.llm,
                prompt=prompt,
                tools=tools,
            )
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
            
            # Execute agent
            raw_response = agent_executor.invoke({
                "query": query,
                "context": json.dumps(context, default=str),
                "agent_scratchpad": "",
                "chat_history": self.conversation_history[-5:] if self.conversation_history else []
            })
            
            # Parse structured response
            structured_response = parser.parse(raw_response.get("output"))
            
            # Update conversation history
            self.conversation_history.append({
                "timestamp": datetime.now(),
                "query": query,
                "response": raw_response.get("output")
            })
            
            return structured_response
            
        except Exception as e:
            logger.error(f"Agent {self.name} processing error: {e}")
            return None

class FinancialAnalysisAgent(BaseAIAgent):
    """Financial Analysis Agent - Comprehensive market analysis"""
    
    def __init__(self):
        super().__init__("FinancialAnalysis", temperature=0.2)
        
    def create_parser(self) -> PydanticOutputParser:
        return PydanticOutputParser(pydantic_object=FinancialAnalysisResponse)
        
    def create_prompt(self) -> ChatPromptTemplate:
        parser = self.create_parser()
        return ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert financial analyst specializing in cryptocurrency markets.
            Your role is to provide comprehensive technical and fundamental analysis.
            
            Analyze the provided market data and context to identify:
            - Current market regime and trends
            - Key support/resistance levels
            - Risk factors and opportunities
            - Price targets and timing
            
            Use technical analysis, market structure, and on-chain data.
            Be precise with levels and timing. Provide actionable insights.
            
            Wrap your analysis in the required format:
            {format_instructions}
            """),
            ("placeholder", "{chat_history}"),
            ("human", "Analyze the current market situation: {query}\n\nContext: {context}"),
            ("placeholder", "{agent_scratchpad}"),
        ]).partial(format_instructions=parser.get_format_instructions())
        
    def create_tools(self) -> List[BaseTool]:
        # Market analysis tools would go here
        return []

class BrokerAgent(BaseAIAgent):
    """Broker Agent - Trade execution and order management"""
    
    def __init__(self):
        super().__init__("Broker", temperature=0.1)
        
    def create_parser(self) -> PydanticOutputParser:
        return PydanticOutputParser(pydantic_object=BrokerResponse)
        
    def create_prompt(self) -> ChatPromptTemplate:
        parser = self.create_parser()
        return ChatPromptTemplate.from_messages([
            ("system", """
            You are a professional cryptocurrency broker focused on optimal trade execution.
            Your role is to make precise trading decisions based on analysis and market conditions.
            
            Consider:
            - Current market conditions and liquidity
            - Position sizing and risk management
            - Optimal entry/exit timing
            - Order types and execution strategy
            
            Always prioritize capital preservation and risk-adjusted returns.
            Be specific with prices, quantities, and execution instructions.
            
            Wrap your recommendations in the required format:
            {format_instructions}
            """),
            ("placeholder", "{chat_history}"),
            ("human", "Trading decision needed: {query}\n\nMarket Context: {context}"),
            ("placeholder", "{agent_scratchpad}"),
        ]).partial(format_instructions=parser.get_format_instructions())
        
    def create_tools(self) -> List[BaseTool]:
        # Trading tools would go here
        return []

class RiskManagementAgent(BaseAIAgent):
    """Risk Management Agent - Portfolio risk assessment and control"""
    
    def __init__(self):
        super().__init__("RiskManagement", temperature=0.1)
        
    def create_parser(self) -> PydanticOutputParser:
        return PydanticOutputParser(pydantic_object=RiskManagementResponse)
        
    def create_prompt(self) -> ChatPromptTemplate:
        parser = self.create_parser()
        return ChatPromptTemplate.from_messages([
            ("system", """
            You are a senior risk management specialist for cryptocurrency trading.
            Your primary responsibility is protecting capital and optimizing risk-adjusted returns.
            
            Assess and manage:
            - Portfolio exposure and concentration risk
            - Position sizing and leverage
            - Stop loss and take profit levels
            - Market correlation and volatility risks
            - Liquidity and execution risks
            
            Always err on the side of caution. Capital preservation is paramount.
            Provide specific, actionable risk management recommendations.
            
            Wrap your risk assessment in the required format:
            {format_instructions}
            """),
            ("placeholder", "{chat_history}"),
            ("human", "Risk assessment needed: {query}\n\nPortfolio Context: {context}"),
            ("placeholder", "{agent_scratchpad}"),
        ]).partial(format_instructions=parser.get_format_instructions())
        
    def create_tools(self) -> List[BaseTool]:
        # Risk management tools would go here
        return []

class MarketSentimentAgent(BaseAIAgent):
    """Market Sentiment Agent - Social and news sentiment analysis"""
    
    def __init__(self):
        super().__init__("MarketSentiment", temperature=0.3)
        
    def create_parser(self) -> PydanticOutputParser:
        return PydanticOutputParser(pydantic_object=MarketSentimentResponse)
        
    def create_prompt(self) -> ChatPromptTemplate:
        parser = self.create_parser()
        return ChatPromptTemplate.from_messages([
            ("system", """
            You are a market sentiment analyst specializing in cryptocurrency markets.
            Your expertise is in analyzing social sentiment, news impact, and market psychology.
            
            Analyze:
            - Social media sentiment and trending topics
            - News impact and narrative changes
            - Fear & Greed indicators
            - Contrarian signals and crowd psychology
            - On-chain sentiment metrics
            
            Provide both current sentiment and trend analysis.
            Identify potential sentiment reversals and contrarian opportunities.
            
            Wrap your sentiment analysis in the required format:
            {format_instructions}
            """),
            ("placeholder", "{chat_history}"),
            ("human", "Sentiment analysis needed: {query}\n\nMarket Data: {context}"),
            ("placeholder", "{agent_scratchpad}"),
        ]).partial(format_instructions=parser.get_format_instructions())
        
    def create_tools(self) -> List[BaseTool]:
        # Sentiment analysis tools would go here
        return []

class PortfolioOptimizationAgent(BaseAIAgent):
    """Portfolio Optimization Agent - Asset allocation and rebalancing"""
    
    def __init__(self):
        super().__init__("PortfolioOptimization", temperature=0.2)
        
    def create_parser(self) -> PydanticOutputParser:
        return PydanticOutputParser(pydantic_object=PortfolioOptimizationResponse)
        
    def create_prompt(self) -> ChatPromptTemplate:
        parser = self.create_parser()
        return ChatPromptTemplate.from_messages([
            ("system", """
            You are a quantitative portfolio manager specializing in cryptocurrency portfolios.
            Your expertise is in optimal asset allocation and risk-return optimization.
            
            Optimize for:
            - Risk-adjusted returns (Sharpe ratio maximization)
            - Diversification across uncorrelated assets
            - Dynamic rebalancing based on market conditions
            - Correlation analysis and factor exposure
            - Liquidity and execution costs
            
            Consider market regimes and adjust allocation accordingly.
            Provide specific allocation percentages and rebalancing actions.
            
            Wrap your optimization recommendations in the required format:
            {format_instructions}
            """),
            ("placeholder", "{chat_history}"),
            ("human", "Portfolio optimization needed: {query}\n\nPortfolio Data: {context}"),
            ("placeholder", "{agent_scratchpad}"),
        ]).partial(format_instructions=parser.get_format_instructions())
        
    def create_tools(self) -> List[BaseTool]:
        # Portfolio optimization tools would go here
        return []

class AIAgentOrchestrator:
    """Orchestrates all AI agents and coordinates their responses"""
    
    def __init__(self):
        self.agents = {
            'financial_analysis': FinancialAnalysisAgent(),
            'broker': BrokerAgent(),
            'risk_management': RiskManagementAgent(),
            'market_sentiment': MarketSentimentAgent(),
            'portfolio_optimization': PortfolioOptimizationAgent()
        }
        self.agent_priorities = {
            'risk_management': 1,  # Highest priority
            'financial_analysis': 2,
            'broker': 3,
            'portfolio_optimization': 4,
            'market_sentiment': 5
        }
        
    async def get_market_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive market analysis from multiple agents"""
        
        # Prepare context for agents
        context = {
            'timestamp': datetime.now().isoformat(),
            'market_data': market_data,
            'request_type': 'market_analysis'
        }
        
        # Query financial analysis agent
        financial_query = f"""
        Analyze the current market situation for the provided symbols.
        Focus on technical analysis, trend identification, and key levels.
        Data includes: OHLCV, order book, funding rates, and volume metrics.
        """
        
        financial_analysis = await self.agents['financial_analysis'].process_request(
            financial_query, context
        )
        
        # Query market sentiment agent
        sentiment_query = f"""
        Analyze current market sentiment and social indicators.
        Look for sentiment extremes, trend changes, and contrarian signals.
        """
        
        sentiment_analysis = await self.agents['market_sentiment'].process_request(
            sentiment_query, context
        )
        
        return {
            'financial_analysis': financial_analysis,
            'sentiment_analysis': sentiment_analysis,
            'timestamp': datetime.now(),
            'context': context
        }
        
    async def get_trading_decision(self, 
                                  signal_data: Dict[str, Any], 
                                  portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get trading decision from broker and risk management agents"""
        
        context = {
            'timestamp': datetime.now().isoformat(),
            'signal_data': signal_data,
            'portfolio_data': portfolio_data,
            'request_type': 'trading_decision'
        }
        
        # Get risk assessment first (highest priority)
        risk_query = f"""
        Assess the risk of the proposed trading signals and current portfolio state.
        Evaluate position sizing, stop losses, portfolio exposure, and correlation risks.
        Provide specific risk management recommendations.
        Signals: {json.dumps(signal_data, default=str)}
        """
        
        risk_assessment = await self.agents['risk_management'].process_request(
            risk_query, context
        )
        
        # Update context with risk assessment
        context['risk_assessment'] = risk_assessment
        
        # Get broker recommendation
        broker_query = f"""
        Based on the trading signals and risk assessment, provide specific trading recommendations.
        Include entry/exit prices, position sizes, and execution strategy.
        Risk assessment: {json.dumps(asdict(risk_assessment) if risk_assessment else {}, default=str)}
        """
        
        broker_decision = await self.agents['broker'].process_request(
            broker_query, context
        )
        
        return {
            'broker_decision': broker_decision,
            'risk_assessment': risk_assessment,
            'timestamp': datetime.now(),
            'context': context
        }
        
    async def get_portfolio_optimization(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get portfolio optimization recommendations"""
        
        context = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_data': portfolio_data,
            'request_type': 'portfolio_optimization'
        }
        
        optimization_query = f"""
        Optimize the current portfolio allocation for maximum risk-adjusted returns.
        Consider correlation, diversification, and current market conditions.
        Provide specific rebalancing recommendations.
        """
        
        optimization_result = await self.agents['portfolio_optimization'].process_request(
            optimization_query, context
        )
        
        return {
            'optimization_result': optimization_result,
            'timestamp': datetime.now(),
            'context': context
        }
        
    async def emergency_risk_check(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Emergency risk assessment for critical situations"""
        
        context = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_data': portfolio_data,
            'request_type': 'emergency_risk_check',
            'emergency': True
        }
        
        emergency_query = f"""
        EMERGENCY RISK ASSESSMENT NEEDED.
        Evaluate portfolio for immediate risks and provide emergency actions if required.
        Focus on capital preservation and immediate risk reduction.
        """
        
        emergency_assessment = await self.agents['risk_management'].process_request(
            emergency_query, context
        )
        
        return {
            'emergency_assessment': emergency_assessment,
            'timestamp': datetime.now(),
            'is_emergency': True,
            'context': context
        }
        
    def get_agent_status(self) -> Dict[str, Dict]:
        """Get status of all agents"""
        status = {}
        for name, agent in self.agents.items():
            status[name] = {
                'name': agent.name,
                'model': agent.model,
                'conversation_history_length': len(agent.conversation_history),
                'last_activity': agent.conversation_history[-1]['timestamp'] if agent.conversation_history else None
            }
        return status

# Specialized Expert Agents using your template structure

class CryptoCurrencyAnalysisExpert:
    """Cryptocurrency Analysis Expert using your template"""
    
    def __init__(self):
        self.gpt = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
    def get_crypto_analysis(self, query: str, market_data: Dict[str, Any]) -> Optional[FinancialAnalysisResponse]:
        """Get cryptocurrency analysis using your template"""
        try:
            parser = PydanticOutputParser(pydantic_object=FinancialAnalysisResponse)
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are a cryptocurrency analysis expert with deep knowledge of blockchain technology, 
                market dynamics, and technical analysis. Your expertise includes:
                
                - On-chain analysis and whale movements
                - DeFi protocol analysis and yield farming
                - Tokenomics and fundamental analysis
                - Market microstructure and liquidity analysis
                - Cross-chain analysis and bridge dynamics
                
                Analyze the cryptocurrency market data and provide insights on:
                - Price movements and technical patterns
                - On-chain metrics and network health
                - Market regime and trend analysis
                - Risk factors specific to crypto markets
                - Trading opportunities and entry/exit points
                
                Wrap the output in this format and provide no other text
                {format_instructions}
                """),
                ("human", "{query}"),
                ("placeholder", "{agent_scratchpad}"),
            ]).partial(format_instructions=parser.get_format_instructions())
            
            tools = []  # Add crypto-specific tools here
            agent = create_tool_calling_agent(llm=self.gpt, prompt=prompt, tools=tools)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            raw_response = agent_executor.invoke({
                "query": f"{query}\n\nMarket Data: {json.dumps(market_data, default=str)}",
            })
            
            structured_response = parser.parse(raw_response.get("output"))
            return structured_response
            
        except Exception as e:
            logger.error(f"Crypto analysis expert error: {e}")
            return None

class FinancialExpert:
    """Financial Expert using your template"""
    
    def __init__(self):
        self.gpt = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
    def get_financial_analysis(self, query: str, financial_data: Dict[str, Any]) -> Optional[FinancialAnalysisResponse]:
        """Get financial analysis using your template"""
        try:
            parser = PydanticOutputParser(pydantic_object=FinancialAnalysisResponse)
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are a senior financial analyst with expertise in:
                
                - Macroeconomic analysis and market cycles
                - Risk management and portfolio theory
                - Quantitative finance and derivatives
                - Market microstructure and liquidity
                - Regulatory impact analysis
                
                Your role is to provide comprehensive financial analysis including:
                - Market regime identification and trend analysis
                - Risk-return optimization and correlation analysis
                - Valuation models and fair value assessment
                - Economic indicators impact on crypto markets
                - Institutional flow analysis and market sentiment
                
                Wrap the output in this format and provide no other text
                {format_instructions}
                """),
                ("human", "{query}"),
                ("placeholder", "{agent_scratchpad}"),
            ]).partial(format_instructions=parser.get_format_instructions())
            
            tools = []  # Add financial analysis tools here
            agent = create_tool_calling_agent(llm=self.gpt, prompt=prompt, tools=tools)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            raw_response = agent_executor.invoke({
                "query": f"{query}\n\nFinancial Data: {json.dumps(financial_data, default=str)}",
            })
            
            structured_response = parser.parse(raw_response.get("output"))
            return structured_response
            
        except Exception as e:
            logger.error(f"Financial expert error: {e}")
            return None

class ExpertBroker:
    """Expert Broker using your template"""
    
    def __init__(self):
        self.gpt = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
    def get_trading_decision(self, query: str, trading_context: Dict[str, Any]) -> Optional[BrokerResponse]:
        """Get trading decision using your template"""
        try:
            parser = PydanticOutputParser(pydantic_object=BrokerResponse)
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are an expert cryptocurrency broker with extensive experience in:
                
                - High-frequency trading and market making
                - Options and derivatives trading
                - Risk management and position sizing
                - Order execution and slippage optimization
                - Cross-exchange arbitrage and liquidity provision
                
                Your expertise includes:
                - Optimal trade execution strategies
                - Market impact analysis and cost minimization
                - Dynamic hedging and risk neutrality
                - Algorithmic trading and smart order routing
                - Regulatory compliance and best execution
                
                Provide specific trading recommendations with:
                - Entry/exit prices and timing
                - Position sizing and risk parameters
                - Order types and execution strategy
                - Risk management and stop losses
                - Market conditions assessment
                
                Wrap the output in this format and provide no other text
                {format_instructions}
                """),
                ("human", "{query}"),
                ("placeholder", "{agent_scratchpad}"),
            ]).partial(format_instructions=parser.get_format_instructions())
            
            tools = []  # Add broker tools here
            agent = create_tool_calling_agent(llm=self.gpt, prompt=prompt, tools=tools)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            raw_response = agent_executor.invoke({
                "query": f"{query}\n\nTrading Context: {json.dumps(trading_context, default=str)}",
            })
            
            structured_response = parser.parse(raw_response.get("output"))
            return structured_response
            
        except Exception as e:
            logger.error(f"Expert broker error: {e}")
            return None

class RiskManagementExpert:
    """Risk Management Expert using your template"""
    
    def __init__(self):
        self.gpt = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
    def get_risk_assessment(self, query: str, portfolio_data: Dict[str, Any]) -> Optional[RiskManagementResponse]:
        """Get risk assessment using your template"""
        try:
            parser = PydanticOutputParser(pydantic_object=RiskManagementResponse)
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are a senior risk management specialist with expertise in:
                
                - Value at Risk (VaR) and Expected Shortfall (ES)
                - Stress testing and scenario analysis
                - Correlation analysis and tail risk assessment
                - Liquidity risk and market impact modeling
                - Regulatory capital requirements (Basel III, MiFID II)
                
                Your responsibilities include:
                - Portfolio risk measurement and monitoring
                - Position sizing and concentration limits
                - Dynamic hedging and risk neutralization
                - Crisis management and emergency procedures
                - Performance attribution and risk-adjusted returns
                
                Focus on:
                - Identifying and quantifying all risk factors
                - Providing specific risk mitigation strategies
                - Setting appropriate position limits and stop losses
                - Monitoring correlation and tail risks
                - Emergency action plans for extreme scenarios
                
                Wrap the output in this format and provide no other text
                {format_instructions}
                """),
                ("human", "{query}"),
                ("placeholder", "{agent_scratchpad}"),
            ]).partial(format_instructions=parser.get_format_instructions())
            
            tools = []  # Add risk management tools here
            agent = create_tool_calling_agent(llm=self.gpt, prompt=prompt, tools=tools)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            raw_response = agent_executor.invoke({
                "query": f"{query}\n\nPortfolio Data: {json.dumps(portfolio_data, default=str)}",
            })
            
            structured_response = parser.parse(raw_response.get("output"))
            return structured_response
            
        except Exception as e:
            logger.error(f"Risk management expert error: {e}")
            return None

class MarketSentimentExpert:
    """Market Sentiment Expert using your template"""
    
    def __init__(self):
        self.gpt = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
    def get_sentiment_analysis(self, query: str, sentiment_data: Dict[str, Any]) -> Optional[MarketSentimentResponse]:
        """Get sentiment analysis using your template"""
        try:
            parser = PydanticOutputParser(pydantic_object=MarketSentimentResponse)
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are a market sentiment expert specializing in:
                
                - Social media sentiment analysis (Twitter, Reddit, Discord)
                - News sentiment and narrative analysis
                - On-chain sentiment metrics and whale behavior
                - Fear & Greed index interpretation
                - Contrarian investing and crowd psychology
                
                Your expertise includes:
                - Real-time sentiment tracking and trend analysis
                - Identifying sentiment extremes and reversal signals
                - Social media influence and viral content impact
                - News cycle analysis and market reaction prediction
                - Behavioral finance and psychological market drivers
                
                Analyze sentiment data to provide:
                - Current sentiment score and trend direction
                - Key sentiment drivers and narrative changes
                - Contrarian signals and crowd behavior analysis
                - Social media buzz and viral content impact
                - Fear/greed extremes and reversal opportunities
                
                Wrap the output in this format and provide no other text
                {format_instructions}
                """),
                ("human", "{query}"),
                ("placeholder", "{agent_scratchpad}"),
            ]).partial(format_instructions=parser.get_format_instructions())
            
            tools = []  # Add sentiment analysis tools here
            agent = create_tool_calling_agent(llm=self.gpt, prompt=prompt, tools=tools)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            raw_response = agent_executor.invoke({
                "query": f"{query}\n\nSentiment Data: {json.dumps(sentiment_data, default=str)}",
            })
            
            structured_response = parser.parse(raw_response.get("output"))
            return structured_response
            
        except Exception as e:
            logger.error(f"Market sentiment expert error: {e}")
            return None

class PortfolioOptimizationExpert:
    """Portfolio Optimization Expert using your template"""
    
    def __init__(self):
        self.gpt = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
    def get_portfolio_optimization(self, query: str, portfolio_data: Dict[str, Any]) -> Optional[PortfolioOptimizationResponse]:
        """Get portfolio optimization using your template"""
        try:
            parser = PydanticOutputParser(pydantic_object=PortfolioOptimizationResponse)
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are a quantitative portfolio optimization expert with expertise in:
                
                - Modern Portfolio Theory and efficient frontier analysis
                - Factor models and risk attribution
                - Dynamic asset allocation and tactical rebalancing
                - Alternative risk premia and smart beta strategies
                - Multi-asset portfolio construction and optimization
                
                Your specializations include:
                - Mean-variance optimization and risk budgeting
                - Black-Litterman model and Bayesian optimization
                - Risk parity and equal risk contribution strategies
                - Dynamic hedging and regime-based allocation
                - Transaction cost analysis and rebalancing optimization
                
                Provide portfolio optimization including:
                - Optimal asset allocation weights
                - Risk-return trade-offs and efficient frontier
                - Rebalancing recommendations and timing
                - Diversification analysis and correlation management
                - Performance attribution and factor exposure
                
                Wrap the output in this format and provide no other text
                {format_instructions}
                """),
                ("human", "{query}"),
                ("placeholder", "{agent_scratchpad}"),
            ]).partial(format_instructions=parser.get_format_instructions())
            
            tools = []  # Add portfolio optimization tools here
            agent = create_tool_calling_agent(llm=self.gpt, prompt=prompt, tools=tools)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            raw_response = agent_executor.invoke({
                "query": f"{query}\n\nPortfolio Data: {json.dumps(portfolio_data, default=str)}",
            })
            
            structured_response = parser.parse(raw_response.get("output"))
            return structured_response
            
        except Exception as e:
            logger.error(f"Portfolio optimization expert error: {e}")
            return None

class TechnicalAnalysisExpert:
    """Technical Analysis Expert using your template"""
    
    def __init__(self):
        self.gpt = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
    def get_technical_analysis(self, query: str, chart_data: Dict[str, Any]) -> Optional[FinancialAnalysisResponse]:
        """Get technical analysis using your template"""
        try:
            parser = PydanticOutputParser(pydantic_object=FinancialAnalysisResponse)
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are a technical analysis expert with deep expertise in:
                
                - Chart pattern recognition and trend analysis
                - Technical indicators and oscillators
                - Volume analysis and market microstructure
                - Support/resistance levels and breakout analysis
                - Elliott Wave theory and Fibonacci analysis
                
                Your technical analysis expertise includes:
                - Multi-timeframe analysis and trend confluence
                - Momentum indicators and divergence analysis
                - Volume profile and order flow analysis
                - Market structure and smart money concepts
                - Algorithmic pattern recognition and backtesting
                
                Provide comprehensive technical analysis with:
                - Trend direction and strength assessment
                - Key support and resistance levels
                - Entry/exit signals and price targets
                - Risk management levels and stop losses
                - Volume analysis and market structure insights
                
                Wrap the output in this format and provide no other text
                {format_instructions}
                """),
                ("human", "{query}"),
                ("placeholder", "{agent_scratchpad}"),
            ]).partial(format_instructions=parser.get_format_instructions())
            
            tools = []  # Add technical analysis tools here
            agent = create_tool_calling_agent(llm=self.gpt, prompt=prompt, tools=tools)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            raw_response = agent_executor.invoke({
                "query": f"{query}\n\nChart Data: {json.dumps(chart_data, default=str)}",
            })
            
            structured_response = parser.parse(raw_response.get("output"))
            return structured_response
            
        except Exception as e:
            logger.error(f"Technical analysis expert error: {e}")
            return None

class AccountantExpert:
    """Accountant Expert using your template - Generates financial reports and tax summaries"""
    
    def __init__(self):
        self.gpt = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
    def generate_monthly_report(self, query: str, trading_data: Dict[str, Any]) -> Optional[AccountantResponse]:
        """Generate monthly financial report using your template"""
        try:
            parser = PydanticOutputParser(pydantic_object=AccountantResponse)
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are a professional cryptocurrency accountant and tax specialist with expertise in:
                
                - Cryptocurrency tax regulations and compliance
                - Capital gains/losses calculation (FIFO, LIFO, specific identification)
                - Trading performance analysis and reporting
                - Financial statement preparation for crypto trading
                - Tax optimization strategies and deductions
                
                Your responsibilities include:
                - Monthly P&L statements and performance reports
                - Realized/unrealized gains and losses tracking
                - Fee analysis and deduction optimization
                - Trade performance metrics and statistics
                - Tax liability calculations and planning
                
                Generate a comprehensive MONTHLY report including:
                - Complete P&L breakdown with realized/unrealized gains
                - Trading statistics (win rate, average trade, volume)
                - Fee analysis and tax deductions
                - Position summaries with cost basis tracking
                - Tax liability estimates and recommendations
                - Performance metrics (Sharpe ratio, max drawdown, etc.)
                - Recommendations for tax optimization
                
                Create detailed file-based reports for record keeping and tax purposes.
                
                Wrap the output in this format and provide no other text
                {format_instructions}
                """),
                ("human", "{query}"),
                ("placeholder", "{agent_scratchpad}"),
            ]).partial(format_instructions=parser.get_format_instructions())
            
            tools = []  # Add accounting tools here
            agent = create_tool_calling_agent(llm=self.gpt, prompt=prompt, tools=tools)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            raw_response = agent_executor.invoke({
                "query": f"{query}\n\nTrading Data: {json.dumps(trading_data, default=str)}",
            })
            
            structured_response = parser.parse(raw_response.get("output"))
            
            # Generate the actual report file
            self._generate_report_file(structured_response, "monthly")
            
            return structured_response
            
        except Exception as e:
            logger.error(f"Accountant expert monthly report error: {e}")
            return None
            
    def generate_yearly_report(self, query: str, trading_data: Dict[str, Any]) -> Optional[AccountantResponse]:
        """Generate yearly financial report using your template"""
        try:
            parser = PydanticOutputParser(pydantic_object=AccountantResponse)
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are a professional cryptocurrency accountant and tax specialist with expertise in:
                
                - Cryptocurrency tax regulations and compliance
                - Capital gains/losses calculation (FIFO, LIFO, specific identification)
                - Annual tax return preparation for crypto trading
                - Financial statement preparation and audit support
                - Tax optimization strategies and long-term planning
                
                Your responsibilities include:
                - Annual tax returns and compliance reporting
                - Year-end financial statements and summaries
                - Long-term performance analysis and trends
                - Tax strategy optimization and planning
                - Audit preparation and documentation
                
                Generate a comprehensive YEARLY report including:
                - Complete annual P&L with detailed breakdowns
                - Annual trading performance and statistics
                - Comprehensive tax liability calculations
                - Year-end position valuations and basis tracking
                - Annual fee analysis and deduction summaries
                - Performance trends and year-over-year comparisons
                - Tax optimization recommendations for next year
                - Detailed audit trail and supporting documentation
                
                Prepare detailed annual reports suitable for tax filing and audit purposes.
                
                Wrap the output in this format and provide no other text
                {format_instructions}
                """),
                ("human", "{query}"),
                ("placeholder", "{agent_scratchpad}"),
            ]).partial(format_instructions=parser.get_format_instructions())
            
            tools = []  # Add accounting tools here
            agent = create_tool_calling_agent(llm=self.gpt, prompt=prompt, tools=tools)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            raw_response = agent_executor.invoke({
                "query": f"{query}\n\nTrading Data: {json.dumps(trading_data, default=str)}",
            })
            
            structured_response = parser.parse(raw_response.get("output"))
            
            # Generate the actual report file
            self._generate_report_file(structured_response, "yearly")
            
            return structured_response
            
        except Exception as e:
            logger.error(f"Accountant expert yearly report error: {e}")
            return None
            
    def _generate_report_file(self, report_data: AccountantResponse, report_type: str):
        """Generate actual report files (CSV, PDF, Excel)"""
        try:
            import pandas as pd
            from datetime import datetime
            import os
            
            # Create reports directory
            reports_dir = "trading_reports"
            os.makedirs(reports_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{report_type}_report_{timestamp}"
            
            # Generate CSV report
            csv_data = {
                'Metric': [
                    'Total P&L', 'Realized Gains', 'Realized Losses', 
                    'Unrealized Gains', 'Unrealized Losses', 'Total Fees',
                    'Total Volume', 'Trade Count', 'Winning Trades', 
                    'Losing Trades', 'Win Rate %', 'Average Trade P&L'
                ],
                'Value': [
                    report_data.total_pnl, report_data.realized_gains, 
                    report_data.realized_losses, report_data.unrealized_gains,
                    report_data.unrealized_losses, report_data.total_fees,
                    report_data.total_volume, report_data.trade_count,
                    report_data.winning_trades, report_data.losing_trades,
                    report_data.win_rate, report_data.average_trade_pnl
                ]
            }
            
            df = pd.DataFrame(csv_data)
            csv_path = os.path.join(reports_dir, f"{base_filename}.csv")
            df.to_csv(csv_path, index=False)
            
            # Generate detailed positions report
            if report_data.positions_summary:
                positions_df = pd.DataFrame(report_data.positions_summary)
                positions_path = os.path.join(reports_dir, f"{base_filename}_positions.csv")
                positions_df.to_csv(positions_path, index=False)
            
            # Generate tax summary report
            tax_data = {
                'Tax Category': list(report_data.tax_summary.keys()),
                'Amount': list(report_data.tax_summary.values())
            }
            tax_df = pd.DataFrame(tax_data)
            tax_path = os.path.join(reports_dir, f"{base_filename}_tax_summary.csv")
            tax_df.to_csv(tax_path, index=False)
            
            # Update report data with file paths
            report_data.file_path = csv_path
            
            logger.info(f"✅ Generated {report_type} report files:")
            logger.info(f"  - Main report: {csv_path}")
            logger.info(f"  - Positions: {positions_path if report_data.positions_summary else 'N/A'}")
            logger.info(f"  - Tax summary: {tax_path}")
            
        except Exception as e:
            logger.error(f"Report file generation error: {e}")
            
    def generate_tax_form_8949(self, trading_data: Dict[str, Any]) -> str:
        """Generate IRS Form 8949 compatible report for crypto trading"""
        try:
            import pandas as pd
            from datetime import datetime
            import os
            
            reports_dir = "trading_reports"
            os.makedirs(reports_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            form_path = os.path.join(reports_dir, f"form_8949_crypto_{timestamp}.csv")
            
            # Form 8949 columns
            form_data = {
                'Description': [],
                'Date_Acquired': [],
                'Date_Sold': [],
                'Proceeds': [],
                'Cost_Basis': [],
                'Adjustment_Code': [],
                'Adjustment_Amount': [],
                'Gain_Loss': []
            }
            
            # Process trading data for Form 8949
            if 'trades' in trading_data:
                for trade in trading_data['trades']:
                    if trade.get('status') == 'closed':
                        form_data['Description'].append(f"{trade.get('symbol', 'CRYPTO')} - {trade.get('quantity', 0)} units")
                        form_data['Date_Acquired'].append(trade.get('entry_time', ''))
                        form_data['Date_Sold'].append(trade.get('exit_time', ''))
                        form_data['Proceeds'].append(trade.get('exit_value', 0))
                        form_data['Cost_Basis'].append(trade.get('entry_value', 0))
                        form_data['Adjustment_Code'].append('')
                        form_data['Adjustment_Amount'].append(0)
                        form_data['Gain_Loss'].append(trade.get('pnl', 0))
            
            df = pd.DataFrame(form_data)
            df.to_csv(form_path, index=False)
            
            logger.info(f"✅ Generated Form 8949 report: {form_path}")
            return form_path
            
        except Exception as e:
            logger.error(f"Form 8949 generation error: {e}")
            return ""

# Enhanced AI Agent Orchestrator with Expert Agents
class EnhancedAIAgentOrchestrator(AIAgentOrchestrator):
    """Enhanced orchestrator with specialized expert agents"""
    
    def __init__(self):
        super().__init__()
        
        # Add expert agents using your template
        self.expert_agents = {
            'crypto_analysis_expert': CryptoCurrencyAnalysisExpert(),
            'financial_expert': FinancialExpert(),
            'expert_broker': ExpertBroker(),
            'risk_management_expert': RiskManagementExpert(),
            'market_sentiment_expert': MarketSentimentExpert(),
            'portfolio_optimization_expert': PortfolioOptimizationExpert(),
            'technical_analysis_expert': TechnicalAnalysisExpert(),
            'accountant_expert': AccountantExpert()
        }
        
    async def get_comprehensive_analysis(self, query: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive analysis from all expert agents"""
        
        results = {}
        
        try:
            # Crypto Analysis Expert
            crypto_analysis = self.expert_agents['crypto_analysis_expert'].get_crypto_analysis(
                f"Analyze the cryptocurrency market conditions: {query}", market_data
            )
            results['crypto_analysis'] = crypto_analysis
            
            # Financial Expert
            financial_analysis = self.expert_agents['financial_expert'].get_financial_analysis(
                f"Provide financial analysis: {query}", market_data
            )
            results['financial_analysis'] = financial_analysis
            
            # Technical Analysis Expert
            technical_analysis = self.expert_agents['technical_analysis_expert'].get_technical_analysis(
                f"Provide technical analysis: {query}", market_data
            )
            results['technical_analysis'] = technical_analysis
            
            # Market Sentiment Expert
            sentiment_analysis = self.expert_agents['market_sentiment_expert'].get_sentiment_analysis(
                f"Analyze market sentiment: {query}", market_data
            )
            results['sentiment_analysis'] = sentiment_analysis
            
            logger.info(f"✅ Comprehensive analysis completed with {len(results)} expert insights")
            
        except Exception as e:
            logger.error(f"Comprehensive analysis error: {e}")
            
        return results
        
    async def get_trading_recommendation(self, query: str, trading_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get trading recommendation from expert broker and risk management"""
        
        results = {}
        
        try:
            # Expert Broker
            broker_decision = self.expert_agents['expert_broker'].get_trading_decision(
                f"Provide trading recommendation: {query}", trading_context
            )
            results['broker_decision'] = broker_decision
            
            # Risk Management Expert
            risk_assessment = self.expert_agents['risk_management_expert'].get_risk_assessment(
                f"Assess trading risk: {query}", trading_context
            )
            results['risk_assessment'] = risk_assessment
            
            logger.info(f"✅ Trading recommendation completed")
            
        except Exception as e:
            logger.error(f"Trading recommendation error: {e}")
            
        return results
        
    async def get_portfolio_insights(self, query: str, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get portfolio optimization insights"""
        
        try:
            optimization_result = self.expert_agents['portfolio_optimization_expert'].get_portfolio_optimization(
                f"Optimize portfolio: {query}", portfolio_data
            )
            
            return {
                'optimization_result': optimization_result,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Portfolio insights error: {e}")
            return {}
            
    async def generate_monthly_financial_report(self, query: str, trading_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate monthly financial report with file output"""
        
        try:
            monthly_report = self.expert_agents['accountant_expert'].generate_monthly_report(
                f"Generate monthly financial report: {query}", trading_data
            )
            
            return {
                'monthly_report': monthly_report,
                'timestamp': datetime.now(),
                'report_type': 'monthly'
            }
            
        except Exception as e:
            logger.error(f"Monthly financial report error: {e}")
            return {}
            
    async def generate_yearly_financial_report(self, query: str, trading_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate yearly financial report with file output"""
        
        try:
            yearly_report = self.expert_agents['accountant_expert'].generate_yearly_report(
                f"Generate yearly financial report: {query}", trading_data
            )
            
            return {
                'yearly_report': yearly_report,
                'timestamp': datetime.now(),
                'report_type': 'yearly'
            }
            
        except Exception as e:
            logger.error(f"Yearly financial report error: {e}")
            return {}
            
    async def generate_tax_reports(self, trading_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive tax reports including Form 8949"""
        
        try:
            # Generate Form 8949
            form_8949_path = self.expert_agents['accountant_expert'].generate_tax_form_8949(trading_data)
            
            # Generate yearly report for tax purposes
            tax_report = self.expert_agents['accountant_expert'].generate_yearly_report(
                "Generate comprehensive tax report for filing purposes", trading_data
            )
            
            return {
                'tax_report': tax_report,
                'form_8949_path': form_8949_path,
                'timestamp': datetime.now(),
                'report_type': 'tax'
            }
            
        except Exception as e:
            logger.error(f"Tax reports generation error: {e}")
            return {}

# Factory function to create the enhanced orchestrator
def create_ai_agent_orchestrator() -> EnhancedAIAgentOrchestrator:
    """Create and initialize the enhanced AI agent orchestrator"""
    return EnhancedAIAgentOrchestrator()

# Factory function for the original orchestrator
def create_basic_ai_agent_orchestrator() -> AIAgentOrchestrator:
    """Create and initialize the basic AI agent orchestrator"""
    return AIAgentOrchestrator()
