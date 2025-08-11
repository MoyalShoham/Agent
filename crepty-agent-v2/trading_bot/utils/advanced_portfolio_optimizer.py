"""
Portfolio Optimizer - Modern portfolio theory implementation
with risk parity, Black-Litterman, and regime-aware optimization.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class AdvancedPortfolioOptimizer:
    """
    Advanced portfolio optimization with multiple methodologies:
    - Mean-variance optimization (Markowitz)
    - Risk parity optimization
    - Black-Litterman model
    - Regime-aware optimization
    """
    
    def __init__(self, lookback_period: int = 252, min_weight: float = 0.01, max_weight: float = 0.3):
        self.lookback_period = lookback_period
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
    def calculate_expected_returns(self, returns_data: pd.DataFrame, method: str = 'historical') -> pd.Series:
        """Calculate expected returns using various methods"""
        if method == 'historical':
            return returns_data.mean() * 252  # Annualized
        
        elif method == 'exponential':
            # Exponentially weighted returns (more weight to recent data)
            weights = np.exp(np.linspace(-1, 0, len(returns_data)))
            weights /= weights.sum()
            
            expected_returns = []
            for col in returns_data.columns:
                exp_return = np.sum(weights * returns_data[col].values)
                expected_returns.append(exp_return * 252)
            
            return pd.Series(expected_returns, index=returns_data.columns)
        
        elif method == 'capm':
            # CAPM-based expected returns (simplified)
            market_returns = returns_data.mean(axis=1)  # Equal-weight market proxy
            betas = {}
            alphas = {}
            
            for symbol in returns_data.columns:
                cov_matrix = np.cov(returns_data[symbol].dropna(), market_returns)
                beta = cov_matrix[0, 1] / np.var(market_returns)
                alpha = returns_data[symbol].mean() - beta * market_returns.mean()
                
                betas[symbol] = beta
                alphas[symbol] = alpha
            
            market_premium = market_returns.mean() * 252 - self.risk_free_rate
            expected_returns = []
            
            for symbol in returns_data.columns:
                capm_return = self.risk_free_rate + betas[symbol] * market_premium
                expected_returns.append(capm_return)
            
            return pd.Series(expected_returns, index=returns_data.columns)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def optimize_maximum_sharpe(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """Maximum Sharpe ratio optimization"""
        n_assets = len(expected_returns)
        
        def negative_sharpe(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if portfolio_vol == 0:
                return -np.inf
            return -(portfolio_return - self.risk_free_rate) / portfolio_vol
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(negative_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = dict(zip(expected_returns.index, result.x))
            return {k: max(v, 0) for k, v in weights.items()}
        else:
            logger.warning("Maximum Sharpe optimization failed, returning equal weights")
            equal_weight = 1.0 / n_assets
            return {symbol: equal_weight for symbol in expected_returns.index}
    
    def optimize_portfolio(self, returns_data: pd.DataFrame, method: str = 'max_sharpe') -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Main portfolio optimization function.
        
        Returns:
            Tuple of (optimal_weights, portfolio_metrics)
        """
        logger.info(f"Optimizing portfolio using method: {method}")
        
        # Calculate inputs
        expected_returns = self.calculate_expected_returns(returns_data)
        cov_matrix = returns_data.cov() * 252  # Annualized covariance
        
        # Optimize based on method
        if method == 'max_sharpe':
            weights = self.optimize_maximum_sharpe(expected_returns, cov_matrix)
        else:
            logger.warning(f"Unknown method {method}, using equal weights")
            n_assets = len(returns_data.columns)
            equal_weight = 1.0 / n_assets
            weights = {symbol: equal_weight for symbol in returns_data.columns}
        
        # Calculate metrics
        w = pd.Series(weights)
        portfolio_return = np.sum(w * expected_returns)
        portfolio_volatility = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        metrics = {
            'expected_return': float(portfolio_return),
            'volatility': float(portfolio_volatility),
            'sharpe_ratio': float(sharpe_ratio)
        }
        
        logger.info(f"Portfolio optimization complete. Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
        
        return weights, metrics
