"""
Portfolio Optimizer - Implements portfolio optimization strategies using modern portfolio theory.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize
from loguru import logger

class PortfolioOptimizer:
    """Portfolio optimization using modern portfolio theory"""
    
    def __init__(self, min_weight: float = 0.01, max_weight: float = 0.3):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
    def calculate_expected_returns(self, returns_data: pd.DataFrame) -> pd.Series:
        """Calculate expected returns (annualized)"""
        return returns_data.mean() * 252
    
    def calculate_covariance_matrix(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate covariance matrix (annualized)"""
        return returns_data.cov() * 252
    
    def optimize_maximum_sharpe(self, expected_returns: pd.Series, 
                               cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """Optimize for maximum Sharpe ratio"""
        n_assets = len(expected_returns)
        
        def negative_sharpe(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if portfolio_vol == 0:
                return -np.inf
            return -(portfolio_return - self.risk_free_rate) / portfolio_vol
        
        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds: min and max weights
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        try:
            result = minimize(negative_sharpe, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = dict(zip(expected_returns.index, result.x))
                return {k: max(v, 0) for k, v in weights.items()}
            else:
                logger.warning("Sharpe optimization failed, using equal weights")
        except Exception as e:
            logger.error(f"Sharpe optimization error: {e}")
        
        # Fallback to equal weights
        equal_weight = 1.0 / n_assets
        return {symbol: equal_weight for symbol in expected_returns.index}
    
    def optimize_minimum_variance(self, cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """Optimize for minimum variance"""
        n_assets = len(cov_matrix)
        
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        try:
            result = minimize(objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = dict(zip(cov_matrix.index, result.x))
                return {k: max(v, 0) for k, v in weights.items()}
        except Exception as e:
            logger.error(f"Min variance optimization error: {e}")
        
        # Fallback to equal weights
        equal_weight = 1.0 / n_assets
        return {symbol: equal_weight for symbol in cov_matrix.index}
    
    def calculate_portfolio_metrics(self, weights: Dict[str, float], 
                                  expected_returns: pd.Series, 
                                  cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        w = pd.Series(weights)
        
        # Align indices
        common_assets = w.index.intersection(expected_returns.index).intersection(cov_matrix.index)
        if len(common_assets) == 0:
            return {'expected_return': 0, 'volatility': 0, 'sharpe_ratio': 0}
            
        w = w[common_assets]
        expected_returns_aligned = expected_returns[common_assets]
        cov_matrix_aligned = cov_matrix.loc[common_assets, common_assets]
        
        # Portfolio metrics
        portfolio_return = np.sum(w * expected_returns_aligned)
        portfolio_variance = np.dot(w, np.dot(cov_matrix_aligned, w))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'expected_return': float(portfolio_return),
            'volatility': float(portfolio_volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_weight': float(max(w.values())) if len(w) > 0 else 0,
            'min_weight': float(min(w.values())) if len(w) > 0 else 0
        }
    
    def optimize_portfolio(self, returns_data: pd.DataFrame, 
                          method: str = 'max_sharpe') -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Main portfolio optimization function.
        
        Args:
            returns_data: DataFrame with asset returns
            method: 'max_sharpe' or 'min_variance'
        
        Returns:
            Tuple of (optimal_weights, portfolio_metrics)
        """
        logger.info(f"Optimizing portfolio using method: {method}")
        
        if returns_data.empty or len(returns_data.columns) == 0:
            return {}, {}
        
        # Calculate inputs
        expected_returns = self.calculate_expected_returns(returns_data)
        cov_matrix = self.calculate_covariance_matrix(returns_data)
        
        # Optimize based on method
        if method == 'max_sharpe':
            weights = self.optimize_maximum_sharpe(expected_returns, cov_matrix)
        elif method == 'min_variance':
            weights = self.optimize_minimum_variance(cov_matrix)
        else:
            logger.warning(f"Unknown method {method}, using equal weights")
            n_assets = len(returns_data.columns)
            equal_weight = 1.0 / n_assets
            weights = {symbol: equal_weight for symbol in returns_data.columns}
        
        # Calculate metrics
        metrics = self.calculate_portfolio_metrics(weights, expected_returns, cov_matrix)
        
        logger.info(f"Portfolio optimization complete. Sharpe ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        
        return weights, metrics

# Legacy function for backwards compatibility
def optimize_portfolio(returns, cov_matrix, risk_free_rate=0.0):
    """Legacy function - maintained for backwards compatibility"""
    if isinstance(returns, (list, np.ndarray)) and isinstance(cov_matrix, np.ndarray):
        # Original simple implementation
        n = len(returns)
        weights = np.ones(n) / n  # Equal weights
        return weights
    else:
        # Use new implementation
        optimizer = PortfolioOptimizer()
        if hasattr(returns, 'index') and hasattr(cov_matrix, 'index'):
            # Pandas data
            df = pd.DataFrame({'returns': returns})
            weights_dict, _ = optimizer.optimize_portfolio(df)
            return list(weights_dict.values())
        else:
            # Fallback
            n = len(returns) if hasattr(returns, '__len__') else 1
            return np.ones(n) / n
