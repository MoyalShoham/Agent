#!/usr/bin/env python3
"""
Workspace Organization Script
Cleans up and organizes the crypto trading system workspace
"""

import os
import shutil
import glob
from pathlib import Path
from datetime import datetime

class WorkspaceOrganizer:
    """Organizes and cleans up the workspace"""
    
    def __init__(self, workspace_path: str = "."):
        self.workspace = Path(workspace_path).resolve()
        self.archive_dir = self.workspace / "archive"
        self.reports_dir = self.workspace / "reports"
        self.scripts_dir = self.workspace / "scripts"
        self.logs_dir = self.workspace / "logs"
        
        # Create necessary directories
        self.create_directories()
        
    def create_directories(self):
        """Create organizational directories"""
        directories = [
            self.archive_dir / "old_logs",
            self.archive_dir / "old_files", 
            self.archive_dir / "deprecated",
            self.reports_dir / "monthly",
            self.reports_dir / "yearly", 
            self.reports_dir / "tax",
            self.scripts_dir / "analysis",
            self.scripts_dir / "training",
            self.scripts_dir / "utilities",
            self.logs_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def move_old_logs(self):
        """Move old log files to archive"""
        print("📁 Moving old log files...")
        
        log_patterns = [
            "*.log",
            "trading_bot.*.log",
            "run_stop_loss.log",
            "enhanced_stop_loss.log"
        ]
        
        moved_count = 0
        for pattern in log_patterns:
            for log_file in self.workspace.glob(pattern):
                if log_file.is_file():
                    destination = self.archive_dir / "old_logs" / log_file.name
                    shutil.move(str(log_file), str(destination))
                    moved_count += 1
                    
        print(f"✅ Moved {moved_count} log files to archive")
        
    def move_deprecated_files(self):
        """Move deprecated/old files to archive"""
        print("📁 Moving deprecated files...")
        
        deprecated_patterns = [
            "*fix*.py",
            "*emergency*.py", 
            "*emergency*.json",
            "complete_emergency_fix.py",
            "adaptive_training.py",
            "auto_optimizer.py",
            "batch_training.py",
            "clean_trade_log.py",
            "continuous_training.py",
            "drawdown_optimization.py",
            "minimal_risk_increase.py",
            "optimize_positions.py",
            "performance_monitor.py",
            "position_analyzer.py",
            "quick_analysis.py",
            "quick_setup.py",
            "setup_test.py",
            "simple_test.py",
            "symbol_analysis.py",
            "trade_summary.py",
            "trading_analysis.py",
            "training_center.py",
            "training_data_mode.py",
            "train_*.py",
            "walkforward_backtest.py"
        ]
        
        moved_count = 0
        for pattern in deprecated_patterns:
            for file_path in self.workspace.glob(pattern):
                if file_path.is_file():
                    destination = self.archive_dir / "deprecated" / file_path.name
                    shutil.move(str(file_path), str(destination))
                    moved_count += 1
                    
        print(f"✅ Moved {moved_count} deprecated files to archive")
        
    def move_data_files(self):
        """Move data files to appropriate locations"""
        print("📁 Organizing data files...")
        
        # Move CSV files to data directory
        data_dir = self.workspace / "data"
        data_dir.mkdir(exist_ok=True)
        
        csv_patterns = [
            "*.csv",
            "*.pkl", 
            "*.joblib",
            "*.json"
        ]
        
        moved_count = 0
        for pattern in csv_patterns:
            for data_file in self.workspace.glob(pattern):
                if data_file.is_file() and data_file.name not in ["config.json", "package.json"]:
                    destination = data_dir / data_file.name
                    if not destination.exists():
                        shutil.move(str(data_file), str(destination))
                        moved_count += 1
                        
        print(f"✅ Moved {moved_count} data files to data directory")
        
    def organize_scripts(self):
        """Organize scripts into categories"""
        print("📁 Organizing scripts...")
        
        # Analysis scripts
        analysis_scripts = [
            "analyze_performance.py",
            "analyze_stop.py", 
            "ai_integration_report.py",
            "optimization_summary.py"
        ]
        
        # Training scripts  
        training_scripts = [
            "generate_synthetic_meta_dataset.py",
            "auto_symbol_agent.py",
            "backtest_optimize.py"
        ]
        
        # Utility scripts
        utility_scripts = [
            "download_binance_ohlcv.py",
            "orderbook_collector.py",
            "orderbook_features.py"
        ]
        
        moved_count = 0
        
        # Move analysis scripts
        for script in analysis_scripts:
            script_path = self.workspace / script
            if script_path.exists():
                destination = self.scripts_dir / "analysis" / script
                shutil.move(str(script_path), str(destination))
                moved_count += 1
                
        # Move training scripts
        for script in training_scripts:
            script_path = self.workspace / script
            if script_path.exists():
                destination = self.scripts_dir / "training" / script
                shutil.move(str(script_path), str(destination))
                moved_count += 1
                
        # Move utility scripts
        for script in utility_scripts:
            script_path = self.workspace / script
            if script_path.exists():
                destination = self.scripts_dir / "utilities" / script
                shutil.move(str(script_path), str(destination))
                moved_count += 1
                
        print(f"✅ Organized {moved_count} scripts into categories")
        
    def organize_tests(self):
        """Organize test files"""
        print("📁 Organizing test files...")
        
        tests_dir = self.workspace / "tests"
        tests_dir.mkdir(exist_ok=True)
        
        test_patterns = ["test_*.py"]
        moved_count = 0
        
        for pattern in test_patterns:
            for test_file in self.workspace.glob(pattern):
                if test_file.is_file():
                    destination = tests_dir / test_file.name
                    if not destination.exists():
                        shutil.move(str(test_file), str(destination))
                        moved_count += 1
                        
        print(f"✅ Organized {moved_count} test files")
        
    def clean_pycache(self):
        """Remove __pycache__ directories"""
        print("🧹 Cleaning __pycache__ directories...")
        
        removed_count = 0
        for pycache_dir in self.workspace.rglob("__pycache__"):
            if pycache_dir.is_dir():
                shutil.rmtree(pycache_dir)
                removed_count += 1
                
        print(f"✅ Removed {removed_count} __pycache__ directories")
        
    def create_workspace_structure_info(self):
        """Create a file documenting the workspace structure"""
        structure_info = """
# Crypto Trading System Workspace Structure

## Main Directories

### /trading_bot/ - Core Trading System
- ai_models/ - AI agents and orchestrator
- strategies/ - Trading strategies  
- coordinators/ - System coordination
- data/ - Data management
- risk/ - Risk management
- execution/ - Trade execution
- analytics/ - Performance analytics

### /data/ - Data Storage
- Historical market data
- Model files
- Training datasets
- Performance metrics

### /reports/ - Generated Reports
- monthly/ - Monthly financial reports
- yearly/ - Annual reports and tax documents
- tax/ - Tax-specific reports (Form 8949, etc.)

### /scripts/ - Utility Scripts
- analysis/ - Performance analysis scripts
- training/ - Model training scripts  
- utilities/ - General utility scripts

### /tests/ - Test Suite
- Unit tests
- Integration tests
- System tests

### /logs/ - System Logs
- Application logs
- Error logs
- Trading activity logs

### /archive/ - Archived Files
- old_logs/ - Historical log files
- deprecated/ - Deprecated code files
- old_files/ - Legacy files

## Main Entry Points

- run_crypto_system.py - Main system launcher
- main.py - Legacy main file
- real_time_crypto_analysis.py - Standalone analysis system

## Configuration

- config.json - System configuration
- .env - Environment variables
- requirements.txt - Python dependencies

## Documentation

- README.md - Project overview
- REAL_TIME_CRYPTO_SYSTEM.md - System documentation
- Various enhancement and optimization summaries
"""
        
        with open(self.workspace / "WORKSPACE_STRUCTURE.md", "w", encoding='utf-8') as f:
            f.write(structure_info)
            
        print("Created workspace structure documentation")
        
    def run_full_organization(self):
        """Run complete workspace organization"""
        print("\n" + "Starting Workspace Organization...")
        print("="*60)
        
        try:
            self.move_old_logs()
            self.move_deprecated_files() 
            self.move_data_files()
            self.organize_scripts()
            self.organize_tests()
            self.clean_pycache()
            self.create_workspace_structure_info()
            
            print("\n" + "="*60)
            print("Workspace organization completed successfully!")
            print("="*60)
            
            # Print summary
            self.print_organization_summary()
            
        except Exception as e:
            print(f"Error during organization: {e}")
            
    def print_organization_summary(self):
        """Print organization summary"""
        print("\n📊 ORGANIZATION SUMMARY")
        print("-" * 30)
        
        # Count files in each directory
        directories = {
            "Archive (old_logs)": self.archive_dir / "old_logs",
            "Archive (deprecated)": self.archive_dir / "deprecated", 
            "Data": self.workspace / "data",
            "Scripts (analysis)": self.scripts_dir / "analysis",
            "Scripts (training)": self.scripts_dir / "training",
            "Scripts (utilities)": self.scripts_dir / "utilities",
            "Tests": self.workspace / "tests",
            "Reports": self.reports_dir
        }
        
        for name, path in directories.items():
            if path.exists():
                file_count = len(list(path.glob("*")))
                print(f"{name}: {file_count} files")
                
        print(f"\n📁 Total directories created: {len(directories)}")
        print("🎯 Workspace is now organized and ready!")

if __name__ == "__main__":
    organizer = WorkspaceOrganizer()
    organizer.run_full_organization()
