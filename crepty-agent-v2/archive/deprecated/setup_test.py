#!/usr/bin/env python3
"""
Quick Setup & Test Script - Validate enhanced trading agent components.
"""
import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_file_exists(file_path, description):
    """Check if a file exists and report status"""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (MISSING)")
        return False

def test_import(module_name, description):
    """Test if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {description}: Import successful")
        return True
    except ImportError as e:
        print(f"❌ {description}: Import failed - {e}")
        return False
    except Exception as e:
        print(f"⚠️ {description}: Import warning - {e}")
        return False

def run_quick_test(script_name, description):
    """Run a quick test script"""
    if os.path.exists(script_name):
        print(f"\n🧪 Testing {description}...")
        try:
            result = subprocess.run([sys.executable, script_name, "--help"], 
                                 capture_output=True, text=True, timeout=10)
            if result.returncode == 0 or "usage:" in result.stdout.lower():
                print(f"✅ {description}: Script is executable")
                return True
            else:
                print(f"⚠️ {description}: Script runs but may have issues")
                return False
        except subprocess.TimeoutExpired:
            print(f"⚠️ {description}: Script timeout (may be working)")
            return False
        except Exception as e:
            print(f"❌ {description}: {e}")
            return False
    else:
        print(f"❌ {description}: Script not found")
        return False

def main():
    """Main setup and test function"""
    print("🚀 ENHANCED TRADING AGENT - SETUP & TEST")
    print("="*50)
    
    print("\n📁 FILE STRUCTURE CHECK")
    print("-" * 30)
    
    # Core files
    core_files = [
        ("main.py", "Main trading script"),
        ("requirements.txt", "Dependencies"),
        ("trading_bot/__init__.py", "Trading bot package"),
        ("trading_bot/agents/manager_agent.py", "Manager agent"),
    ]
    
    # Enhanced files (recently added/modified)
    enhanced_files = [
        ("trading_bot/ai_models/ml_signals.py", "Enhanced ML signals"),
        ("trading_bot/risk/risk_manager.py", "Enhanced risk manager"),
        ("trading_bot/utils/websocket_client.py", "WebSocket client"),
        ("trading_bot/utils/portfolio_optimizer.py", "Portfolio optimizer"),
        ("train_ml_model.py", "ML training script"),
        ("performance_monitor.py", "Performance monitor"),
    ]
    
    all_good = True
    
    for file_path, description in core_files + enhanced_files:
        if not check_file_exists(file_path, description):
            all_good = False
    
    print(f"\n📦 PYTHON IMPORTS CHECK")
    print("-" * 30)
    
    # Test critical imports
    critical_imports = [
        ("trading_bot", "Trading bot package"),
        ("trading_bot.agents.manager_agent", "Manager agent"),
        ("trading_bot.ai_models.ml_signals", "ML signals"),
        ("trading_bot.risk.risk_manager", "Risk manager"),
        ("trading_bot.utils.portfolio_optimizer", "Portfolio optimizer"),
    ]
    
    for module, description in critical_imports:
        if not test_import(module, description):
            all_good = False
    
    print(f"\n🔬 DEPENDENCIES CHECK")
    print("-" * 30)
    
    # Test key dependencies
    key_deps = [
        ("pandas", "Data processing"),
        ("numpy", "Numerical computing"),
        ("sklearn", "Machine learning"),
        ("websockets", "WebSocket client"),
        ("loguru", "Logging"),
        ("pydantic", "Data validation"),
    ]
    
    for dep, description in key_deps:
        test_import(dep, description)
    
    print(f"\n⚡ QUICK FUNCTIONALITY TESTS")
    print("-" * 30)
    
    # Test key scripts
    test_scripts = [
        ("train_ml_model.py", "ML model training"),
        ("performance_monitor.py", "Performance monitoring"),
    ]
    
    for script, description in test_scripts:
        run_quick_test(script, description)
    
    print(f"\n📈 TRADING DATA CHECK")
    print("-" * 30)
    
    # Check for trading data files
    data_files = [
        "trade_log.csv",
        "futures_trades_log.csv", 
        "BTCUSDT_1h.csv",
        "synthetic_meta_training.csv"
    ]
    
    data_found = False
    for file in data_files:
        if os.path.exists(file):
            print(f"✅ Found trading data: {file}")
            data_found = True
        
    if not data_found:
        print("⚠️ No trading data files found")
        print("   Run the trading agent to generate data, or use synthetic data")
    
    print(f"\n🎯 NEXT STEPS")
    print("-" * 30)
    
    if all_good:
        print("✅ Setup looks good! Here's what you can do:")
        print()
        print("1. 🧠 Train ML model:")
        print("   python train_ml_model.py")
        print()
        print("2. 📊 Check performance:")
        print("   python performance_monitor.py")
        print()
        print("3. 🚀 Start enhanced trading:")
        print("   python main.py")
        print()
        print("4. 📈 Monitor live performance:")
        print("   python performance_monitor.py (run periodically)")
        
    else:
        print("⚠️ Some issues found. Please check the errors above.")
        print()
        print("Common fixes:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Check file paths and permissions")
        print("3. Ensure all enhanced components are in place")
    
    print(f"\n💡 ENHANCEMENT SUMMARY")
    print("-" * 30)
    print("🔬 Enhanced ML Signals: RandomForest with technical indicators")
    print("🛡️ Advanced Risk Manager: Portfolio optimization & correlation limits")
    print("📡 WebSocket Client: Real-time market data streaming")
    print("⚖️ Portfolio Optimizer: Modern portfolio theory & Sharpe ratio")
    print("🧠 ML Training: Automated model training with synthetic data")
    print("📊 Performance Monitor: Real-time trading metrics")
    
    print("\n🎉 Enhanced trading agent ready for testing!")

if __name__ == "__main__":
    main()
