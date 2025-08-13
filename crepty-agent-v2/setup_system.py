#!/usr/bin/env python3
"""
Crypto Trading System Setup Script
Installs dependencies, organizes workspace, and prepares the system
"""

import subprocess
import sys
import os
from pathlib import Path
import json

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} is compatible")

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)

def setup_environment():
    """Setup environment variables"""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("📝 Creating .env file template...")
        
        env_template = """# Crypto Trading System Environment Variables

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Binance API Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here

# Optional: Binance Testnet (set to true for testing)
BINANCE_TESTNET=false

# Optional: Email Configuration for Reports
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

# Optional: Database Configuration
DATABASE_URL=sqlite:///crypto_trading.db

# Optional: Telegram Bot (for notifications)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
"""
        
        with open(env_file, "w") as f:
            f.write(env_template)
            
        print("✅ .env file created")
        print("⚠️  Please edit .env file with your API keys!")
    else:
        print("✅ .env file already exists")

def organize_workspace():
    """Run workspace organization"""
    print("📁 Organizing workspace...")
    
    try:
        from organize_workspace import WorkspaceOrganizer
        organizer = WorkspaceOrganizer()
        organizer.run_full_organization()
    except Exception as e:
        print(f"❌ Workspace organization failed: {e}")

def create_startup_script():
    """Create startup script for different platforms"""
    
    # Windows batch file
    windows_script = """@echo off
echo Starting Crypto Trading System...
python run_crypto_system.py
pause
"""
    
    with open("start_crypto_system.bat", "w") as f:
        f.write(windows_script)
        
    # Unix shell script
    unix_script = """#!/bin/bash
echo "Starting Crypto Trading System..."
python3 run_crypto_system.py
"""
    
    with open("start_crypto_system.sh", "w") as f:
        f.write(unix_script)
        
    # Make shell script executable on Unix systems
    if os.name != 'nt':
        os.chmod("start_crypto_system.sh", 0o755)
        
    print("✅ Startup scripts created")

def verify_setup():
    """Verify that setup is complete"""
    print("\n🔍 Verifying setup...")
    
    issues = []
    
    # Check required files
    required_files = [
        "run_crypto_system.py",
        "config.json", 
        ".env",
        "requirements.txt"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            issues.append(f"Missing file: {file_path}")
            
    # Check environment variables
    env_vars = ["OPENAI_API_KEY", "BINANCE_API_KEY", "BINANCE_SECRET_KEY"]
    for var in env_vars:
        if not os.getenv(var):
            issues.append(f"Missing environment variable: {var}")
            
    if issues:
        print("⚠️  Setup issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease resolve these issues before running the system.")
    else:
        print("✅ Setup verification passed!")

def print_final_instructions():
    """Print final setup instructions"""
    print("\n" + "="*60)
    print("🎉 CRYPTO TRADING SYSTEM SETUP COMPLETE!")
    print("="*60)
    
    print("\n📋 NEXT STEPS:")
    print("1. Edit .env file with your API keys")
    print("2. Review config.json settings")
    print("3. Run the system using one of these methods:")
    print("   • Windows: double-click start_crypto_system.bat")
    print("   • Unix/Mac: ./start_crypto_system.sh")
    print("   • Direct: python run_crypto_system.py")
    
    print("\n🚀 SYSTEM FEATURES:")
    print("• 8 AI Expert Agents (Crypto, Financial, Broker, Risk, etc.)")
    print("• Real-time market data collection")
    print("• 4 Trading strategies with meta-learning")
    print("• Automated monthly/yearly reports")
    print("• Tax report generation (Form 8949)")
    print("• Risk management and position sizing")
    
    print("\n📁 WORKSPACE ORGANIZATION:")
    print("• /trading_bot/ - Core system")
    print("• /reports/ - Generated reports")
    print("• /data/ - Market data and models")
    print("• /logs/ - System logs")
    print("• /archive/ - Old/deprecated files")
    
    print("\n⚙️ CONFIGURATION:")
    print("• config.json - System settings")
    print("• .env - API keys and secrets")
    
    print("\n📖 DOCUMENTATION:")
    print("• WORKSPACE_STRUCTURE.md - Workspace overview")
    print("• REAL_TIME_CRYPTO_SYSTEM.md - System documentation")
    
    print("="*60)

def main():
    """Main setup function"""
    print("🚀 Crypto Trading System Setup")
    print("="*40)
    
    try:
        check_python_version()
        install_dependencies()
        setup_environment()
        organize_workspace()
        create_startup_script()
        verify_setup()
        print_final_instructions()
        
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
