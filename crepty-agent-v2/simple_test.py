"""
Simple test for AI enhancements
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    print("Testing imports...")
    
    try:
        from trading_bot.utils.openai_client import OpenAIClient
        print("✅ OpenAI client imported")
    except Exception as e:
        print(f"❌ OpenAI client import failed: {e}")
        return False
    
    try:
        from trading_bot.utils.ai_enhanced_ml_signals import generate_ai_enhanced_ml_signal
        print("✅ AI enhanced ML signals imported")
    except Exception as e:
        print(f"❌ AI enhanced ML signals import failed: {e}")
        return False
    
    try:
        from trading_bot.utils.ai_position_sizer import calculate_ai_position_size
        print("✅ AI position sizer imported")
    except Exception as e:
        print(f"❌ AI position sizer import failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🤖 AI Enhancement Import Test")
    success = test_imports()
    if success:
        print("🎉 All imports successful!")
    else:
        print("❌ Some imports failed")
