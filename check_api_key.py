"""
Quick script to verify GEMINI_API_KEY is properly configured.
Run: python -m gem_core.check_api_key
"""
import os
import dotenv

dotenv.load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")

print("=" * 60)
print("GEMINI_API_KEY Configuration Check")
print("=" * 60)

if not api_key:
    print("❌ ERROR: GEMINI_API_KEY is not set!")
    print("\nTo fix this:")
    print("1. Open your .env file in the project root")
    print("2. Add this line:")
    print("   GEMINI_API_KEY=your_api_key_here")
    print("\n3. Get your API key from:")
    print("   https://makersuite.google.com/app/apikey")
    print("   or")
    print("   https://aistudio.google.com/app/apikey")
    exit(1)

print(f"✅ GEMINI_API_KEY is set")
print(f"   Length: {len(api_key)} characters")
print(f"   Starts with: {api_key[:5]}...")

# Basic validation
if not api_key.startswith(("AIza", "AI")):
    print("⚠️  WARNING: API key format looks unusual.")
    print("   Expected format: AIza... (usually 39 characters)")
else:
    print("✅ API key format looks correct")

if len(api_key) < 20:
    print("⚠️  WARNING: API key seems too short (expected ~39 characters)")

print("\n" + "=" * 60)
print("To test if the key works, try:")
print("  python -c 'from gem_core.gem import Gem; g = Gem(); print(g.ask(\"Hello\"))'")
print("=" * 60)
