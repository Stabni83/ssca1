# main.py
import sys
import os

# اضافه کردن پوشه src به مسیر
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

try:
    from pipeline import multiscale_ssca
    print("✅ همه importها موفقیت‌آمیز بود")
except ImportError as e:
    print(f"❌ خطای import: {e}")
    exit()

if __name__ == "__main__":
    multiscale_ssca()