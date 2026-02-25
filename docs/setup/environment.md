# Environment Setup

## 🐍 Python Environment

### System Requirements
- Python 3.8+ (recommended 3.9 or 3.10)
- 8GB+ RAM (16GB recommended for ML models)
- 50GB+ disk space (for data and models)

### Core Dependencies
```bash
# Core ML and data stack
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install xgboost>=1.7.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install pyarrow>=12.0.0
pip install scikit-learn>=1.3.0

# Trading and backtesting
pip install backtrader>=1.9.76.123
pip install requests>=2.31.0
pip install websockets>=11.0

# Visualization and monitoring
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install plotly>=5.15.0

# Utilities
pip install python-dotenv>=1.0.0
pip install schedule>=1.2.0
pip install joblib>=1.3.0
```

### GPU Support (Optional but recommended)
```bash
# For CUDA support (NVIDIA GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 📁 Project Structure Setup

### Create Directory Structure
```bash
mkdir -p /Users/apple/finbert-sol/{src,tests,notebooks,logs,models,configs,data}
mkdir -p /Users/apple/finbert-sol/src/{data,models,strategies,backtesting,risk,production,utils}
mkdir -p /Users/apple/finbert-sol/tests/{unit,integration,data}
```

### Environment Variables
Create `.env` file:
```bash
# API Configuration
HYPERLIQUID_API_URL=https://api.hyperliquid.xyz
HYPERLIQUID_API_KEY=your_api_key_here
HYPERLIQUID_SECRET_KEY=your_secret_key_here

# Data Configuration
DATA_ROOT=/Users/apple/finbert-sol/lakehouse
MODEL_ROOT=/Users/apple/finbert-sol/models
LOG_ROOT=/Users/apple/finbert-sol/logs

# Trading Configuration
PAPER_TRADING=true
MAX_POSITION_SIZE=0.2
RISK_LIMIT_DAILY_LOSS=0.05

# Monitoring Configuration
ALERT_EMAIL=your_email@example.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

---

## 🔧 Development Environment

### IDE Setup (VSCode recommended)
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "/Users/apple/finbert-sol/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "jupyter.jupyterServerType": "local"
}
```

### Git Configuration
```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit: SOL trading system setup"

# Create .gitignore
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Data and models
data/raw/
models/*.joblib
logs/*.log

# Secrets
.env
*.key
*.pem

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF
```

---

## 🧪 Testing Environment

### Test Dependencies
```bash
pip install pytest>=7.4.0
pip install pytest-cov>=4.1.0
pip install pytest-mock>=3.11.0
```

### Test Configuration
```python
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = --cov=src --cov-report=html --cov-report=term-missing
```

---

## 📊 Data Validation

### Verify Existing Data
```python
# scripts/validate_data.py
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

def validate_lakehouse_data():
    """Validate existing SOL lakehouse data"""
    lakehouse_path = Path("/Users/apple/finbert-sol/lakehouse")
    
    # Check consolidated data
    consolidated_path = lakehouse_path / "consolidated" / "SOL"
    if consolidated_path.exists():
        print("✅ Found consolidated SOL data")
        
        # List available years/months
        for year_dir in consolidated_path.glob("year=*"):
            year = year_dir.name.split("=")[1]
            months = list(year_dir.glob("month=*"))
            print(f"  Year {year}: {len(months)} months available")
    else:
        print("❌ No consolidated SOL data found")
    
    # Check enhanced data
    enhanced_path = lakehouse_path / "enhanced" / "l2_enhanced" / "SOL"
    if enhanced_path.exists():
        dates = list(enhanced_path.glob("*"))
        print(f"✅ Found enhanced data: {len(dates)} daily files")
    else:
        print("❌ No enhanced SOL data found")

if __name__ == "__main__":
    validate_lakehouse_data()
```

### Run Data Validation
```bash
python scripts/validate_data.py
```

---

## 🚀 Quick Start Script

### Automated Setup
```bash
#!/bin/bash
# setup.sh - Automated environment setup

echo "🚀 Setting up SOL Trading System Environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
echo "📁 Creating directory structure..."
mkdir -p src/{data,models,strategies,backtesting,risk,production,utils}
mkdir -p tests/{unit,integration,data}
mkdir -p notebooks logs models configs

# Set up environment file
if [ ! -f .env ]; then
    echo "⚙️ Creating environment file..."
    cp .env.example .env
    echo "Please edit .env file with your API keys and configuration"
fi

# Validate data
echo "📊 Validating existing data..."
python scripts/validate_data.py

echo "✅ Environment setup complete!"
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: source venv/bin/activate"
echo "3. Start with: python -m pytest tests/"
```

---

## 🔍 Environment Validation

### Health Check Script
```python
# scripts/health_check.py
import sys
import importlib
from pathlib import Path

def check_environment():
    """Check if environment is properly set up"""
    checks = []
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        checks.append(("✅ Python version", f"{python_version.major}.{python_version.minor}.{python_version.micro}"))
    else:
        checks.append(("❌ Python version", f"Need 3.8+, got {python_version.major}.{python_version.minor}"))
    
    # Check critical packages
    packages = ['torch', 'transformers', 'xgboost', 'pandas', 'numpy', 'pyarrow']
    for package in packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            checks.append(("✅", f"{package} ({version})"))
        except ImportError:
            checks.append(("❌", f"{package} not installed"))
    
    # Check directories
    directories = ['src', 'tests', 'notebooks', 'logs', 'models']
    for directory in directories:
        if Path(directory).exists():
            checks.append(("✅", f"Directory {directory} exists"))
        else:
            checks.append(("❌", f"Directory {directory} missing"))
    
    # Check data
    lakehouse_path = Path("/Users/apple/finbert-sol/lakehouse")
    if lakehouse_path.exists():
        checks.append(("✅", "Lakehouse data directory exists"))
    else:
        checks.append(("❌", "Lakehouse data directory missing"))
    
    # Print results
    print("Environment Health Check")
    print("=" * 40)
    for status, item in checks:
        print(f"{status} {item}")
    
    # Summary
    failed = sum(1 for status, _ in checks if status == "❌")
    total = len(checks)
    print(f"\nSummary: {total - failed}/{total} checks passed")
    
    if failed == 0:
        print("🎉 Environment is ready!")
        return True
    else:
        print("⚠️ Please fix the issues above before proceeding")
        return False

if __name__ == "__main__":
    check_environment()
```

---

## 📚 Next Steps

After environment setup:
1. **Validate**: Run `python scripts/health_check.py`
2. **Configure**: Edit `.env` file with API keys
3. **Test**: Run `python -m pytest tests/`
4. **Start**: Begin with [Phase 1: Foundation](../phases/phase-1-foundation.md)

---

## 🚨 Troubleshooting

### Common Issues

#### PyTorch Installation
```bash
# If PyTorch fails to install with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For M1/M2 Macs
pip install torch torchvision torchaudio
```

#### Memory Issues
```bash
# Increase memory limit for pandas
export PYTHONMALLOC=malloc

# Use chunked processing for large datasets
```

#### Permission Issues
```bash
# Fix permissions for logs directory
chmod 755 logs/
chmod 644 logs/*.log
```

---

*Environment setup typically takes 30-60 minutes depending on internet speed and hardware.*
