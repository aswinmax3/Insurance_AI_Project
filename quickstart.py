#!/usr/bin/env python
"""
Insurance AI - Quick Start Setup Script
Automates data download and model training
"""

import os
import sys
import subprocess
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def run_command(cmd, description=""):
    """Run shell command and handle output"""
    try:
        if description:
            print_info(description)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            if description:
                print_success(description)
            return True
        else:
            print_error(f"Command failed: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
            if result.stderr:
                print(f"  Error: {result.stderr}")
            return False
    except Exception as e:
        print_error(f"Error executing command: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        sys.exit(1)
    print_success(f"Python {version.major}.{version.minor}.{version.micro}")

def install_requirements():
    """Install Python dependencies"""
    print_header("STEP 1: Installing Dependencies")
    
    # Check if requirements.txt exists
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        print_error(f"{req_file} not found!")
        return False
    
    print_info("Installing packages from requirements.txt...")
    if run_command(f"pip install -r {req_file}"):
        print_success("Dependencies installed")
        return True
    else:
        print_error("Failed to install dependencies")
        return False

def download_datasets():
    """Download insurance datasets"""
    print_header("STEP 2: Downloading Datasets")
    
    ml_dir = "analyzer/ml"
    if not os.path.exists(ml_dir):
        print_error(f"Directory {ml_dir} not found!")
        return False
    
    # Check if download_datasets.py exists
    download_script = os.path.join(ml_dir, "download_datasets.py")
    if not os.path.exists(download_script):
        print_error("download_datasets.py not found!")
        return False
    
    print_info("Downloading datasets...")
    print("Options:")
    print("  1. GitHub (Recommended - no authentication)")
    print("  2. Kaggle (Requires API setup)")
    print("  3. Synthetic (Generate sample data)")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        cmd = f"cd {ml_dir} && python download_datasets.py --download github"
        if run_command(cmd, "Downloading from GitHub..."):
            print_success("GitHub datasets downloaded")
            return True
    elif choice == '2':
        print_info("Kaggle API requires setup. Visit:")
        print("  https://www.kaggle.com/settings/account")
        print("  Then run: python download_datasets.py --download kaggle")
        return True
    elif choice == '3':
        cmd = f"cd {ml_dir} && python download_datasets.py --download synthetic"
        if run_command(cmd, "Generating synthetic data..."):
            print_success("Synthetic data generated")
            return True
    else:
        print_error("Invalid selection")
        return False
    
    return False

def train_models():
    """Train ML models"""
    print_header("STEP 3: Training ML Models")
    
    ml_dir = "analyzer/ml"
    train_script = os.path.join(ml_dir, "train_comprehensive_models.py")
    
    if not os.path.exists(train_script):
        print_error("train_comprehensive_models.py not found!")
        return False
    
    print_info("This may take a few minutes...")
    cmd = f"cd {ml_dir} && python train_comprehensive_models.py"
    
    if run_command(cmd, "Training premium prediction and recommendation models..."):
        print_success("Models trained successfully")
        return True
    else:
        print_error("Model training failed")
        return False

def test_recommendations():
    """Test recommendation engine"""
    print_header("STEP 4: Testing Recommendation Engine")
    
    ml_dir = "analyzer/ml"
    test_script = os.path.join(ml_dir, "advanced_recommender.py")
    
    if not os.path.exists(test_script):
        print_error("advanced_recommender.py not found!")
        return False
    
    print_info("Running recommendation engine test...")
    cmd = f"cd {ml_dir} && python advanced_recommender.py"
    
    if run_command(cmd, "Testing recommendations..."):
        print_success("Recommendation engine working correctly")
        return True
    else:
        print_error("Recommendation test failed")
        return False

def create_directories():
    """Create necessary directories"""
    ml_dir = "analyzer/ml"
    
    dirs = [
        os.path.join(ml_dir, "datasets"),
        os.path.join(ml_dir, "models"),
        os.path.join(ml_dir, "reports"),
        os.path.join(ml_dir, "reports/plots"),
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print_success(f"Directory ready: {dir_path}")

def main():
    """Main setup wizard"""
    
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║          Insurance AI - Quick Start Setup                         ║
    ║  Train ML models for intelligent insurance recommendations        ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    print(Colors.END)
    
    print_header("PRE-FLIGHT CHECKS")
    
    # Check Python version
    print_info("Checking Python version...")
    check_python_version()
    
    # Create directories
    print_info("Creating necessary directories...")
    create_directories()
    
    print("\n")
    
    # Installation steps
    steps = [
        ("Install Dependencies", install_requirements),
        ("Download Datasets", download_datasets),
        ("Train Models", train_models),
        ("Test Recommendations", test_recommendations),
    ]
    
    completed = 0
    for i, (step_name, step_func) in enumerate(steps, 1):
        print(f"\n{Colors.YELLOW}[Step {i}/{len(steps)}]{Colors.END}")
        
        if step_func():
            completed += 1
        else:
            print_error(f"Setup incomplete at: {step_name}")
            print_info("You can continue manually or restart this script")
            break
    
    # Final summary
    print_header("SETUP SUMMARY")
    
    if completed == len(steps):
        print_success("All setup steps completed!")
        print(f"\n{Colors.GREEN}{Colors.BOLD}Your Insurance AI system is ready!{Colors.END}\n")
        
        print("Next steps:")
        print("  1. Check the TRAINING_GUIDE.md for detailed documentation")
        print("  2. Review model performance in: analyzer/ml/reports/")
        print("  3. Integrate with Django views (see TRAINING_GUIDE.md)")
        print("  4. Test the recommendation API in your Django app")
        print()
    else:
        print(f"Setup completed {completed}/{len(steps)} steps")
        print("\nTo continue manually:")
        print(f"  cd {os.path.join('analyzer', 'ml')}")
        print("  python download_datasets.py --download github")
        print("  python train_comprehensive_models.py")
        print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Setup cancelled by user{Colors.END}\n")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
