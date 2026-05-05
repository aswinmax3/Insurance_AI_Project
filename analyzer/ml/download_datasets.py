"""
Download insurance datasets from Kaggle and other open sources
Supports multiple data sources for comprehensive training
"""

import os
import sys
import requests
import pandas as pd
import zipfile
from pathlib import Path
import json

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")


class DatasetDownloader:
    """Download insurance datasets from various sources"""
    
    # Kaggle datasets (requires kaggle API setup)
    KAGGLE_DATASETS = {
        'insurance': 'mirichoi0814/insurance',  # Medical Insurance
        'claim_prediction': 'buntyshah/health-insurance-cross-sell-prediction',
        'customer_lifetime_value': 'harlfoxem/customerlifetimevalue',
    }
    
    # Direct download URLs from GitHub and other sources
    GITHUB_DATASETS = {
        'insurance_data': 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-Python/master/data/insurance.csv',
    }
    
    def __init__(self):
        os.makedirs(DATASETS_DIR, exist_ok=True)
    
    def download_from_kaggle(self, dataset_name: str = 'insurance'):
        """
        Download dataset from Kaggle
        Requires: kaggle API setup (pip install kaggle)
        
        Setup instructions:
        1. Visit https://www.kaggle.com/settings/account
        2. Click 'Create New API Token'
        3. Place kaggle.json in ~/.kaggle/
        4. chmod 600 ~/.kaggle/kaggle.json (Linux/Mac)
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            api = KaggleApi()
            api.authenticate()
            
            dataset_id = self.KAGGLE_DATASETS.get(dataset_name)
            if not dataset_id:
                print(f"✗ Unknown dataset: {dataset_name}")
                return False
            
            print(f"Downloading {dataset_name} from Kaggle...")
            api.dataset_download_files(
                dataset_id,
                path=DATASETS_DIR,
                unzip=True
            )
            
            print(f"✓ Downloaded {dataset_name}")
            return True
            
        except ImportError:
            print("✗ Kaggle API not installed. Install with: pip install kaggle")
            return False
        except Exception as e:
            print(f"✗ Kaggle download failed: {e}")
            return False
    
    def download_from_github(self, dataset_name: str = 'insurance_data'):
        """Download dataset from GitHub raw URLs"""
        try:
            url = self.GITHUB_DATASETS.get(dataset_name)
            if not url:
                print(f"✗ Unknown dataset: {dataset_name}")
                return False
            
            print(f"Downloading {dataset_name} from GitHub...")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            filename = os.path.join(DATASETS_DIR, f"{dataset_name}.csv")
            with open(filename, 'w') as f:
                f.write(response.text)
            
            print(f"✓ Downloaded {dataset_name} to {filename}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Download failed: {e}")
            return False
    
    def download_claim_data(self):
        """Download health insurance claim dataset"""
        try:
            url = "https://raw.githubusercontent.com/datasets/insurance-claims/master/data/claims.csv"
            print("Downloading health insurance claim data...")
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            filename = os.path.join(DATASETS_DIR, "claim_data.csv")
            with open(filename, 'w') as f:
                f.write(response.text)
            
            print(f"✓ Claim data downloaded")
            return True
            
        except Exception as e:
            print(f"⚠ Could not download claim data: {e}")
            return False
    
    def create_synthetic_data(self):
        """Generate synthetic insurance data if downloads fail"""
        import random
        
        print("Generating synthetic insurance dataset...")
        
        companies = ['HDFC Life', 'ICICI Prudential', 'SBI Life', 'LIC', 
                    'Bajaj Life', 'Max Life', 'Aditya Birla', 'Reliance']
        policy_types = ['Term Life', 'Endowment', 'ULIP', 'Health', 'Critical Illness', 'Investment', 'Whole Life', 'Pension']
        regions = ['North', 'South', 'East', 'West', 'Northeast']
        
        data = []
        for i in range(200):
            data.append({
                'policy_id': f'POL{i+1:04d}',
                'company': random.choice(companies),
                'policy_type': random.choice(policy_types),
                'age': random.randint(18, 70),
                'gender': random.choice(['male', 'female']),
                'bmi': round(random.uniform(18.5, 35.0), 1),
                'smoker': random.choice(['yes', 'no']),
                'health_score': random.randint(50, 100),
                'income': random.randint(250000, 2000000),
                'family_size': random.randint(1, 6),
                'coverage_amount': random.choice([500000, 1000000, 2000000, 3500000, 5000000, 8000000]),
                'premium': random.randint(8000, 80000),
                'exclusions_count': random.randint(0, 6),
                'waiting_period_months': random.randint(0, 24),
                'claim_settlement_days': random.randint(7, 45),
                'rating': round(random.uniform(3.0, 5.0), 1),
                'region': random.choice(regions)
            })
        
        df = pd.DataFrame(data)
        filepath = os.path.join(DATASETS_DIR, "synthetic_insurance_data.csv")
        df.to_csv(filepath, index=False)
        
        print(f"✓ Generated {len(df)} synthetic records")
        print(f"  Saved to: {filepath}")
        
        return filepath
    
    def list_available_datasets(self):
        """List all available datasets in the datasets directory"""
        if not os.path.exists(DATASETS_DIR):
            print("✗ Datasets directory not found")
            return []
        
        datasets = [f for f in os.listdir(DATASETS_DIR) if f.endswith('.csv')]
        
        print("\nAvailable Datasets:")
        print("-" * 50)
        for dataset in datasets:
            filepath = os.path.join(DATASETS_DIR, dataset)
            try:
                df = pd.read_csv(filepath)
                size = os.path.getsize(filepath) / 1024  # KB
                print(f"  • {dataset}")
                print(f"    Records: {len(df)}, Columns: {len(df.columns)}, Size: {size:.1f} KB")
            except Exception as e:
                print(f"  • {dataset} (Error reading file)")
        
        return datasets


def setup_kaggle_api():
    """
    Interactive setup for Kaggle API
    """
    print("\n" + "="*70)
    print("KAGGLE API SETUP")
    print("="*70 + "\n")
    
    print("To use Kaggle datasets, you need to set up the Kaggle API:\n")
    print("1. Go to: https://www.kaggle.com/settings/account")
    print("2. Click 'Create New API Token'")
    print("3. This downloads kaggle.json")
    print("4. Place it in:")
    print(f"   Windows: C:\\Users\\<YourUsername>\\.kaggle\\kaggle.json")
    print(f"   Linux/Mac: ~/.kaggle/kaggle.json")
    print("5. Set permissions: chmod 600 ~/.kaggle/kaggle.json (Linux/Mac only)")
    print("\nOnce setup, you can download datasets using this tool.\n")


def main():
    """Main entry point for dataset management"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download and manage insurance datasets"
    )
    parser.add_argument(
        '--download',
        choices=['kaggle', 'github', 'claim', 'synthetic', 'all'],
        help='Download specific dataset'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets'
    )
    parser.add_argument(
        '--setup-kaggle',
        action='store_true',
        help='Show Kaggle API setup instructions'
    )
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader()
    
    if args.setup_kaggle:
        setup_kaggle_api()
        return
    
    if args.list:
        downloader.list_available_datasets()
        return
    
    if args.download:
        print("\n" + "="*70)
        print("INSURANCE DATASET DOWNLOADER")
        print("="*70 + "\n")
        
        if args.download == 'kaggle':
            downloader.download_from_kaggle('insurance')
            downloader.download_from_kaggle('claim_prediction')
        
        elif args.download == 'github':
            downloader.download_from_github('insurance_data')
        
        elif args.download == 'claim':
            downloader.download_claim_data()
        
        elif args.download == 'synthetic':
            downloader.create_synthetic_data()
        
        elif args.download == 'all':
            print("Downloading all datasets...\n")
            downloader.download_from_github('insurance_data')
            downloader.download_claim_data()
            downloader.create_synthetic_data()
            print("\nAttempting Kaggle downloads...")
            downloader.download_from_kaggle('insurance')
            downloader.download_from_kaggle('claim_prediction')
        
        print("\n" + "="*70)
        print("Download Summary")
        print("="*70)
        downloader.list_available_datasets()
        print()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
