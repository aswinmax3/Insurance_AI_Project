"""
Download Real Insurance PDF Documents from Open Sources
Supports multiple public datasets and repositories
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from urllib.parse import urljoin
import json
from datetime import datetime

PDFs_DIR = os.path.join(os.path.dirname(__file__), "raw_pdfs")
os.makedirs(PDFs_DIR, exist_ok=True)


class InsurancePDFDownloader:
    """Download real insurance documents from open sources"""
    
    # Open source repositories with insurance documents
    SOURCES = {
        # Health Insurance Claims Dataset
        'health_insurance_claims': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/insurance/insurance.data',
            'type': 'csv',
            'description': 'Health insurance claims with medical charges'
        },
        
        # Insurance Policy Documents
        'sample_policies': {
            'url': 'https://www.soa.org/globalassets/assets/files/research/exp-study/research-2011-mortalityresearch.zip',
            'type': 'zip',
            'description': 'Mortality research and insurance policy data'
        },
        
        # NHTSA Insurance Data
        'vehicle_insurance': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto/imports-85.data',
            'type': 'data',
            'description': 'Vehicle insurance and claim data'
        },
        
        # Indian Insurance Dataset (IRDA)
        'indian_insurance': {
            'url': 'https://data.gov.in/api/datastore/sql',
            'type': 'api',
            'description': 'Indian insurance market data'
        },
        
        # Kaggle Insurance Datasets
        'kaggle_medical': {
            'dataset_id': 'mirichoi0814/insurance',
            'type': 'kaggle',
            'description': 'Medical insurance with charges'
        },
        
        'kaggle_claims': {
            'dataset_id': 'anoopsmohan/insurance-data',
            'type': 'kaggle',
            'description': 'Insurance claim history'
        },
        
        'kaggle_health_cross_sell': {
            'dataset_id': 'buntyshah/health-insurance-cross-sell-prediction',
            'type': 'kaggle',
            'description': 'Health insurance cross-sell prediction'
        }
    }
    
    def __init__(self):
        self.log = []
        self.downloaded_files = []
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for downloads"""
        self.log_file = os.path.join(PDFs_DIR, f"download_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    def _log(self, message, level="INFO"):
        """Log download activity"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message
        }
        self.log.append(log_entry)
        print(f"[{level}] {message}")
    
    def download_from_uci(self):
        """Download from UCI Machine Learning Repository"""
        self._log("Downloading from UCI ML Repository...")
        
        datasets = {
            'insurance': 'https://archive.ics.uci.edu/ml/machine-learning-databases/insurance/insurance.data',
            'auto_imports': 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto/imports-85.data',
        }
        
        for name, url in datasets.items():
            try:
                self._log(f"Fetching {name}...")
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                
                filepath = os.path.join(PDFs_DIR, f"uci_{name}.csv")
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                self.downloaded_files.append(filepath)
                self._log(f"✓ Downloaded {name}: {filepath}", "SUCCESS")
                
            except Exception as e:
                self._log(f"✗ Failed to download {name}: {e}", "ERROR")
    
    def download_from_kaggle_api(self):
        """Download from Kaggle using API"""
        self._log("Downloading from Kaggle...")
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            api = KaggleApi()
            api.authenticate()
            
            kaggle_datasets = {
                'mirichoi0814/insurance': 'medical_insurance',
                'anoopsmohan/insurance-data': 'insurance_data',
                'buntyshah/health-insurance-cross-sell-prediction': 'health_cross_sell'
            }
            
            for dataset_id, name in kaggle_datasets.items():
                try:
                    self._log(f"Downloading {name} from Kaggle...")
                    output_path = os.path.join(PDFs_DIR, f"kaggle_{name}")
                    os.makedirs(output_path, exist_ok=True)
                    
                    api.dataset_download_files(
                        dataset_id,
                        path=output_path,
                        unzip=True
                    )
                    
                    self.downloaded_files.append(output_path)
                    self._log(f"✓ Downloaded Kaggle dataset: {name}", "SUCCESS")
                    
                except Exception as e:
                    self._log(f"⚠ Could not download {name}: {e}", "WARNING")
        
        except ImportError:
            self._log("Kaggle API not installed. Install with: pip install kaggle", "WARNING")
    
    def download_government_data(self):
        """Download from government open data portals"""
        self._log("Downloading from government data portals...")
        
        # India's open data portal
        try:
            self._log("Fetching data from India's open data portal...")
            
            # Example: Insurance statistics dataset
            url = "https://data.gov.in/sites/default/files/insurance_statistics.csv"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                filepath = os.path.join(PDFs_DIR, "india_insurance_stats.csv")
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                self.downloaded_files.append(filepath)
                self._log(f"✓ Downloaded government data", "SUCCESS")
            
        except Exception as e:
            self._log(f"⚠ Could not fetch government data: {e}", "WARNING")
    
    def download_github_datasets(self):
        """Download prepared datasets from GitHub"""
        self._log("Downloading from GitHub repositories...")
        
        github_urls = [
            {
                'name': 'insurance_dataset',
                'url': 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-Python/master/data/insurance.csv',
                'description': 'Medical insurance dataset'
            },
            {
                'name': 'insurance_claims',
                'url': 'https://raw.githubusercontent.com/datasets/insurance-claims/master/data/claims.csv',
                'description': 'Insurance claims dataset'
            }
        ]
        
        for dataset in github_urls:
            try:
                self._log(f"Downloading {dataset['name']}...")
                response = requests.get(dataset['url'], timeout=15)
                response.raise_for_status()
                
                filepath = os.path.join(PDFs_DIR, f"github_{dataset['name']}.csv")
                with open(filepath, 'w') as f:
                    f.write(response.text)
                
                self.downloaded_files.append(filepath)
                self._log(f"✓ Downloaded {dataset['name']}", "SUCCESS")
                
            except Exception as e:
                self._log(f"⚠ Failed to download {dataset['name']}: {e}", "WARNING")
    
    def download_sample_pdfs(self):
        """Download sample insurance PDFs and forms"""
        self._log("Downloading sample insurance documents...")
        
        # Sample insurance policy documents
        pdf_urls = [
            {
                'name': 'sample_policy.pdf',
                'url': 'https://www.soa.org/globalassets/assets/files/research/exp-study/research-2011-mortalityresearch.pdf',
                'description': 'Sample insurance policy'
            },
            {
                'name': 'insurance_guide.pdf',
                'url': 'https://www.iii.org/sites/default/files/docs/pdf/Insurance101.pdf',
                'description': 'Insurance guide and terminology'
            }
        ]
        
        for pdf_info in pdf_urls:
            try:
                self._log(f"Downloading {pdf_info['name']}...")
                response = requests.get(pdf_info['url'], timeout=15)
                
                if response.status_code == 200:
                    filepath = os.path.join(PDFs_DIR, pdf_info['name'])
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    self.downloaded_files.append(filepath)
                    self._log(f"✓ Downloaded PDF: {pdf_info['name']}", "SUCCESS")
                
            except Exception as e:
                self._log(f"⚠ Could not download {pdf_info['name']}: {e}", "WARNING")
    
    def create_synthetic_enhanced_dataset(self):
        """Create enhanced synthetic dataset based on real patterns"""
        import pandas as pd
        import numpy as np
        
        self._log("Creating enhanced synthetic insurance dataset...")
        
        # Create comprehensive dataset based on real-world patterns
        np.random.seed(42)
        
        n_records = 5000  # More records for better training
        
        data = {
            'policy_id': [f'POL{i:06d}' for i in range(1, n_records + 1)],
            'age': np.random.normal(40, 15, n_records).clip(18, 80).astype(int),
            'gender': np.random.choice(['male', 'female'], n_records),
            'bmi': np.random.normal(27, 5, n_records).clip(15, 45),
            'children': np.random.choice([0, 1, 2, 3, 4], n_records, p=[0.2, 0.3, 0.3, 0.15, 0.05]),
            'smoker': np.random.choice(['yes', 'no'], n_records, p=[0.2, 0.8]),
            'region': np.random.choice(['north', 'south', 'east', 'west'], n_records),
            'premium': np.random.uniform(5000, 100000, n_records),
            'coverage_amount': np.random.choice([500000, 1000000, 2000000, 5000000, 8000000], n_records),
            'health_score': np.random.normal(75, 15, n_records).clip(30, 100).astype(int),
            'claim_history': np.random.choice(['none', 'minor', 'major'], n_records, p=[0.6, 0.3, 0.1]),
            'exclusions_count': np.random.choice([0, 1, 2, 3, 4, 5], n_records, p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.03]),
            'waiting_period_months': np.random.choice([0, 3, 6, 12, 24], n_records, p=[0.3, 0.25, 0.25, 0.15, 0.05]),
            'claim_settlement_days': np.random.normal(20, 10, n_records).clip(5, 60).astype(int),
            'company_rating': np.random.normal(4.0, 0.7, n_records).clip(2.0, 5.0),
            'customer_satisfaction': np.random.normal(4.2, 0.6, n_records).clip(1, 5),
            'claim_settlement_ratio': np.random.uniform(0.7, 0.99, n_records),
            'policy_type': np.random.choice(['Term Life', 'Endowment', 'ULIP', 'Health', 'Critical Illness'], n_records),
            'income': np.random.exponential(500000, n_records).astype(int).clip(100000, 5000000),
            'marital_status': np.random.choice(['single', 'married', 'divorced', 'widowed'], n_records),
            'occupation': np.random.choice(['professional', 'business', 'service', 'labour', 'other'], n_records)
        }
        
        df = pd.DataFrame(data)
        
        # Calculate recommendation score based on real factors
        df['recommendation_score'] = (
            (df['company_rating'] / 5) * 25 +
            (df['customer_satisfaction'] / 5) * 20 +
            (df['claim_settlement_ratio'] * 25) +
            ((100 - df['exclusions_count'] * 5) / 100) * 20 +
            ((100 - df['waiting_period_months']) / 100) * 10
        ).clip(0, 100)
        
        filepath = os.path.join(PDFs_DIR, "synthetic_enhanced_dataset.csv")
        df.to_csv(filepath, index=False)
        self.downloaded_files.append(filepath)
        
        self._log(f"✓ Created enhanced dataset with {n_records} records", "SUCCESS")
        return filepath
    
    def download_all(self):
        """Download all available datasets"""
        print("\n" + "="*70)
        print("INSURANCE DOCUMENT & DATA DOWNLOADER")
        print("="*70 + "\n")
        
        self._log("Starting comprehensive data download...")
        
        # Download from all sources
        self.download_from_uci()
        self.download_from_kaggle_api()
        self.download_government_data()
        self.download_github_datasets()
        self.download_sample_pdfs()
        self.create_synthetic_enhanced_dataset()
        
        # Save log
        with open(self.log_file, 'w') as f:
            json.dump(self.log, f, indent=2)
        
        # Summary
        self._print_summary()
    
    def _print_summary(self):
        """Print download summary"""
        print("\n" + "="*70)
        print("DOWNLOAD SUMMARY")
        print("="*70 + "\n")
        
        print(f"Total files downloaded: {len(self.downloaded_files)}")
        print(f"\nFiles:")
        for i, filepath in enumerate(self.downloaded_files, 1):
            size = os.path.getsize(filepath) / 1024 if os.path.exists(filepath) else 0
            print(f"  {i}. {os.path.basename(filepath)} ({size:.1f} KB)")
        
        print(f"\nLog file: {self.log_file}")
        print(f"Download directory: {PDFs_DIR}\n")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download insurance documents and datasets")
    parser.add_argument('--all', action='store_true', help='Download all sources')
    parser.add_argument('--uci', action='store_true', help='Download from UCI ML Repository')
    parser.add_argument('--kaggle', action='store_true', help='Download from Kaggle')
    parser.add_argument('--github', action='store_true', help='Download from GitHub')
    parser.add_argument('--pdfs', action='store_true', help='Download PDF documents')
    parser.add_argument('--synthetic', action='store_true', help='Create synthetic dataset')
    parser.add_argument('--list', action='store_true', help='List available sources')
    
    args = parser.parse_args()
    
    downloader = InsurancePDFDownloader()
    
    if args.list:
        print("\nAvailable Sources:")
        for name, info in downloader.SOURCES.items():
            print(f"  • {name}: {info['description']}")
        return
    
    if args.all or (not args.uci and not args.kaggle and not args.github and not args.pdfs and not args.synthetic):
        downloader.download_all()
    else:
        if args.uci:
            downloader.download_from_uci()
        if args.kaggle:
            downloader.download_from_kaggle_api()
        if args.github:
            downloader.download_github_datasets()
        if args.pdfs:
            downloader.download_sample_pdfs()
        if args.synthetic:
            downloader.create_synthetic_enhanced_dataset()
        
        downloader._print_summary()


if __name__ == "__main__":
    main()
