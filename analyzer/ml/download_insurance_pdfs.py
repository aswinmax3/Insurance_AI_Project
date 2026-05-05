"""
Enhanced Insurance PDF & Data Downloader
Downloads real insurance documents from Kaggle, GitHub, and open sources
Extracts and processes insurance data for ML training
"""

import os
import sys
import requests
import zipfile
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil
from typing import List, Dict, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
RAW_PDFS_DIR = os.path.join(BASE_DIR, "raw_pdfs")
EXTRACTED_DATA_DIR = os.path.join(BASE_DIR, "extracted_data")

os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(RAW_PDFS_DIR, exist_ok=True)
os.makedirs(EXTRACTED_DATA_DIR, exist_ok=True)


class InsurancePDFDownloader:
    """Download real insurance documents from multiple open sources"""
    
    # Open source data repositories
    SOURCES = {
        # 1. UCI Machine Learning Repository
        'uci_insurance': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/insurance/insurance.data',
            'name': 'uci_insurance.csv',
            'type': 'csv',
            'description': 'Health insurance data with medical charges',
            'enabled': True
        },
        
        # 2. Kaggle Insurance Datasets (Raw data links)
        'kaggle_medical_insurance': {
            'url': 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R/master/Chapter%207/insurance.csv',
            'name': 'kaggle_medical.csv',
            'type': 'csv',
            'description': 'Medical insurance charges dataset',
            'enabled': True
        },
        
        # 3. GitHub - Insurance Data
        'github_insurance_claims': {
            'url': 'https://raw.githubusercontent.com/irfanyusantos/insurance_prediction/main/insurance.csv',
            'name': 'github_insurance_claims.csv',
            'type': 'csv',
            'description': 'Insurance claims prediction data',
            'enabled': True
        },
        
        # 4. GitHub - Vehicle Insurance
        'github_vehicle_insurance': {
            'url': 'https://raw.githubusercontent.com/rahul-raoniar/Insurance-data-prediction/main/insurance.csv',
            'name': 'vehicle_insurance.csv',
            'type': 'csv',
            'description': 'Vehicle insurance data',
            'enabled': True
        },
        
        # 5. Data.gov - Insurance Data
        'govt_health_insurance': {
            'url': 'https://raw.githubusercontent.com/fivethirtyeight/data/master/insurance/insurance.csv',
            'name': 'govt_insurance.csv',
            'type': 'csv',
            'description': 'Government health insurance statistics',
            'enabled': True
        },
        
        # 6. Statsmodels - Insurance Dataset
        'statsmodels_insurance': {
            'url': 'https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Insurance.csv',
            'name': 'statsmodels_insurance.csv',
            'type': 'csv',
            'description': 'Insurance claims from statsmodels',
            'enabled': True
        },
        
        # 7. RDatasets - Motor Insurance
        'motor_insurance': {
            'url': 'https://vincentarelbundock.github.io/Rdatasets/csv/insuranceData/AutoClaims.csv',
            'name': 'motor_insurance.csv',
            'type': 'csv',
            'description': 'Motor insurance claims data',
            'enabled': True
        },
        
        # 8. Enhanced Synthetic Dataset (from GitHub)
        'synthetic_insurance': {
            'url': 'https://raw.githubusercontent.com/KeerthanVT/Insurance_Cross_Sell_Prediction/main/train.csv',
            'name': 'synthetic_enhanced.csv',
            'type': 'csv',
            'description': 'Enhanced synthetic insurance data',
            'enabled': True
        }
    }
    
    def __init__(self):
        self.downloaded_files = []
        self.download_log = {
            'timestamp': datetime.now().isoformat(),
            'downloads': [],
            'errors': [],
            'summary': {}
        }
    
    def download_all_sources(self) -> Dict:
        """Download from all available sources"""
        logger.info("🔽 Starting Insurance Data Download from Multiple Sources")
        logger.info(f"Total sources to download: {len([s for s in self.SOURCES.values() if s['enabled']])}")
        
        successful_downloads = 0
        failed_downloads = 0
        total_records = 0
        
        for source_key, source_info in self.SOURCES.items():
            if not source_info['enabled']:
                continue
            
            logger.info(f"\n📥 Downloading: {source_info['description']}")
            logger.info(f"   URL: {source_info['url'][:80]}...")
            
            try:
                if source_info['type'] == 'csv':
                    df = self._download_csv(source_info)
                    if df is not None:
                        records = len(df)
                        total_records += records
                        successful_downloads += 1
                        logger.info(f"   ✓ Downloaded: {records} records")
                        
                        self.download_log['downloads'].append({
                            'source': source_key,
                            'name': source_info['name'],
                            'records': records,
                            'status': 'success',
                            'timestamp': datetime.now().isoformat()
                        })
                    else:
                        failed_downloads += 1
                        logger.warning(f"   ✗ Failed to download {source_key}")
                        self.download_log['errors'].append({
                            'source': source_key,
                            'reason': 'Failed to parse or download'
                        })
            except Exception as e:
                failed_downloads += 1
                logger.error(f"   ✗ Error downloading {source_key}: {str(e)}")
                self.download_log['errors'].append({
                    'source': source_key,
                    'error': str(e)
                })
        
        self.download_log['summary'] = {
            'total_sources': len(self.SOURCES),
            'enabled_sources': len([s for s in self.SOURCES.values() if s['enabled']]),
            'successful': successful_downloads,
            'failed': failed_downloads,
            'total_records_downloaded': total_records
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("📊 Download Summary:")
        logger.info(f"   Successful: {successful_downloads}")
        logger.info(f"   Failed: {failed_downloads}")
        logger.info(f"   Total Records: {total_records:,}")
        logger.info(f"{'='*60}")
        
        self._save_download_log()
        return self.download_log['summary']
    
    def _download_csv(self, source_info: Dict) -> pd.DataFrame:
        """Download and parse CSV file"""
        try:
            response = requests.get(source_info['url'], timeout=30)
            response.raise_for_status()
            
            # Save raw file
            file_path = os.path.join(DATASETS_DIR, source_info['name'])
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Parse CSV
            df = pd.read_csv(file_path)
            self.downloaded_files.append(file_path)
            return df
        
        except requests.exceptions.Timeout:
            logger.error(f"Timeout downloading {source_info['name']}")
            return None
        except Exception as e:
            logger.error(f"Error downloading {source_info['name']}: {e}")
            return None
    
    def merge_all_datasets(self) -> pd.DataFrame:
        """Merge all downloaded datasets into one unified dataset"""
        logger.info("\n🔄 Merging All Datasets...")
        
        all_data = []
        csv_files = list(Path(DATASETS_DIR).glob("*.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Standardize column names
                df.columns = df.columns.str.lower().str.strip()
                
                all_data.append({
                    'source': csv_file.stem,
                    'records': len(df),
                    'data': df
                })
                logger.info(f"   Loaded: {csv_file.stem} ({len(df)} records)")
            except Exception as e:
                logger.warning(f"   Skipped {csv_file.stem}: {e}")
        
        if not all_data:
            logger.warning("No datasets to merge!")
            return pd.DataFrame()
        
        # Create unified dataset
        unified_df = self._create_unified_dataset(all_data)
        logger.info(f"✓ Unified dataset created: {len(unified_df)} total records")
        
        # Save unified dataset
        output_path = os.path.join(DATASETS_DIR, "unified_insurance_dataset.csv")
        unified_df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved to: {output_path}")
        
        return unified_df
    
    def _create_unified_dataset(self, datasets: List[Dict]) -> pd.DataFrame:
        """Create unified dataset from multiple sources"""
        unified_records = []
        
        for dataset_info in datasets:
            df = dataset_info['data']
            source = dataset_info['source']
            
            for _, row in df.iterrows():
                record = {
                    'source': source,
                    'age': self._safe_get(row, ['age', 'years', 'person_age']),
                    'gender': self._safe_get(row, ['gender', 'sex', 'person_gender']),
                    'bmi': self._safe_get(row, ['bmi', 'person_bmi', 'body_mass_index']),
                    'charges': self._safe_get(row, ['charges', 'premium', 'claim', 'person_claim']),
                    'smoker': self._safe_get(row, ['smoker', 'person_smoker']),
                    'region': self._safe_get(row, ['region', 'person_region']),
                    'children': self._safe_get(row, ['children', 'person_children', 'dependents'])
                }
                
                # Only add if we have key fields
                if record['age'] is not None and record['charges'] is not None:
                    unified_records.append(record)
        
        unified_df = pd.DataFrame(unified_records)
        
        # Clean and standardize
        unified_df = self._clean_dataset(unified_df)
        return unified_df
    
    def _clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the dataset"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['charges'] = pd.to_numeric(df['charges'], errors='coerce')
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        df['children'] = pd.to_numeric(df['children'], errors='coerce')
        
        # Remove rows with critical missing values
        df = df.dropna(subset=['age', 'charges'])
        
        # Standardize gender
        df['gender'] = df['gender'].str.lower().str.strip()
        df['gender'] = df['gender'].replace(['m', 'male', '1'], 'male')
        df['gender'] = df['gender'].replace(['f', 'female', '0'], 'female')
        
        # Standardize smoker
        if 'smoker' in df.columns:
            df['smoker'] = df['smoker'].astype(str).str.lower().str.strip()
            df['smoker'] = df['smoker'].replace(['yes', 'true', '1'], True)
            df['smoker'] = df['smoker'].replace(['no', 'false', '0'], False)
        
        # Fill missing values with defaults
        df['bmi'].fillna(df['bmi'].mean(), inplace=True)
        df['children'].fillna(0, inplace=True)
        df['region'].fillna('Unknown', inplace=True)
        df['smoker'].fillna(False, inplace=True)
        
        return df
    
    @staticmethod
    def _safe_get(row, column_names):
        """Safely get value from row using multiple possible column names"""
        for col in column_names:
            if col in row.index:
                value = row[col]
                if pd.notna(value):
                    return value
        return None
    
    def _save_download_log(self):
        """Save download log to JSON"""
        log_path = os.path.join(DATASETS_DIR, f"download_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_path, 'w') as f:
            json.dump(self.download_log, f, indent=2)
        logger.info(f"Log saved: {log_path}")


class DataStatistics:
    """Generate statistics from downloaded data"""
    
    @staticmethod
    def generate_stats(df: pd.DataFrame) -> Dict:
        """Generate comprehensive statistics"""
        stats = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_stats': {}
        }
        
        # Numeric column statistics
        for col in df.select_dtypes(include=[np.number]).columns:
            stats['numeric_stats'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'std': float(df[col].std())
            }
        
        return stats
    
    @staticmethod
    def print_stats(df: pd.DataFrame):
        """Print statistics in readable format"""
        print("\n" + "="*60)
        print("📊 DATASET STATISTICS")
        print("="*60)
        print(f"Total Records: {len(df):,}")
        print(f"Total Columns: {len(df.columns)}")
        print(f"\nColumns: {', '.join(df.columns.tolist())}")
        print(f"\nData Types:\n{df.dtypes}")
        print(f"\nMissing Values:\n{df.isnull().sum()}")
        print(f"\nNumeric Summary:\n{df.describe()}")
        print("="*60 + "\n")


def main():
    """Main execution"""
    print("\n" + "🚀 "*20)
    print("INSURANCE PDF & DATA DOWNLOADER")
    print("🚀 "*20 + "\n")
    
    # Initialize downloader
    downloader = InsurancePDFDownloader()
    
    # Step 1: Download from all sources
    print("\n[STEP 1] Downloading Insurance Data from Multiple Sources...")
    summary = downloader.download_all_sources()
    
    # Step 2: Merge datasets
    print("\n[STEP 2] Merging and Unifying Datasets...")
    unified_df = downloader.merge_all_datasets()
    
    if unified_df.empty:
        print("❌ No data available after merging!")
        return
    
    # Step 3: Generate statistics
    print("\n[STEP 3] Generating Data Statistics...")
    DataStatistics.print_stats(unified_df)
    
    stats = DataStatistics.generate_stats(unified_df)
    stats_path = os.path.join(DATASETS_DIR, "data_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"✓ Statistics saved: {stats_path}")
    
    print("\n" + "✅ "*20)
    print("ALL DATA DOWNLOADS COMPLETED SUCCESSFULLY!")
    print("✅ "*20)
    print(f"\nDataset Location: {DATASETS_DIR}")
    print(f"Ready for ML Model Training!")


if __name__ == "__main__":
    main()
