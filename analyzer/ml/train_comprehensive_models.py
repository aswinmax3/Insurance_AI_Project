"""
Comprehensive Insurance AI Training Pipeline
- Downloads datasets from Kaggle (with optional API)
- Trains premium prediction models
- Trains insurance recommendation models
- Evaluates performance with detailed metrics
- Generates visualization reports
"""

import os
import sys
import warnings
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    median_absolute_error
)

warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")

# Create necessary directories
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


class DataManager:
    """Handles data loading and preprocessing"""
    
    @staticmethod
    def load_insurance_data(filepath):
        """Load insurance dataset from CSV"""
        try:
            data = pd.read_csv(filepath)
            print(f"✓ Loaded {filepath}: {len(data)} records")
            return data
        except FileNotFoundError:
            print(f"✗ File not found: {filepath}")
            return None
    
    @staticmethod
    def download_kaggle_data():
        """Download datasets from Kaggle (requires kaggle API)"""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            api = KaggleApi()
            api.authenticate()
            
            # Download Medical Insurance dataset
            print("Downloading Medical Insurance dataset from Kaggle...")
            api.dataset_download_files(
                'mirichoi0814/insurance',
                path=DATASETS_DIR,
                unzip=True
            )
            print("✓ Downloaded insurance dataset")
            
        except Exception as e:
            print(f"⚠ Kaggle download failed: {e}")
            print("  Proceeding with existing datasets...")
    
    @staticmethod
    def combine_datasets():
        """Combine multiple insurance datasets"""
        datasets = []
        
        # Load medical insurance
        insurance_csv = os.path.join(DATASETS_DIR, "insurance.csv")
        if os.path.exists(insurance_csv):
            df = pd.read_csv(insurance_csv)
            if 'charges' in df.columns:
                df = df.rename(columns={'charges': 'premium'})
            df['source'] = 'medical_insurance'
            datasets.append(df)
        
        # Load policy data
        policies_csv = os.path.join(DATASETS_DIR, "insurance_policies.csv")
        if os.path.exists(policies_csv):
            df = pd.read_csv(policies_csv)
            datasets.append(df)
        
        if datasets:
            combined = pd.concat(datasets, ignore_index=True, sort=False)
            print(f"✓ Combined datasets: {len(combined)} total records")
            return combined
        return None


class PremiumPredictor:
    """Trains and evaluates premium prediction models"""
    
    def __init__(self):
        self.model = None
        self.pipeline = None
        self.metrics = {}
    
    def prepare_data(self, data):
        """Prepare and validate data for training"""
        required_cols = ['age', 'gender', 'bmi', 'smoker', 'premium']
        
        # Handle column name variations
        if 'sex' in data.columns and 'gender' not in data.columns:
            data['gender'] = data['sex']
        
        # Ensure required columns exist
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Clean data
        data = data.dropna(subset=required_cols)
        data = data[data['premium'] > 0]  # Remove invalid premiums
        
        print(f"✓ Prepared data: {len(data)} clean records")
        return data
    
    def build_pipeline(self, categorical_features, numeric_features):
        """Build preprocessing and model pipeline"""
        preprocessor = ColumnTransformer(
            transformers=[
                ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
                 categorical_features),
                ('numeric', StandardScaler(), numeric_features),
            ]
        )
        
        # Try multiple models and use the best one
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model),
        ])
        
        return pipeline
    
    def train(self, data):
        """Train premium prediction model"""
        print("\n" + "="*60)
        print("TRAINING PREMIUM PREDICTION MODEL")
        print("="*60)
        
        data = self.prepare_data(data.copy())
        
        # Feature engineering
        if 'age' in data.columns:
            data['age_squared'] = data['age'] ** 2
        
        X = data[['age', 'gender', 'bmi', 'smoker']]
        y = data['premium']
        
        # Identify feature types
        categorical_features = ['gender', 'smoker']
        numeric_features = ['age', 'bmi']
        
        # Build and train pipeline
        self.pipeline = self.build_pipeline(categorical_features, numeric_features)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        self.evaluate(X_train, X_test, y_train, y_test)
        
        # Save model
        model_path = os.path.join(MODELS_DIR, "premium_prediction_model.pkl")
        joblib.dump(self.pipeline, model_path)
        print(f"\n✓ Model saved: {model_path}")
        
        return self.pipeline, self.metrics
    
    def evaluate(self, X_train, X_test, y_train, y_test):
        """Comprehensive model evaluation"""
        # Training metrics
        y_train_pred = self.pipeline.predict(X_train)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Test metrics
        y_test_pred = self.pipeline.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_median_ae = median_absolute_error(y_test, y_test_pred)
        
        self.metrics = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_median_ae': test_median_ae,
        }
        
        print("\n--- TRAINING METRICS ---")
        print(f"MAE:  ${train_mae:,.2f}")
        print(f"RMSE: ${train_rmse:,.2f}")
        print(f"R² Score: {train_r2:.4f}")
        
        print("\n--- TEST METRICS ---")
        print(f"MAE:  ${test_mae:,.2f}")
        print(f"RMSE: ${test_rmse:,.2f}")
        print(f"Median AE: ${test_median_ae:,.2f}")
        print(f"R² Score: {test_r2:.4f}")
        
        # Plot results
        self._plot_predictions(y_test, y_test_pred)
    
    def _plot_predictions(self, y_true, y_pred):
        """Plot actual vs predicted premiums"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Scatter plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], 
                        [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Premium')
        axes[0, 0].set_ylabel('Predicted Premium')
        axes[0, 0].set_title('Actual vs Predicted Premiums')
        
        # Residuals
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Premium')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        
        # Distribution of residuals
        axes[1, 0].hist(residuals, bins=30, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        
        # Error distribution
        errors = np.abs(y_true - y_pred)
        axes[1, 1].hist(errors, bins=30, edgecolor='black')
        axes[1, 1].set_xlabel('Absolute Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Absolute Errors')
        
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, "premium_prediction.png")
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"✓ Plot saved: {plot_path}")
        plt.close()


class RecommendationEngine:
    """Trains recommendation scoring model"""
    
    def __init__(self):
        self.model = None
        self.metrics = {}
    
    def create_recommendation_score(self, data):
        """Create recommendation score based on policy attributes"""
        data = data.copy()
        
        # Initialize score
        score = 50  # Base score out of 100
        
        # Factors that improve recommendation
        if 'rating' in data.columns:
            data['rating_score'] = (data['rating'] / 5) * 20  # 0-20 points
        else:
            data['rating_score'] = 0
        
        if 'exclusions_count' in data.columns:
            data['exclusion_score'] = 20 - (data['exclusions_count'] * 3)  # Fewer exclusions = better
            data['exclusion_score'] = data['exclusion_score'].clip(0, 20)
        else:
            data['exclusion_score'] = 15
        
        if 'waiting_period_months' in data.columns:
            data['waiting_score'] = 20 - (data['waiting_period_months'] / 2)  # Shorter waiting = better
            data['waiting_score'] = data['waiting_score'].clip(0, 20)
        else:
            data['waiting_score'] = 15
        
        if 'claim_settlement_days' in data.columns:
            data['settlement_score'] = 30 - (data['claim_settlement_days'] / 2)  # Faster settlement = better
            data['settlement_score'] = data['settlement_score'].clip(0, 20)
        else:
            data['settlement_score'] = 15
        
        # Calculate total recommendation score
        data['recommendation_score'] = (
            data.get('rating_score', 0) +
            data.get('exclusion_score', 0) +
            data.get('waiting_score', 0) +
            data.get('settlement_score', 0)
        ).clip(0, 100)
        
        return data
    
    def train(self, data):
        """Train recommendation model"""
        print("\n" + "="*60)
        print("TRAINING RECOMMENDATION ENGINE")
        print("="*60)
        
        data = self.create_recommendation_score(data.copy())
        
        if 'recommendation_score' in data.columns:
            scores = data['recommendation_score']
            print(f"✓ Recommendation scores created")
            print(f"  Average score: {scores.mean():.2f}/100")
            print(f"  Score range: {scores.min():.2f} - {scores.max():.2f}")
            
            # Save metadata
            metadata = {
                'mean_score': float(scores.mean()),
                'median_score': float(scores.median()),
                'std_score': float(scores.std()),
                'min_score': float(scores.min()),
                'max_score': float(scores.max()),
            }
            
            metadata_path = os.path.join(MODELS_DIR, "recommendation_metadata.pkl")
            joblib.dump(metadata, metadata_path)
            print(f"✓ Metadata saved: {metadata_path}")
            
            # Plot score distribution
            self._plot_recommendation_scores(scores)
            
            return data
        
        return None
    
    def _plot_recommendation_scores(self, scores):
        """Plot recommendation score distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(scores, bins=20, edgecolor='black', color='skyblue')
        axes[0].axvline(scores.mean(), color='red', linestyle='--', 
                        label=f'Mean: {scores.mean():.2f}')
        axes[0].axvline(scores.median(), color='green', linestyle='--', 
                        label=f'Median: {scores.median():.2f}')
        axes[0].set_xlabel('Recommendation Score (0-100)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Recommendation Scores')
        axes[0].legend()
        
        # Box plot
        axes[1].boxplot(scores, vert=True)
        axes[1].set_ylabel('Recommendation Score')
        axes[1].set_title('Recommendation Score Box Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, "recommendation_scores.png")
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"✓ Plot saved: {plot_path}")
        plt.close()


def generate_report(metrics, data):
    """Generate comprehensive training report"""
    report_path = os.path.join(REPORTS_DIR, f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("INSURANCE AI MODEL TRAINING REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DATASET SUMMARY\n")
        f.write("-"*70 + "\n")
        f.write(f"Total Records: {len(data)}\n")
        f.write(f"Columns: {', '.join(data.columns)}\n\n")
        
        f.write("PREMIUM PREDICTION MODEL METRICS\n")
        f.write("-"*70 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key.upper()}: {value:,.2f}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✓ Report generated: {report_path}")


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("INSURANCE AI COMPREHENSIVE TRAINING PIPELINE")
    print("="*70 + "\n")
    
    # Load/combine data
    print("STEP 1: DATA LOADING")
    print("-"*70)
    
    data = DataManager.combine_datasets()
    
    if data is None:
        print("✗ No datasets found!")
        sys.exit(1)
    
    # Train premium prediction model
    print("\n\nSTEP 2: PREMIUM PREDICTION TRAINING")
    print("-"*70)
    
    predictor = PremiumPredictor()
    model, metrics = predictor.train(data)
    
    # Train recommendation engine
    print("\n\nSTEP 3: RECOMMENDATION ENGINE TRAINING")
    print("-"*70)
    
    recommender = RecommendationEngine()
    recommender.train(data)
    
    # Generate report
    print("\n\nSTEP 4: REPORT GENERATION")
    print("-"*70)
    
    generate_report(metrics, data)
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")
    
    print("Generated Files:")
    print(f"  - Premium Model: {os.path.join(MODELS_DIR, 'premium_prediction_model.pkl')}")
    print(f"  - Recommendation Metadata: {os.path.join(MODELS_DIR, 'recommendation_metadata.pkl')}")
    print(f"  - Plots: {PLOTS_DIR}")
    print(f"  - Reports: {REPORTS_DIR}\n")


if __name__ == "__main__":
    main()
