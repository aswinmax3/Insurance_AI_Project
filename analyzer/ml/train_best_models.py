"""
Advanced Insurance AI Model Training - Compare Multiple Models & Select Best
Tests 8+ algorithms and picks the top performer
"""

import os
import sys
import warnings
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Import all model types
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    AdaBoostRegressor, VotingRegressor, StackingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    median_absolute_error, mean_absolute_percentage_error
)

warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
RAW_PDFS_DIR = os.path.join(BASE_DIR, "raw_pdfs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
COMPARISON_DIR = os.path.join(REPORTS_DIR, "model_comparison")

# Create directories
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(RAW_PDFS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(COMPARISON_DIR, exist_ok=True)


class DataLoader:
    """Load and preprocess insurance data"""
    
    @staticmethod
    def load_all_datasets():
        """Load and combine all available datasets"""
        datasets = []
        
        # Load CSV datasets
        csv_files = [
            os.path.join(DATASETS_DIR, "insurance.csv"),
            os.path.join(DATASETS_DIR, "insurance_policies.csv"),
            os.path.join(DATASETS_DIR, "synthetic_enhanced_dataset.csv"),
            os.path.join(RAW_PDFS_DIR, "synthetic_enhanced_dataset.csv"),
        ]
        
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    print(f"✓ Loaded {os.path.basename(csv_file)}: {len(df)} records")
                    datasets.append(df)
                except Exception as e:
                    print(f"⚠ Error loading {csv_file}: {e}")
        
        if not datasets:
            print("✗ No datasets found!")
            return None
        
        # Combine datasets
        combined = pd.concat(datasets, ignore_index=True, sort=False)
        
        # Remove duplicates
        combined = combined.drop_duplicates()
        
        print(f"✓ Combined dataset: {len(combined)} total records\n")
        return combined
    
    @staticmethod
    def prepare_training_data(data):
        """Prepare data for training"""
        data = data.copy()
        
        # Handle missing values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].isnull().any():
                data[col].fillna(data[col].median(), inplace=True)
        
        # Drop rows with critical missing values
        critical_cols = ['age', 'bmi', 'premium']
        if any(col in data.columns for col in critical_cols):
            data = data.dropna(subset=[col for col in critical_cols if col in data.columns])
        
        return data


class ModelTrainer:
    """Train and evaluate multiple insurance prediction models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = -np.inf
        self.best_name = None
    
    def get_preprocessing_pipeline(self, categorical_features, numeric_features):
        """Create preprocessing pipeline"""
        return ColumnTransformer(
            transformers=[
                ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
                 categorical_features),
                ('numeric', StandardScaler(), numeric_features),
            ]
        )
    
    def create_models(self):
        """Create all models to test"""
        print("Creating model candidates...\n")
        
        models_config = {
            # Linear models
            'Linear Regression': LinearRegression(),
            
            'Ridge Regression': Ridge(
                alpha=10.0,
                random_state=42
            ),
            
            'Lasso Regression': Lasso(
                alpha=0.1,
                random_state=42
            ),
            
            'ElasticNet': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=42
            ),
            
            # Tree-based models
            'Random Forest': RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            ),
            
            'XGBoost': XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            
            'LightGBM': LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                random_state=42,
                n_jobs=-1
            ),
            
            'AdaBoost': AdaBoostRegressor(
                n_estimators=200,
                learning_rate=0.05,
                random_state=42
            ),
            
            # KNN
            'K-Nearest Neighbors': KNeighborsRegressor(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            ),
            
            # Neural Network
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            ),
            
            # Support Vector Machine
            'Support Vector Machine': SVR(
                kernel='rbf',
                C=100,
                epsilon=0.1
            ),
        }
        
        self.models = models_config
        print(f"✓ Created {len(models_config)} model candidates\n")
        
        return models_config
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, 
                          preprocessing_pipeline):
        """Train all models and evaluate"""
        
        print("="*70)
        print("TRAINING AND EVALUATING ALL MODELS")
        print("="*70 + "\n")
        
        for model_name, model in self.models.items():
            print(f"Training: {model_name}...", end=" ")
            
            try:
                # Create full pipeline
                pipeline = Pipeline([
                    ('preprocessor', preprocessing_pipeline),
                    ('model', model),
                ])
                
                # Train
                pipeline.fit(X_train, y_train)
                
                # Evaluate
                y_pred = pipeline.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                median_ae = median_absolute_error(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    pipeline, X_train, y_train, 
                    cv=5, scoring='r2'
                )
                
                # Store results
                self.results[model_name] = {
                    'pipeline': pipeline,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'median_ae': median_ae,
                    'mape': mape,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred
                }
                
                # Track best model (using R² score)
                if r2 > self.best_score:
                    self.best_score = r2
                    self.best_model = pipeline
                    self.best_name = model_name
                
                print(f"✓ R²={r2:.4f}, MAE=₹{mae:,.0f}")
                
            except Exception as e:
                print(f"✗ Failed: {e}")
        
        print("\n")
        return self.results
    
    def print_comparison(self):
        """Print model comparison table"""
        print("="*70)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*70 + "\n")
        
        # Sort by R² score
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['r2'],
            reverse=True
        )
        
        print(f"{'Rank':<6} {'Model':<25} {'R² Score':<12} {'MAE':<15} {'RMSE':<15}")
        print("-"*73)
        
        for rank, (name, metrics) in enumerate(sorted_results, 1):
            r2 = metrics['r2']
            mae = metrics['mae']
            rmse = metrics['rmse']
            
            marker = "🏆" if rank == 1 else " "
            print(f"{rank:<5} {marker} {name:<23} {r2:>10.4f}  ₹{mae:>13,.0f}  ₹{rmse:>13,.0f}")
        
        print("\n")
    
    def save_best_model(self):
        """Save the best model"""
        if self.best_model is None:
            print("✗ No model trained yet!")
            return False
        
        model_path = os.path.join(MODELS_DIR, "best_premium_prediction_model.pkl")
        joblib.dump(self.best_model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': self.best_name,
            'timestamp': datetime.now().isoformat(),
            'r2_score': float(self.best_score),
            'metrics': {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in self.results[self.best_name].items()
                if k != 'pipeline' and k != 'predictions'
            }
        }
        
        metadata_path = os.path.join(MODELS_DIR, "best_model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Best model saved: {model_path}")
        print(f"✓ Metadata saved: {metadata_path}\n")
        
        return True


class ReportGenerator:
    """Generate comprehensive comparison reports"""
    
    def __init__(self, trainer, X_test, y_test):
        self.trainer = trainer
        self.X_test = X_test
        self.y_test = y_test
    
    def generate_comparison_report(self):
        """Generate detailed comparison report"""
        report_path = os.path.join(
            COMPARISON_DIR, 
            f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("INSURANCE AI - MODEL COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Models Tested: {len(self.trainer.results)}\n")
            f.write(f"Best Model: {self.trainer.best_name}\n")
            f.write(f"Best R² Score: {self.trainer.best_score:.4f}\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS\n")
            f.write("-"*80 + "\n\n")
            
            sorted_results = sorted(
                self.trainer.results.items(),
                key=lambda x: x[1]['r2'],
                reverse=True
            )
            
            for rank, (name, metrics) in enumerate(sorted_results, 1):
                f.write(f"{rank}. {name}\n")
                f.write(f"   R² Score:          {metrics['r2']:.4f}\n")
                f.write(f"   MAE:               ₹{metrics['mae']:,.2f}\n")
                f.write(f"   RMSE:              ₹{metrics['rmse']:,.2f}\n")
                f.write(f"   Median AE:         ₹{metrics['median_ae']:,.2f}\n")
                f.write(f"   MAPE:              {metrics['mape']:.4f}\n")
                f.write(f"   CV Mean (5-fold):  {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})\n")
                f.write("\n")
            
            # Best model analysis
            f.write("\n" + "="*80 + "\n")
            f.write("BEST MODEL ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            best_metrics = self.trainer.results[self.trainer.best_name]
            
            f.write(f"Model: {self.trainer.best_name}\n")
            f.write(f"R² Score: {best_metrics['r2']:.4f}\n")
            f.write(f"Mean Absolute Error: ₹{best_metrics['mae']:,.2f}\n")
            f.write(f"Root Mean Squared Error: ₹{best_metrics['rmse']:,.2f}\n")
            f.write(f"Median Absolute Error: ₹{best_metrics['median_ae']:,.2f}\n")
            f.write(f"Mean Absolute Percentage Error: {best_metrics['mape']:.4f}\n\n")
            
            f.write("INTERPRETATION:\n")
            f.write(f"- The model explains {best_metrics['r2']*100:.2f}% of variance in insurance premiums\n")
            f.write(f"- Average prediction error: ₹{best_metrics['mae']:,.2f}\n")
            f.write(f"- 68% of errors are within ₹{best_metrics['rmse']:,.2f}\n")
            f.write(f"- Cross-validation score: {best_metrics['cv_mean']:.4f} (consistent)\n")
        
        print(f"✓ Report saved: {report_path}\n")
        return report_path
    
    def generate_visualizations(self):
        """Generate comparison visualizations"""
        best_metrics = self.trainer.results[self.trainer.best_name]
        y_pred_best = best_metrics['predictions']
        
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Model comparison by R² Score
        ax1 = plt.subplot(2, 3, 1)
        sorted_results = sorted(
            self.trainer.results.items(),
            key=lambda x: x[1]['r2'],
            reverse=True
        )
        models = [name for name, _ in sorted_results]
        r2_scores = [metrics['r2'] for _, metrics in sorted_results]
        
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(models))]
        ax1.barh(models, r2_scores, color=colors)
        ax1.set_xlabel('R² Score')
        ax1.set_title('Model Comparison - R² Score')
        ax1.set_xlim(0, 1)
        
        # 2. Model comparison by MAE
        ax2 = plt.subplot(2, 3, 2)
        mae_values = [metrics['mae'] for _, metrics in sorted_results]
        ax2.barh(models, mae_values, color='#e74c3c')
        ax2.set_xlabel('Mean Absolute Error (₹)')
        ax2.set_title('Model Comparison - MAE')
        
        # 3. Predictions vs Actual (Best Model)
        ax3 = plt.subplot(2, 3, 3)
        ax3.scatter(self.y_test, y_pred_best, alpha=0.6, s=30)
        ax3.plot([self.y_test.min(), self.y_test.max()],
                [self.y_test.min(), self.y_test.max()],
                'r--', lw=2)
        ax3.set_xlabel('Actual Premium')
        ax3.set_ylabel('Predicted Premium')
        ax3.set_title(f'Best Model: {self.trainer.best_name}')
        
        # 4. Residuals
        ax4 = plt.subplot(2, 3, 4)
        residuals = self.y_test - y_pred_best
        ax4.scatter(y_pred_best, residuals, alpha=0.6, s=30)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_xlabel('Predicted Premium')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residual Plot')
        
        # 5. Residuals Distribution
        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(residuals, bins=40, edgecolor='black', color='#9b59b6')
        ax5.set_xlabel('Residuals')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Distribution of Residuals')
        
        # 6. Error Distribution by Model
        ax6 = plt.subplot(2, 3, 6)
        error_data = []
        error_labels = []
        for name, metrics in sorted_results[:5]:  # Top 5 models
            errors = np.abs(self.y_test - metrics['predictions'])
            error_data.append(errors)
            error_labels.append(name[:15])
        
        ax6.boxplot(error_data, labels=error_labels)
        ax6.set_ylabel('Absolute Error (₹)')
        ax6.set_title('Error Distribution - Top 5 Models')
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plot_path = os.path.join(COMPARISON_DIR, "model_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved: {plot_path}\n")
        plt.close()


def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("INSURANCE AI - ADVANCED MODEL TRAINING & SELECTION")
    print("Testing 11+ Algorithms to Find Best Performer")
    print("="*70 + "\n")
    
    # 1. Load data
    print("STEP 1: LOADING DATA")
    print("-"*70)
    data = DataLoader.load_all_datasets()
    
    if data is None or len(data) < 100:
        print("✗ Insufficient data!")
        sys.exit(1)
    
    # 2. Prepare data
    print("\nSTEP 2: PREPARING DATA")
    print("-"*70)
    data = DataLoader.prepare_training_data(data)
    print(f"✓ Data prepared: {len(data)} records\n")
    
    # 3. Feature selection
    print("STEP 3: FEATURE ENGINEERING")
    print("-"*70)
    
    # Standardize column names
    data.columns = data.columns.str.lower().str.replace(' ', '_')
    
    # Find target variable
    target_cols = ['premium', 'charges', 'cost']
    target_col = None
    for col in target_cols:
        if col in data.columns:
            target_col = col
            break
    
    if target_col is None:
        print("✗ No target column found!")
        sys.exit(1)
    
    # Identify feature types
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_features if col != target_col]
    
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Target Variable: {target_col}")
    print(f"Numeric Features: {numeric_features}")
    print(f"Categorical Features: {categorical_features}\n")
    
    # 4. Train/test split
    print("STEP 4: TRAIN/TEST SPLIT")
    print("-"*70)
    
    X = data[numeric_features + categorical_features]
    y = data[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples\n")
    
    # 5. Train models
    print("STEP 5: MODEL TRAINING")
    print("-"*70 + "\n")
    
    trainer = ModelTrainer()
    trainer.create_models()
    
    preprocessing = trainer.get_preprocessing_pipeline(
        categorical_features, numeric_features
    )
    
    trainer.train_and_evaluate(X_train, X_test, y_train, y_test, preprocessing)
    
    # 6. Print comparison
    print("\nSTEP 6: RESULTS & COMPARISON")
    print("-"*70 + "\n")
    trainer.print_comparison()
    
    # 7. Save best model
    print("STEP 7: SAVING BEST MODEL")
    print("-"*70)
    trainer.save_best_model()
    
    # 8. Generate reports
    print("STEP 8: GENERATING REPORTS")
    print("-"*70)
    
    report_gen = ReportGenerator(trainer, X_test, y_test)
    report_gen.generate_comparison_report()
    report_gen.generate_visualizations()
    
    # Final summary
    print("="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print(f"\nBest Model: {trainer.best_name}")
    print(f"R² Score: {trainer.best_score:.4f}")
    print(f"\nModel saved to: {os.path.join(MODELS_DIR, 'best_premium_prediction_model.pkl')}")
    print(f"Reports saved to: {COMPARISON_DIR}\n")


if __name__ == "__main__":
    main()
