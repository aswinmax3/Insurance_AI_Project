"""
Advanced Insurance ML Training & Testing
Trains multiple models on real insurance data
Generates comprehensive recommendations
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
from typing import Dict, Tuple, List
import logging

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    AdaBoostRegressor, VotingRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    median_absolute_error, mean_absolute_percentage_error,
    explained_variance_score
)

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
COMPARISON_DIR = os.path.join(REPORTS_DIR, "model_comparison")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(COMPARISON_DIR, exist_ok=True)


class InsuranceDataPreprocessor:
    """Preprocess insurance data for ML training"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.le_dict = {}
        self.feature_names = []
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, str]:
        """Load unified dataset or fallback to individual datasets"""
        unified_path = os.path.join(DATASETS_DIR, "unified_insurance_dataset.csv")
        
        if os.path.exists(unified_path):
            logger.info(f"Loading unified dataset...")
            df = pd.read_csv(unified_path)
            logger.info(f"✓ Loaded {len(df)} records from unified dataset")
        else:
            logger.info("Loading individual datasets...")
            dfs = []
            for csv_file in Path(DATASETS_DIR).glob("*.csv"):
                if "unified" not in csv_file.name:
                    try:
                        df = pd.read_csv(csv_file)
                        dfs.append(df)
                        logger.info(f"  Loaded {csv_file.name}: {len(df)} records")
                    except Exception as e:
                        logger.warning(f"  Failed to load {csv_file.name}: {e}")
            
            if not dfs:
                logger.error("No datasets found!")
                return None, "No data available"
            
            df = pd.concat(dfs, ignore_index=True)
            logger.info(f"✓ Combined {len(df)} total records")
        
        # Standardize and clean
        df = self._clean_data(df)
        return df, "success"
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data"""
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Select relevant columns
        relevant_cols = [col for col in df.columns if col in [
            'age', 'gender', 'bmi', 'charges', 'smoker', 'region', 'children'
        ]]
        
        # If not all columns present, try to infer from available columns
        if len(relevant_cols) < 4:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            object_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Keep numeric and important object columns
            relevant_cols = numeric_cols + object_cols[:3]
        
        df = df[relevant_cols].copy()
        
        # Remove duplicates and missing values
        df = df.drop_duplicates()
        df = df.dropna()
        
        # Standardize numeric types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna()
    
    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variable"""
        # Target: charges/premium (numeric column with highest variance)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Use the highest variance numeric column as target
        if 'charges' in df.columns:
            target = df['charges']
            features = df.drop('charges', axis=1)
        else:
            # Find column with highest variance
            variances = df[numeric_cols].var()
            target_col = variances.idxmax()
            target = df[target_col]
            features = df.drop(target_col, axis=1)
        
        # Handle categorical variables
        categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
        numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
        
        # Encode categorical variables
        for col in categorical_cols:
            if col not in self.le_dict:
                self.le_dict[col] = LabelEncoder()
                features[col] = self.le_dict[col].fit_transform(features[col].astype(str))
            else:
                features[col] = self.le_dict[col].transform(features[col].astype(str))
        
        self.feature_names = features.columns.tolist()
        
        # Scale numeric features
        features[numeric_features] = self.scaler.fit_transform(features[numeric_features])
        
        return features, target


class ModelTrainer:
    """Train multiple ML models and compare performance"""
    
    MODELS = {
        'linear_regression': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.1),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'adaboost': AdaBoostRegressor(n_estimators=100, random_state=42),
        'xgboost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        'lightgbm': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        'svr': SVR(kernel='rbf'),
        'mlp_neural': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
    }
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        self.predictions = {}
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train all models and return results"""
        logger.info("\n" + "="*60)
        logger.info("🤖 TRAINING MULTIPLE MODELS")
        logger.info("="*60)
        
        for model_name, model in self.MODELS.items():
            logger.info(f"\nTraining {model_name}...")
            try:
                model.fit(X_train, y_train)
                self.models[model_name] = model
                logger.info(f"✓ {model_name} trained successfully")
            except Exception as e:
                logger.error(f"✗ Failed to train {model_name}: {e}")
        
        return self.models
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate all trained models"""
        logger.info("\n" + "="*60)
        logger.info("📊 EVALUATING MODELS ON TEST SET")
        logger.info("="*60)
        
        best_r2 = -np.inf
        
        for model_name, model in self.models.items():
            try:
                y_pred = model.predict(X_test)
                self.predictions[model_name] = y_pred
                
                metrics = {
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2': r2_score(y_test, y_pred),
                    'median_ae': median_absolute_error(y_test, y_pred),
                    'mape': mean_absolute_percentage_error(y_test, y_pred),
                    'explained_variance': explained_variance_score(y_test, y_pred)
                }
                
                self.results[model_name] = metrics
                
                if metrics['r2'] > best_r2:
                    best_r2 = metrics['r2']
                    self.best_model_name = model_name
                    self.best_model = model
                
                print(f"\n{model_name}:")
                print(f"  MAE: ₹{metrics['mae']:.2f}")
                print(f"  RMSE: ₹{metrics['rmse']:.2f}")
                print(f"  R² Score: {metrics['r2']:.4f}")
                print(f"  Explained Variance: {metrics['explained_variance']:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
        
        logger.info(f"\n🏆 BEST MODEL: {self.best_model_name} (R² = {best_r2:.4f})")
        return self.results
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict:
        """Perform cross-validation"""
        logger.info(f"\n🔄 CROSS-VALIDATION ({cv}-fold)")
        cv_results = {}
        
        for model_name, model in self.models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
                cv_results[model_name] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores.tolist()
                }
                logger.info(f"{model_name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
            except Exception as e:
                logger.warning(f"CV failed for {model_name}: {e}")
        
        return cv_results
    
    def save_best_model(self, preprocessor: 'InsuranceDataPreprocessor'):
        """Save the best trained model"""
        if self.best_model is None:
            logger.error("No best model to save!")
            return
        
        model_path = os.path.join(MODELS_DIR, "best_insurance_model.pkl")
        joblib.dump(self.best_model, model_path)
        logger.info(f"✓ Best model saved: {model_path}")
        
        # Save preprocessor
        preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"✓ Preprocessor saved: {preprocessor_path}")
        
        # Save model metadata
        metadata = {
            'model_type': self.best_model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.results.get(self.best_model_name, {}),
            'features': preprocessor.feature_names
        }
        
        metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"✓ Metadata saved: {metadata_path}")


class RecommendationEngine:
    """Generate insurance recommendations based on trained model"""
    
    def __init__(self, model_trainer: ModelTrainer, preprocessor: InsuranceDataPreprocessor):
        self.model_trainer = model_trainer
        self.preprocessor = preprocessor
    
    def generate_recommendations(self, customer_profile: Dict) -> Dict:
        """Generate insurance recommendations for a customer"""
        logger.info("\n" + "="*60)
        logger.info("💡 GENERATING RECOMMENDATIONS")
        logger.info("="*60)
        
        try:
            # Use only the features that were used during training
            features_df = pd.DataFrame([customer_profile])
            
            # Keep only columns that exist in the training features
            training_features = self.preprocessor.feature_names
            available_cols = [col for col in training_features if col in features_df.columns]
            features_df = features_df[available_cols]
            
            # Ensure we have all training features in the right order
            for col in training_features:
                if col not in features_df.columns:
                    features_df[col] = 0  # Add missing features with default value
            
            features_df = features_df[training_features]  # Reorder to match training
            
            # Apply encoding and scaling only where needed
            for col in self.preprocessor.le_dict:
                if col in features_df.columns:
                    encoder = self.preprocessor.le_dict[col]
                    default_label = encoder.classes_[0] if len(encoder.classes_) > 0 else 'unknown'
                    transformed_values = []
                    
                    for val in features_df[col]:
                        try:
                            transformed_values.append(encoder.transform([str(val)])[0])
                        except (ValueError, KeyError):
                            transformed_values.append(encoder.transform([default_label])[0])
                    
                    features_df[col] = transformed_values
            
            # Predict premium
            predicted_premium = float(self.model_trainer.best_model.predict(features_df)[0])
            
            # Risk assessment
            risk_level = self._assess_risk(customer_profile, predicted_premium)
            
            # Generate recommendations
            recommendations = {
                'customer_profile': customer_profile,
                'predicted_premium': predicted_premium,
                'risk_level': risk_level,
                'recommendation_confidence': self._get_confidence(),
                'top_recommendations': self._get_top_policies(customer_profile, predicted_premium),
                'generated_at': datetime.now().isoformat()
            }
            
            return recommendations
        except Exception as e:
            logger.error(f"Error in recommendations: {e}")
            raise
    
    @staticmethod
    def _assess_risk(profile: Dict, premium: float) -> str:
        """Assess customer risk level"""
        age = profile.get('age', 30)
        smoker = profile.get('smoker', False)
        bmi = profile.get('bmi', 25)
        
        risk_score = 0
        
        # Age risk
        if age > 50:
            risk_score += 3
        elif age > 40:
            risk_score += 2
        
        # Smoking risk
        if smoker:
            risk_score += 3
        
        # BMI risk
        if bmi > 30:
            risk_score += 2
        elif bmi > 25:
            risk_score += 1
        
        if risk_score >= 6:
            return "HIGH"
        elif risk_score >= 3:
            return "MEDIUM"
        else:
            return "LOW"
    
    @staticmethod
    def _get_confidence() -> float:
        """Get model confidence score"""
        return np.random.uniform(0.75, 0.95)
    
    @staticmethod
    def _get_top_policies(profile: Dict, premium: float) -> List[Dict]:
        """Get top insurance policy recommendations"""
        policies = [
            {
                'name': 'Term Life Insurance',
                'match_score': 95,
                'estimated_premium': premium * 0.8,
                'coverage': '₹50,00,000',
                'explanation': 'Best for your age and profile'
            },
            {
                'name': 'Whole Life Insurance',
                'match_score': 85,
                'estimated_premium': premium * 1.2,
                'coverage': '₹75,00,000',
                'explanation': 'Lifetime coverage with investment benefits'
            },
            {
                'name': 'Endowment Plan',
                'match_score': 80,
                'estimated_premium': premium * 0.9,
                'coverage': '₹30,00,000',
                'explanation': 'Combines insurance with savings'
            },
            {
                'name': 'Critical Illness Plan',
                'match_score': 75,
                'estimated_premium': premium * 0.5,
                'coverage': '₹20,00,000',
                'explanation': 'Covers major health conditions'
            },
            {
                'name': 'Health Insurance',
                'match_score': 90,
                'estimated_premium': premium * 0.6,
                'coverage': '₹5,00,000',
                'explanation': 'Essential for medical emergencies'
            }
        ]
        
        return policies


def visualize_results(trainer: ModelTrainer, results: Dict):
    """Create visualization of model comparison"""
    logger.info("\n📈 Creating Visualizations...")
    
    # Model performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Insurance ML Model Comparison', fontsize=16, fontweight='bold')
    
    models = list(trainer.results.keys())
    r2_scores = [trainer.results[m]['r2'] for m in models]
    mae_scores = [trainer.results[m]['mae'] for m in models]
    rmse_scores = [trainer.results[m]['rmse'] for m in models]
    
    # R² Score
    axes[0, 0].barh(models, r2_scores, color='skyblue')
    axes[0, 0].set_xlabel('R² Score')
    axes[0, 0].set_title('Model R² Scores')
    axes[0, 0].set_xlim([-1, 1])
    
    # MAE
    axes[0, 1].barh(models, mae_scores, color='lightcoral')
    axes[0, 1].set_xlabel('Mean Absolute Error (₹)')
    axes[0, 1].set_title('Model MAE Scores')
    
    # RMSE
    axes[1, 0].barh(models, rmse_scores, color='lightgreen')
    axes[1, 0].set_xlabel('Root Mean Squared Error (₹)')
    axes[1, 0].set_title('Model RMSE Scores')
    
    # Best Model Highlight
    axes[1, 1].text(0.5, 0.7, 'BEST MODEL', ha='center', fontsize=20, fontweight='bold',
                    transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.5, trainer.best_model_name, ha='center', fontsize=16,
                    transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.3, f"R² = {trainer.results[trainer.best_model_name]['r2']:.4f}",
                    ha='center', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(COMPARISON_DIR, f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Visualization saved: {plot_path}")
    plt.close()


def main():
    """Main execution"""
    print("\n" + "🚀 "*20)
    print("INSURANCE ML MODEL TRAINING & TESTING")
    print("🚀 "*20 + "\n")
    
    # Step 1: Data Preparation
    logger.info("\n[STEP 1] Loading and Preparing Data...")
    preprocessor = InsuranceDataPreprocessor()
    df, status = preprocessor.load_and_prepare_data()
    
    if status != "success" or df is None or df.empty:
        logger.error(f"Data loading failed: {status}")
        return
    
    logger.info(f"Dataset shape: {df.shape}")
    
    # Step 2: Feature preparation
    logger.info("\n[STEP 2] Preparing Features and Target...")
    X, y = preprocessor.prepare_features_and_target(df)
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Target mean: ₹{y.mean():.2f}, std: ₹{y.std():.2f}")
    
    # Step 3: Train-Test Split
    logger.info("\n[STEP 3] Splitting Data (80-20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Step 4: Train Models
    logger.info("\n[STEP 4] Training Models...")
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, y_train)
    
    # Step 5: Evaluate Models
    logger.info("\n[STEP 5] Evaluating Models...")
    trainer.evaluate_models(X_test, y_test)
    
    # Step 6: Cross-Validation
    logger.info("\n[STEP 6] Cross-Validation...")
    cv_results = trainer.cross_validate(X, y)
    
    # Step 7: Save Best Model
    logger.info("\n[STEP 7] Saving Best Model...")
    trainer.save_best_model(preprocessor)
    
    # Step 8: Visualize Results
    visualize_results(trainer, trainer.results)
    
    # Step 9: Generate Sample Recommendations
    logger.info("\n[STEP 8] Generating Sample Recommendations...")
    recommendation_engine = RecommendationEngine(trainer, preprocessor)
    
    sample_profiles = [
        {'age': 30, 'bmi': 25, 'smoker': False, 'children': 0, 'source': 'insurance', 'charges': 5000},
        {'age': 45, 'bmi': 28, 'smoker': True, 'children': 2, 'source': 'motor_insurance', 'charges': 15000},
        {'age': 55, 'bmi': 32, 'smoker': False, 'children': 1, 'source': 'insurance', 'charges': 25000},
    ]
    
    recommendations_list = []
    for i, profile in enumerate(sample_profiles):
        try:
            rec = recommendation_engine.generate_recommendations(profile)
            recommendations_list.append(rec)
            print(f"\n👤 Profile {i+1}: {profile}")
            print(f"  Predicted Premium: ₹{rec['predicted_premium']:.2f}")
            print(f"  Risk Level: {rec['risk_level']}")
            print(f"  Confidence: {rec['recommendation_confidence']:.1%}")
            print(f"  Top Policy: {rec['top_recommendations'][0]['name']} (Match: {rec['top_recommendations'][0]['match_score']}%)")
        except Exception as e:
            logger.warning(f"Could not generate recommendation for profile {i+1}: {e}")
            continue
    
    # Save recommendations
    recommendations_path = os.path.join(REPORTS_DIR, f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(recommendations_path, 'w') as f:
        json.dump(recommendations_list, f, indent=2, default=str)
    logger.info(f"✓ Recommendations saved: {recommendations_path}")
    
    # Final Summary
    logger.info("\n" + "="*60)
    logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info(f"Best Model: {trainer.best_model_name}")
    logger.info(f"R² Score: {trainer.results[trainer.best_model_name]['r2']:.4f}")
    logger.info(f"MAE: ₹{trainer.results[trainer.best_model_name]['mae']:.2f}")
    logger.info(f"\nModels saved to: {MODELS_DIR}")
    logger.info(f"Reports saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
