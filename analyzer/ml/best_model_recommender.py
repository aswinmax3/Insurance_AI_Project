"""
Production Insurance Recommendation Engine
Uses best trained model for premium prediction & policy recommendations
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
RAW_PDFS_DIR = os.path.join(BASE_DIR, "raw_pdfs")


class BestModelRecommender:
    """
    Production-grade recommendation engine using best trained model
    Automatically loads best model and provides recommendations
    """
    
    def __init__(self):
        """Initialize with best model"""
        self.best_model = self._load_best_model()
        self.policy_data = self._load_policies()
        self.model_metadata = self._load_metadata()
        
        if self.best_model is None:
            raise RuntimeError("No trained model found! Run train_best_models.py first")
    
    def _load_best_model(self):
        """Load best trained model"""
        model_path = os.path.join(MODELS_DIR, "best_premium_prediction_model.pkl")
        
        if not os.path.exists(model_path):
            print("⚠ Best model not found. Using fallback model...")
            # Try fallback
            fallback_path = os.path.join(MODELS_DIR, "premium_prediction_model.pkl")
            if os.path.exists(fallback_path):
                return joblib.load(fallback_path)
            return None
        
        try:
            model = joblib.load(model_path)
            print("✓ Loaded best model successfully")
            return model
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return None
    
    def _load_policies(self):
        """Load policy database"""
        # Try multiple policy sources
        policy_files = [
            os.path.join(DATASETS_DIR, "insurance_policies.csv"),
            os.path.join(RAW_PDFS_DIR, "policies.csv"),
        ]
        
        for policy_file in policy_files:
            if os.path.exists(policy_file):
                try:
                    return pd.read_csv(policy_file)
                except:
                    continue
        
        return pd.DataFrame()
    
    def _load_metadata(self):
        """Load model metadata"""
        metadata_path = os.path.join(MODELS_DIR, "best_model_metadata.json")
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {}
    
    def predict_premium(
        self,
        age: int,
        gender: str,
        bmi: float,
        smoker: str,
        children: int = 0,
        region: str = "northeast"
    ) -> Dict:
        """
        Predict insurance premium using best model
        
        Args:
            age: Age (18-80)
            gender: 'male' or 'female'
            bmi: Body Mass Index
            smoker: 'yes' or 'no'
            children: Number of children
            region: Geographic region
        
        Returns:
            Dictionary with predicted premium and confidence
        """
        if self.best_model is None:
            return {
                'success': False,
                'error': 'Model not available',
                'predicted_premium': None
            }
        
        try:
            # Create input dataframe with same format as training
            input_data = pd.DataFrame([{
                'age': age,
                'gender': gender.lower(),
                'bmi': bmi,
                'children': children,
                'smoker': smoker.lower(),
                'region': region.lower(),
            }])
            
            # Make prediction
            prediction = self.best_model.predict(input_data)[0]
            
            # Get confidence from metadata
            confidence = self.model_metadata.get('metrics', {}).get('r2', 0.75)
            
            return {
                'success': True,
                'predicted_premium': round(float(prediction), 2),
                'confidence': round(float(confidence) * 100, 2),
                'model': self.model_metadata.get('model_name', 'Best Model'),
                'currency': 'INR'
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'predicted_premium': None
            }
    
    def recommend_policies(
        self,
        age: int,
        gender: str,
        bmi: float,
        smoker: str,
        income: float,
        family_size: int,
        coverage_needed: float = 5000000,
        budget_max: float = 50000,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Get top policy recommendations
        
        Args:
            age: User age
            gender: Gender
            bmi: BMI
            smoker: Smoking status
            income: Annual income
            family_size: Family size
            coverage_needed: Required coverage
            budget_max: Maximum budget
            top_n: Number of recommendations
        
        Returns:
            List of top N policy recommendations
        """
        
        if self.policy_data.empty:
            return self._generate_default_recommendations()
        
        # Predict expected premium
        premium_prediction = self.predict_premium(
            age, gender, bmi, smoker, family_size - 1
        )
        
        expected_premium = premium_prediction.get('predicted_premium', 30000)
        
        # Score each policy
        recommendations = []
        
        for _, policy in self.policy_data.iterrows():
            score = self._calculate_recommendation_score(
                policy, age, income, expected_premium,
                family_size, coverage_needed, budget_max
            )
            
            recommendations.append({
                'policy_id': policy.get('policy_id', 'N/A'),
                'company': policy.get('company', 'Unknown'),
                'policy_type': policy.get('policy_type', 'General'),
                'premium': float(policy.get('premium', 0)),
                'coverage': float(policy.get('coverage_amount', 0)),
                'rating': float(policy.get('rating', 3.5)),
                'match_score': round(score['score'], 2),
                'risk_level': score['risk_level'],
                'recommendation': score['recommendation'],
                'affordability_ratio': round(
                    (float(policy.get('premium', 0)) * 12) / income, 4
                ) if income > 0 else 0
            })
        
        # Sort and return top N
        recommendations = sorted(
            recommendations,
            key=lambda x: x['match_score'],
            reverse=True
        )[:top_n]
        
        return recommendations
    
    def _calculate_recommendation_score(
        self, policy: pd.Series, age: int, income: float,
        expected_premium: float, family_size: int,
        coverage_needed: float, budget_max: float
    ) -> Dict:
        """Calculate comprehensive recommendation score"""
        
        score = 50  # Base score
        policy_premium = float(policy.get('premium', expected_premium))
        policy_coverage = float(policy.get('coverage_amount', 0))
        policy_rating = float(policy.get('rating', 3.5))
        
        # Affordability (40%)
        if income > 0:
            annual_premium = policy_premium * 12
            premium_ratio = annual_premium / income
            
            if premium_ratio < 0.05:
                affordability_score = 100
            elif premium_ratio < 0.08:
                affordability_score = 85
            elif premium_ratio < 0.12:
                affordability_score = 70
            elif premium_ratio < 0.15:
                affordability_score = 55
            else:
                affordability_score = 30
        else:
            affordability_score = 50
        
        score += affordability_score * 0.40
        
        # Coverage Match (35%)
        if policy_coverage >= coverage_needed:
            coverage_score = 100
        elif policy_coverage >= coverage_needed * 0.8:
            coverage_score = 85
        elif policy_coverage >= coverage_needed * 0.6:
            coverage_score = 70
        elif policy_coverage >= 500000:
            coverage_score = 50
        else:
            coverage_score = 30
        
        score += coverage_score * 0.35
        
        # Rating & Quality (25%)
        rating_score = min(100, (policy_rating / 5) * 100)
        score += rating_score * 0.25
        
        # Determine risk level
        if score >= 80:
            risk_level = "Low"
        elif score >= 60:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Recommendation reason
        reasons = []
        if affordability_score >= 80:
            reasons.append("Very affordable")
        elif affordability_score >= 70:
            reasons.append("Good affordability")
        
        if coverage_score >= 85:
            reasons.append("Excellent coverage")
        
        if rating_score >= 85:
            reasons.append("Highly rated")
        
        recommendation = " • ".join(reasons) if reasons else "Suitable match"
        
        return {
            'score': score,
            'risk_level': risk_level,
            'recommendation': recommendation
        }
    
    def _generate_default_recommendations(self) -> List[Dict]:
        """Generate default recommendations when no policies loaded"""
        return [
            {
                'policy_id': 'DEFAULT_1',
                'company': 'HDFC Life',
                'policy_type': 'Term Life',
                'premium': 25000,
                'coverage': 5000000,
                'rating': 4.5,
                'match_score': 85.0,
                'risk_level': 'Low',
                'recommendation': 'Industry leader with excellent coverage',
                'affordability_ratio': 0.03
            },
            {
                'policy_id': 'DEFAULT_2',
                'company': 'ICICI Prudential',
                'policy_type': 'Term Life',
                'premium': 28000,
                'coverage': 4500000,
                'rating': 4.3,
                'match_score': 82.0,
                'risk_level': 'Low',
                'recommendation': 'Strong coverage with good service',
                'affordability_ratio': 0.034
            },
        ]


def get_best_recommendations(
    age: int,
    gender: str,
    bmi: float,
    smoker: str,
    income: float,
    family_size: int,
    coverage_needed: float = 5000000,
    budget_max: float = 50000,
    top_n: int = 5
) -> Dict:
    """
    Get insurance recommendations using best trained model
    
    Usage:
        results = get_best_recommendations(
            age=30, gender='male', bmi=24.5, smoker='no',
            income=800000, family_size=3
        )
        
        print(f"Top recommendation: {results['recommendations'][0]['company']}")
        print(f"Predicted premium: ₹{results['predicted_premium']}")
    """
    
    try:
        recommender = BestModelRecommender()
        
        # Get premium prediction
        premium_pred = recommender.predict_premium(
            age, gender, bmi, smoker, family_size - 1
        )
        
        # Get recommendations
        recommendations = recommender.recommend_policies(
            age, gender, bmi, smoker, income, family_size,
            coverage_needed, budget_max, top_n
        )
        
        return {
            'success': True,
            'user_profile': {
                'age': age,
                'gender': gender,
                'bmi': bmi,
                'smoker': smoker,
                'income': income,
                'family_size': family_size
            },
            'predicted_premium': premium_pred.get('predicted_premium'),
            'prediction_confidence': premium_pred.get('confidence'),
            'model_name': premium_pred.get('model'),
            'recommendations': recommendations
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'recommendations': []
        }


if __name__ == "__main__":
    # Example usage
    results = get_best_recommendations(
        age=30,
        gender='male',
        bmi=24.5,
        smoker='no',
        income=800000,
        family_size=3
    )
    
    if results['success']:
        print("\n" + "="*70)
        print("BEST MODEL INSURANCE RECOMMENDATIONS")
        print("="*70 + "\n")
        
        print(f"Predicted Premium: ₹{results['predicted_premium']:,.0f}")
        print(f"Model Confidence: {results['prediction_confidence']}%\n")
        
        print("Top Recommendations:\n")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec['company']} - {rec['policy_type']}")
            print(f"   Match Score: {rec['match_score']}/100 ({rec['risk_level']} Risk)")
            print(f"   Premium: ₹{rec['premium']:,.0f} | Coverage: ₹{rec['coverage']:,.0f}")
            print(f"   Why: {rec['recommendation']}\n")
    else:
        print(f"Error: {results['error']}")
