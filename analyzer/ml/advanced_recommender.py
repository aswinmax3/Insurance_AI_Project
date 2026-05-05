"""
Enhanced Insurance Recommendation Engine
- Uses trained models for policy recommendations
- Considers user profile and policy attributes
- Provides weighted scoring for multiple policies
- Generates personalized recommendations
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")


class PolicyRecommender:
    """Advanced insurance policy recommendation engine"""
    
    def __init__(self):
        self.premium_model = self._load_model("premium_prediction_model.pkl")
        self.recommendation_metadata = self._load_model("recommendation_metadata.pkl")
        self.policy_data = self._load_policies()
    
    def _load_model(self, model_name):
        """Load trained model"""
        model_path = os.path.join(MODELS_DIR, model_name)
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None
    
    def _load_policies(self):
        """Load policy database"""
        policies_path = os.path.join(DATASETS_DIR, "insurance_policies.csv")
        if os.path.exists(policies_path):
            return pd.read_csv(policies_path)
        return pd.DataFrame()
    
    def recommend_policies(
        self,
        age: int,
        gender: str,
        bmi: float,
        smoker: str,
        income: float,
        family_size: int,
        requirements: Dict,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Generate personalized insurance policy recommendations
        
        Args:
            age: User's age
            gender: 'male' or 'female'
            bmi: Body Mass Index
            smoker: 'yes' or 'no'
            income: Annual income in rupees
            family_size: Number of family members
            requirements: Dict with keys like 'coverage_needed', 'budget_max', 'policy_type'
            top_n: Number of recommendations to return
        
        Returns:
            List of recommended policies with scores and explanations
        """
        
        if self.policy_data.empty or self.premium_model is None:
            return self._generate_fallback_recommendations(age, income)
        
        recommendations = []
        
        # Calculate expected premium for user profile
        expected_premium = self._predict_premium(age, gender, bmi, smoker)
        
        # Score each available policy
        for _, policy in self.policy_data.iterrows():
            score = self._calculate_policy_score(
                policy, 
                age, 
                income, 
                expected_premium, 
                family_size,
                requirements
            )
            
            recommendations.append({
                'policy_id': policy.get('policy_id', 'N/A'),
                'company': policy.get('company', 'Unknown'),
                'policy_type': policy.get('policy_type', 'General'),
                'premium': policy.get('premium', 0),
                'coverage': policy.get('coverage_amount', 0),
                'rating': policy.get('rating', 0),
                'waiting_period': policy.get('waiting_period_months', 0),
                'settlement_days': policy.get('claim_settlement_days', 0),
                'match_score': score['total_score'],
                'recommendation_reason': score['reason'],
                'risk_level': score['risk_level'],
                'affordability': score['affordability'],
                'match_breakdown': score['breakdown']
            })
        
        # Sort by score
        recommendations = sorted(
            recommendations, 
            key=lambda x: x['match_score'], 
            reverse=True
        )[:top_n]
        
        return recommendations
    
    def _predict_premium(self, age: int, gender: str, bmi: float, smoker: str) -> float:
        """Predict expected premium for user profile"""
        if self.premium_model is None:
            return 30000.0  # Default estimate
        
        try:
            prediction = self.premium_model.predict([[age, gender, bmi, smoker]])
            return float(prediction[0])
        except:
            return 30000.0
    
    def _calculate_policy_score(
        self, 
        policy: pd.Series, 
        age: int, 
        income: float, 
        expected_premium: float,
        family_size: int,
        requirements: Dict
    ) -> Dict:
        """
        Calculate comprehensive policy score
        
        Scoring factors (weighted):
        - Affordability (25%): Premium vs income
        - Coverage (25%): Coverage vs needs
        - Quality (20%): Rating and settlement speed
        - Risk (20%): Exclusions and waiting period
        - Company (10%): Brand reputation
        """
        
        breakdown = {}
        total_score = 0
        
        # 1. AFFORDABILITY SCORE (25%)
        policy_premium = policy.get('premium', expected_premium)
        annual_income = income
        
        if annual_income > 0:
            premium_ratio = (policy_premium * 12) / annual_income
            
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
        
        breakdown['affordability'] = affordability_score
        total_score += affordability_score * 0.25
        
        # 2. COVERAGE SCORE (25%)
        coverage = policy.get('coverage_amount', 0)
        if coverage == 0:
            coverage_score = 50
        elif coverage >= 5000000:
            coverage_score = 100
        elif coverage >= 3000000:
            coverage_score = 85
        elif coverage >= 1000000:
            coverage_score = 70
        elif coverage >= 500000:
            coverage_score = 55
        else:
            coverage_score = 30
        
        breakdown['coverage'] = coverage_score
        total_score += coverage_score * 0.25
        
        # 3. QUALITY SCORE (20%)
        rating = policy.get('rating', 3.5)
        settlement_days = policy.get('claim_settlement_days', 30)
        
        rating_score = min(100, (rating / 5) * 100)
        
        if settlement_days <= 7:
            settlement_score = 100
        elif settlement_days <= 14:
            settlement_score = 85
        elif settlement_days <= 21:
            settlement_score = 70
        elif settlement_days <= 30:
            settlement_score = 55
        else:
            settlement_score = 30
        
        quality_score = (rating_score * 0.6) + (settlement_score * 0.4)
        breakdown['quality'] = quality_score
        total_score += quality_score * 0.20
        
        # 4. RISK SCORE (20%)
        exclusions = policy.get('exclusions_count', 0)
        waiting_period = policy.get('waiting_period_months', 0)
        
        if exclusions <= 1:
            exclusion_score = 100
        elif exclusions <= 2:
            exclusion_score = 85
        elif exclusions <= 3:
            exclusion_score = 70
        elif exclusions <= 4:
            exclusion_score = 55
        else:
            exclusion_score = 30
        
        if waiting_period == 0:
            waiting_score = 100
        elif waiting_period <= 3:
            waiting_score = 85
        elif waiting_period <= 6:
            waiting_score = 70
        elif waiting_period <= 12:
            waiting_score = 55
        else:
            waiting_score = 30
        
        risk_score = (exclusion_score * 0.6) + (waiting_score * 0.4)
        breakdown['risk'] = risk_score
        total_score += risk_score * 0.20
        
        # 5. COMPANY SCORE (10%)
        company_ratings = {
            'HDFC Life': 95,
            'ICICI Prudential': 93,
            'SBI Life': 92,
            'LIC': 90,
            'Bajaj Life': 88,
            'Max Life': 87,
            'Aditya Birla': 85,
            'Reliance': 84,
        }
        company = policy.get('company', 'Unknown')
        company_score = company_ratings.get(company, 75)
        breakdown['company'] = company_score
        total_score += company_score * 0.10
        
        # Determine risk level
        if total_score >= 80:
            risk_level = "Low"
        elif total_score >= 60:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Generate reason
        reason = self._generate_reason(
            policy, 
            affordability_score, 
            coverage_score,
            quality_score, 
            risk_score
        )
        
        return {
            'total_score': round(total_score, 2),
            'affordability': affordability_score,
            'coverage': coverage_score,
            'quality': quality_score,
            'risk': risk_score,
            'company': company_score,
            'risk_level': risk_level,
            'reason': reason,
            'breakdown': breakdown
        }
    
    def _generate_reason(
        self, 
        policy: pd.Series,
        affordability: float,
        coverage: float,
        quality: float,
        risk: float
    ) -> str:
        """Generate explanation for recommendation"""
        reasons = []
        
        if affordability >= 85:
            reasons.append("Excellent affordability")
        elif affordability >= 70:
            reasons.append("Good affordability")
        
        if coverage >= 85:
            reasons.append("Strong coverage amount")
        
        if quality >= 80:
            reasons.append("High rating and quick claim settlement")
        
        if risk >= 80:
            reasons.append("Minimal exclusions and waiting period")
        
        if not reasons:
            reasons.append("Reasonable fit for your profile")
        
        return " • ".join(reasons)
    
    def _generate_fallback_recommendations(self, age: int, income: float) -> List[Dict]:
        """Generate recommendations when models are not available"""
        
        recommendations = [
            {
                'policy_id': 'FALLBACK_1',
                'company': 'HDFC Life',
                'policy_type': 'Term Life',
                'premium': 25000,
                'coverage': 5000000,
                'rating': 4.5,
                'waiting_period': 0,
                'settlement_days': 14,
                'match_score': 85.0,
                'recommendation_reason': 'Industry leading company with excellent coverage',
                'risk_level': 'Low',
                'affordability': 90,
                'match_breakdown': {}
            },
            {
                'policy_id': 'FALLBACK_2',
                'company': 'ICICI Prudential',
                'policy_type': 'Term Life',
                'premium': 28000,
                'coverage': 4500000,
                'rating': 4.3,
                'waiting_period': 0,
                'settlement_days': 16,
                'match_score': 82.0,
                'recommendation_reason': 'Strong coverage with good settlement speed',
                'risk_level': 'Low',
                'affordability': 85,
                'match_breakdown': {}
            },
        ]
        
        return recommendations


def get_policy_recommendations(
    age: int,
    gender: str,
    bmi: float,
    smoker: str,
    income: float,
    family_size: int,
    coverage_needed: float = 5000000,
    budget_max: float = 50000,
    policy_type: str = "Term Life"
) -> List[Dict]:
    """
    Main function to get policy recommendations
    
    Usage:
        recommendations = get_policy_recommendations(
            age=30,
            gender='male',
            bmi=24.5,
            smoker='no',
            income=800000,
            family_size=3,
            coverage_needed=5000000,
            budget_max=50000,
            policy_type='Term Life'
        )
    """
    
    recommender = PolicyRecommender()
    
    requirements = {
        'coverage_needed': coverage_needed,
        'budget_max': budget_max,
        'policy_type': policy_type
    }
    
    recommendations = recommender.recommend_policies(
        age=age,
        gender=gender,
        bmi=bmi,
        smoker=smoker,
        income=income,
        family_size=family_size,
        requirements=requirements,
        top_n=5
    )
    
    return recommendations


if __name__ == "__main__":
    # Example usage
    recommendations = get_policy_recommendations(
        age=30,
        gender='male',
        bmi=24.5,
        smoker='no',
        income=800000,
        family_size=3
    )
    
    print("\n" + "="*70)
    print("PERSONALIZED INSURANCE POLICY RECOMMENDATIONS")
    print("="*70 + "\n")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['company']} - {rec['policy_type']}")
        print(f"   Match Score: {rec['match_score']}/100 ({rec['risk_level']} Risk)")
        print(f"   Premium: ₹{rec['premium']:,.0f} | Coverage: ₹{rec['coverage']:,.0f}")
        print(f"   Rating: {rec['rating']}/5 | Settlement: {rec['settlement_days']} days")
        print(f"   Why: {rec['recommendation_reason']}")
        print()
