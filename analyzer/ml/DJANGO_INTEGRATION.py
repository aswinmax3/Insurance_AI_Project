"""
Django Integration Module for Insurance AI Recommendation System
Add these functions and classes to your Django views and models
"""

import json
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import os
import sys

# Add ML module to path
ML_PATH = os.path.join(os.path.dirname(__file__), 'ml')
if ML_PATH not in sys.path:
    sys.path.insert(0, ML_PATH)

from advanced_recommender import PolicyRecommender, get_policy_recommendations
from prediction_engine import predict_expected_cost


# ============================================================================
# API ENDPOINTS
# ============================================================================

@require_http_methods(["POST"])
@csrf_exempt
def api_get_recommendations(request):
    """
    API endpoint: Get personalized insurance recommendations
    
    POST /api/recommendations/
    
    Request body:
    {
        "age": 30,
        "gender": "male",
        "bmi": 24.5,
        "smoker": "no",
        "income": 800000,
        "family_size": 3,
        "coverage_needed": 5000000,
        "budget_max": 50000,
        "policy_type": "Term Life"
    }
    
    Response:
    {
        "success": true,
        "recommendations": [
            {
                "policy_id": "POL001",
                "company": "HDFC Life",
                "policy_type": "Term Life",
                "premium": 25000,
                "coverage": 5000000,
                "rating": 4.5,
                "match_score": 87.5,
                "recommendation_reason": "Excellent affordability...",
                "risk_level": "Low"
            }
        ],
        "expected_premium": 28500.00,
        "user_profile": {...}
    }
    """
    try:
        data = json.loads(request.body)
        
        # Validate required fields
        required_fields = ['age', 'gender', 'bmi', 'smoker', 'income', 'family_size']
        for field in required_fields:
            if field not in data:
                return JsonResponse(
                    {'success': False, 'error': f'Missing field: {field}'},
                    status=400
                )
        
        # Get recommendations
        recommendations = get_policy_recommendations(
            age=int(data['age']),
            gender=data['gender'].lower(),
            bmi=float(data['bmi']),
            smoker=data['smoker'].lower(),
            income=float(data['income']),
            family_size=int(data['family_size']),
            coverage_needed=float(data.get('coverage_needed', 5000000)),
            budget_max=float(data.get('budget_max', 50000)),
            policy_type=data.get('policy_type', 'Term Life')
        )
        
        # Predict expected premium
        premium_result = predict_expected_cost(
            age=int(data['age']),
            sex=data['gender'].lower(),
            bmi=float(data['bmi']),
            children=int(data['family_size']) - 1,  # Assuming first person is policy holder
            smoker=data['smoker'].lower(),
            region='northeast'  # Default region, can be made dynamic
        )
        
        return JsonResponse({
            'success': True,
            'recommendations': recommendations,
            'expected_premium': premium_result.get('predicted_cost', None),
            'user_profile': {
                'age': data['age'],
                'gender': data['gender'],
                'bmi': data['bmi'],
                'smoker': data['smoker'],
                'income': data['income'],
                'family_size': data['family_size']
            }
        })
    
    except json.JSONDecodeError:
        return JsonResponse(
            {'success': False, 'error': 'Invalid JSON'},
            status=400
        )
    except Exception as e:
        return JsonResponse(
            {'success': False, 'error': str(e)},
            status=500
        )


@require_http_methods(["POST"])
@csrf_exempt
def api_predict_premium(request):
    """
    API endpoint: Predict insurance premium for a user profile
    
    POST /api/predict-premium/
    
    Request body:
    {
        "age": 30,
        "sex": "male",
        "bmi": 24.5,
        "children": 2,
        "smoker": "no",
        "region": "northeast"
    }
    
    Response:
    {
        "success": true,
        "predicted_cost": 28500.00,
        "message": "Prediction successful."
    }
    """
    try:
        data = json.loads(request.body)
        
        result = predict_expected_cost(
            age=int(data['age']),
            sex=data['sex'].lower(),
            bmi=float(data['bmi']),
            children=int(data['children']),
            smoker=data['smoker'].lower(),
            region=data.get('region', 'northeast').lower()
        )
        
        return JsonResponse(result)
    
    except Exception as e:
        return JsonResponse(
            {'success': False, 'error': str(e)},
            status=500
        )


# ============================================================================
# VIEW FUNCTIONS (HTML PAGES)
# ============================================================================

def recommendations_page(request):
    """
    Render recommendations page with user input form
    
    URL: /recommendations/
    Template: recommendations.html
    """
    if request.method == 'POST':
        # Handle form submission
        try:
            user_data = {
                'age': int(request.POST.get('age', 0)),
                'gender': request.POST.get('gender', 'male'),
                'bmi': float(request.POST.get('bmi', 24)),
                'smoker': request.POST.get('smoker', 'no'),
                'income': float(request.POST.get('income', 0)),
                'family_size': int(request.POST.get('family_size', 1)),
                'coverage_needed': float(request.POST.get('coverage_needed', 5000000)),
                'budget_max': float(request.POST.get('budget_max', 50000)),
                'policy_type': request.POST.get('policy_type', 'Term Life'),
            }
            
            # Get recommendations
            recommendations = get_policy_recommendations(**user_data)
            
            # Predict premium
            premium_result = predict_expected_cost(
                age=user_data['age'],
                sex=user_data['gender'],
                bmi=user_data['bmi'],
                children=max(0, user_data['family_size'] - 1),
                smoker=user_data['smoker'],
                region='northeast'
            )
            
            context = {
                'recommendations': recommendations,
                'expected_premium': premium_result.get('predicted_cost'),
                'user_data': user_data,
            }
            
            return render(request, 'analyzer/recommendations.html', context)
        
        except Exception as e:
            context = {'error': str(e)}
            return render(request, 'analyzer/recommendations.html', context)
    
    return render(request, 'analyzer/recommendations.html')


def recommendation_comparison(request):
    """
    Compare multiple insurance policies side-by-side
    
    URL: /compare-policies/
    """
    if request.method == 'POST':
        data = json.loads(request.body)
        policy_ids = data.get('policy_ids', [])
        
        # Load policy data and compare
        try:
            recommender = PolicyRecommender()
            policies = recommender.policy_data[
                recommender.policy_data['policy_id'].isin(policy_ids)
            ].to_dict('records')
            
            return JsonResponse({
                'success': True,
                'policies': policies
            })
        
        except Exception as e:
            return JsonResponse(
                {'success': False, 'error': str(e)},
                status=500
            )
    
    return JsonResponse({'error': 'POST required'}, status=400)


# ============================================================================
# DJANGO MODEL INTEGRATION
# ============================================================================

"""
Add these fields to your InsuranceDocument model in models.py:

from django.db import models

class InsuranceDocument(models.Model):
    # ... existing fields ...
    
    # ML Recommendation Fields
    predicted_premium = models.DecimalField(
        max_digits=10, 
        decimal_places=2, 
        null=True, 
        blank=True,
        help_text="ML predicted premium amount"
    )
    recommended_policies = models.JSONField(
        default=list,
        help_text="Top recommended policies"
    )
    recommendation_score = models.FloatField(
        default=0,
        help_text="Confidence score of recommendations (0-100)"
    )
    risk_assessment = models.CharField(
        max_length=50,
        choices=[('Low', 'Low'), ('Medium', 'Medium'), ('High', 'High')],
        default='Medium',
        help_text="Risk level assessment"
    )
    
    def save(self, *args, **kwargs):
        # Auto-generate recommendations on save
        if self.age and self.gender and self.bmi:
            try:
                recs = get_policy_recommendations(
                    age=self.age,
                    gender=self.gender,
                    bmi=self.bmi,
                    smoker=self.smoker,
                    income=self.income or 500000,
                    family_size=1
                )
                
                self.recommended_policies = recs
                self.recommendation_score = recs[0]['match_score'] if recs else 0
                self.risk_assessment = recs[0]['risk_level'] if recs else 'Medium'
            
            except Exception as e:
                print(f"Error generating recommendations: {e}")
        
        super().save(*args, **kwargs)
"""

# ============================================================================
# MANAGEMENT COMMAND
# ============================================================================

"""
Create file: analyzer/management/commands/train_recommendation_models.py

from django.core.management.base import BaseCommand
from analyzer.ml.train_comprehensive_models import main as train_models

class Command(BaseCommand):
    help = 'Train insurance recommendation models'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force retraining even if models exist',
        )
    
    def handle(self, *args, **options):
        self.stdout.write(
            self.style.WARNING('Starting model training...')
        )
        
        try:
            train_models()
            self.stdout.write(
                self.style.SUCCESS('✓ Models trained successfully')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'✗ Training failed: {e}')
            )

Run with: python manage.py train_recommendation_models
"""

# ============================================================================
# EXAMPLE USAGE IN TEMPLATES
# ============================================================================

"""
Example HTML form for recommendations (templates/analyzer/recommendations.html):

{% extends "base.html" %}

{% block content %}
<div class="container">
    <h1>Get Insurance Recommendations</h1>
    
    <form method="post" class="form">
        {% csrf_token %}
        
        <div class="form-group">
            <label for="age">Age</label>
            <input type="number" id="age" name="age" required min="18" max="100">
        </div>
        
        <div class="form-group">
            <label for="gender">Gender</label>
            <select id="gender" name="gender">
                <option value="male">Male</option>
                <option value="female">Female</option>
            </select>
        </div>
        
        <div class="form-group">
            <label for="bmi">BMI</label>
            <input type="number" id="bmi" name="bmi" required min="10" max="50" step="0.1">
        </div>
        
        <div class="form-group">
            <label for="income">Annual Income (₹)</label>
            <input type="number" id="income" name="income" required min="100000">
        </div>
        
        <button type="submit">Get Recommendations</button>
    </form>
    
    {% if recommendations %}
    <div class="recommendations">
        <h2>Top Recommendations for You</h2>
        
        {% for rec in recommendations %}
        <div class="recommendation-card">
            <h3>{{ rec.company }} - {{ rec.policy_type }}</h3>
            <p>Match Score: <strong>{{ rec.match_score }}/100</strong></p>
            <p>Premium: <strong>₹{{ rec.premium }}</strong></p>
            <p>Coverage: ₹{{ rec.coverage }}</p>
            <p>{{ rec.recommendation_reason }}</p>
        </div>
        {% endfor %}
    </div>
    {% endif %}
</div>
{% endblock %}
"""
