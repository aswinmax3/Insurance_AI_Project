# Insurance AI System - Data Training & Recommendation Setup Guide

## Overview

This guide helps you set up realistic insurance policy data, train machine learning models, and deploy an intelligent recommendation engine for your Insurance AI project.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Dataset Setup](#dataset-setup)
3. [Model Training](#model-training)
4. [Recommendation Engine](#recommendation-engine)
5. [Integration](#integration)
6. [API Usage](#api-usage)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Datasets

```bash
cd analyzer/ml

# Download from GitHub (recommended - no authentication needed)
python download_datasets.py --download github

# Or generate synthetic data
python download_datasets.py --download synthetic

# List available datasets
python download_datasets.py --list
```

### 3. Train Models

```bash
# Train comprehensive models (premium prediction + recommendations)
python train_comprehensive_models.py
```

### 4. Test Recommendations

```bash
# Test the recommendation engine
python advanced_recommender.py
```

---

## Dataset Setup

### Available Datasets

#### 1. Medical Insurance Dataset (Default)
- **Source:** Kaggle (mirichoi0814/insurance)
- **Records:** 1,338
- **Features:** age, sex, bmi, children, smoker, region, charges
- **Use Case:** Premium prediction training

#### 2. Insurance Policies Dataset (Custom)
- **Location:** `datasets/insurance_policies.csv`
- **Records:** 50 realistic policies
- **Features:** Company, policy type, coverage, premium, ratings, etc.
- **Use Case:** Policy matching and recommendations

#### 3. Synthetic Dataset
- **Records:** 200 auto-generated
- **Purpose:** Testing and demonstration
- **Command:** `python download_datasets.py --download synthetic`

#### 4. Claim Data (Optional)
- **Source:** GitHub (datasets/insurance-claims)
- **Use Case:** Claim settlement analysis
- **Command:** `python download_datasets.py --download claim`

### Using Kaggle Datasets

**Setup Kaggle API:**

1. Go to https://www.kaggle.com/settings/account
2. Click **"Create New API Token"**
3. Download `kaggle.json`
4. Place it in:
   - **Windows:** `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **Linux/Mac:** `~/.kaggle/kaggle.json`
5. Set permissions (Linux/Mac only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

**Download Kaggle datasets:**

```bash
python download_datasets.py --download kaggle
python download_datasets.py --download all  # All sources
```

---

## Model Training

### Overview

The training pipeline creates two main models:

1. **Premium Prediction Model** - Predicts insurance costs based on user profile
2. **Recommendation Engine** - Scores and ranks policies for users

### Training Process

```bash
cd analyzer/ml
python train_comprehensive_models.py
```

**Output:**
- `models/premium_prediction_model.pkl` - Trained RandomForest model
- `models/recommendation_metadata.pkl` - Recommendation parameters
- `reports/training_report_*.txt` - Detailed metrics report
- `reports/plots/premium_prediction.png` - Model visualization
- `reports/plots/recommendation_scores.png` - Score distribution

### Model Evaluation Metrics

**Premium Prediction:**
- MAE (Mean Absolute Error): Measures average prediction error
- RMSE (Root Mean Squared Error): Penalizes larger errors
- R² Score: Indicates how well the model fits data

**Example Output:**
```
--- TRAINING METRICS ---
MAE:  $3,456.75
RMSE: $5,234.23
R² Score: 0.8234

--- TEST METRICS ---
MAE:  $3,876.45
RMSE: $5,678.90
Median AE: $2,345.67
R² Score: 0.8012
```

### Recommendation Scoring

The system uses weighted scoring (0-100) based on:

| Factor | Weight | Details |
|--------|--------|---------|
| **Affordability** | 25% | Premium vs. annual income |
| **Coverage** | 25% | Coverage amount vs. needs |
| **Quality** | 20% | Company rating + settlement speed |
| **Risk** | 20% | Exclusions + waiting periods |
| **Company** | 10% | Brand reputation |

---

## Recommendation Engine

### Architecture

```
User Profile (age, income, health, etc.)
        ↓
Premium Predictor → Expected Cost
        ↓
Policy Database
        ↓
Scoring Engine (weighted factors)
        ↓
Ranked Recommendations (0-100 score)
        ↓
Risk Assessment (Low/Medium/High)
        ↓
User Output (Top 5 policies with reasons)
```

### How Recommendations Work

1. **User Input**: Age, gender, BMI, income, family size, requirements
2. **Premium Estimation**: ML model predicts expected cost
3. **Policy Scoring**: Each available policy is scored across 5 dimensions
4. **Ranking**: Policies ranked by total score
5. **Explanation**: Why each policy matches the user's needs

### Example: Create Recommendations

```python
from advanced_recommender import get_policy_recommendations

# Get personalized recommendations
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

# Print results
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec['company']} - {rec['policy_type']}")
    print(f"   Score: {rec['match_score']}/100")
    print(f"   Premium: ₹{rec['premium']:,}")
    print(f"   Why: {rec['recommendation_reason']}")
```

**Sample Output:**
```
1. HDFC Life - Term Life
   Score: 87.50/100 (Low Risk)
   Premium: ₹25,000
   Why: Excellent affordability • Strong coverage amount • High rating and quick claim settlement

2. ICICI Prudential - Term Life
   Score: 85.30/100 (Low Risk)
   Premium: ₹28,000
   Why: Good affordability • Strong coverage amount • High rating and quick claim settlement
```

---

## Integration

### Integrating with Django Views

#### 1. Update `analyzer/views.py`

```python
from analyzer.ml.advanced_recommender import get_policy_recommendations
from analyzer.ml.prediction_engine import predict_expected_cost

def get_recommendations(request):
    """API endpoint for policy recommendations"""
    if request.method == 'POST':
        data = json.loads(request.body)
        
        # Get recommendations
        recommendations = get_policy_recommendations(
            age=data['age'],
            gender=data['gender'],
            bmi=data['bmi'],
            smoker=data['smoker'],
            income=data['income'],
            family_size=data['family_size']
        )
        
        return JsonResponse({'recommendations': recommendations})
    
    return JsonResponse({'error': 'POST required'}, status=400)
```

#### 2. Create Django Management Command

```python
# analyzer/management/commands/train_models.py

from django.core.management.base import BaseCommand
from analyzer.ml.train_comprehensive_models import PremiumPredictor, RecommendationEngine

class Command(BaseCommand):
    def handle(self, *args, **options):
        self.stdout.write("Training insurance AI models...")
        
        # Your training code here
        predictor = PremiumPredictor()
        # ... training logic
        
        self.stdout.write(self.style.SUCCESS("✓ Models trained successfully"))
```

#### 3. Create Cron Job for Periodic Retraining

```python
# analyzer/tasks.py (for Celery)

from celery import shared_task
from analyzer.ml.train_comprehensive_models import main as train_models

@shared_task
def retrain_models_monthly():
    """Retrain models monthly with new data"""
    train_models()
```

### URL Configuration

```python
# analyzer/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('api/recommendations/', views.get_recommendations, name='get_recommendations'),
    path('recommendations/', views.recommendations_page, name='recommendations'),
]
```

---

## API Usage

### Python API

```python
from analyzer.ml.advanced_recommender import PolicyRecommender

# Initialize recommender
recommender = PolicyRecommender()

# Get top 5 recommendations
recommendations = recommender.recommend_policies(
    age=35,
    gender='female',
    bmi=23.5,
    smoker='no',
    income=1200000,
    family_size=2,
    requirements={
        'coverage_needed': 5000000,
        'budget_max': 60000,
        'policy_type': 'Term Life'
    },
    top_n=5
)
```

### Response Format

```json
[
  {
    "policy_id": "POL001",
    "company": "HDFC Life",
    "policy_type": "Term Life",
    "premium": 25000,
    "coverage": 5000000,
    "rating": 4.5,
    "waiting_period": 0,
    "settlement_days": 14,
    "match_score": 87.5,
    "recommendation_reason": "Excellent affordability • Strong coverage amount",
    "risk_level": "Low",
    "affordability": 95,
    "match_breakdown": {
      "affordability": 95,
      "coverage": 100,
      "quality": 85,
      "risk": 80,
      "company": 95
    }
  }
]
```

---

## Troubleshooting

### Issue: Model not found
**Solution:**
```bash
python train_comprehensive_models.py  # Retrain models
```

### Issue: No datasets available
**Solution:**
```bash
python download_datasets.py --download synthetic
# or
python download_datasets.py --download github
```

### Issue: Kaggle authentication failed
**Solution:**
1. Verify `kaggle.json` is in correct location
2. Run `python download_datasets.py --setup-kaggle`
3. Manually place the file and retry

### Issue: Low model accuracy
**Solution:**
1. Add more training data
2. Collect real user feedback
3. Implement A/B testing
4. Retrain with updated parameters

---

## Next Steps

1. ✅ **Data Setup** - Download realistic datasets
2. ✅ **Model Training** - Train ML models
3. ✅ **Recommendation Engine** - Deploy recommendation system
4. ⬜ **A/B Testing** - Test recommendations with real users
5. ⬜ **Feedback Loop** - Collect user feedback to improve models
6. ⬜ **Continuous Learning** - Implement periodic model retraining
7. ⬜ **API Documentation** - Document recommendation API endpoints

---

## References

- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Medical Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0814/insurance)
- [Health Insurance Cross Sell](https://www.kaggle.com/datasets/buntyshah/health-insurance-cross-sell-prediction)

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review model training reports in `reports/`
3. Examine detailed logs during training
4. Test models individually with `advanced_recommender.py`
