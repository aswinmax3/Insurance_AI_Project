"""
Insurance AI Master Pipeline
Complete end-to-end workflow: Download → Process → Train → Test → Recommend
Run with: python insurance_ai_pipeline.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


class MasterPipeline:
    """Orchestrate complete insurance AI pipeline"""
    
    def __init__(self):
        self.pipeline_log = {
            'start_time': datetime.now().isoformat(),
            'steps': [],
            'status': 'running'
        }
        self.start_time = time.time()
    
    def run_step(self, step_num: int, step_name: str, script_name: str) -> bool:
        """Run a pipeline step"""
        logger.info("\n" + "="*70)
        logger.info(f"[STEP {step_num}] {step_name}")
        logger.info("="*70)
        
        step_start = time.time()
        script_path = os.path.join(BASE_DIR, script_name)
        
        if not os.path.exists(script_path):
            logger.error(f"Script not found: {script_path}")
            self.pipeline_log['steps'].append({
                'step': step_num,
                'name': step_name,
                'status': 'failed',
                'error': 'Script not found',
                'duration': 0
            })
            return False
        
        try:
            logger.info(f"Executing: {script_name}")
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=False,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            step_duration = time.time() - step_start
            
            if result.returncode == 0:
                logger.info(f"✓ {step_name} completed in {step_duration:.2f}s")
                self.pipeline_log['steps'].append({
                    'step': step_num,
                    'name': step_name,
                    'status': 'success',
                    'duration': step_duration
                })
                return True
            else:
                logger.error(f"✗ {step_name} failed with return code {result.returncode}")
                self.pipeline_log['steps'].append({
                    'step': step_num,
                    'name': step_name,
                    'status': 'failed',
                    'return_code': result.returncode,
                    'duration': step_duration
                })
                return False
        
        except subprocess.TimeoutExpired:
            logger.error(f"✗ {step_name} timed out (1 hour limit)")
            self.pipeline_log['steps'].append({
                'step': step_num,
                'name': step_name,
                'status': 'timeout',
                'duration': 3600
            })
            return False
        
        except Exception as e:
            logger.error(f"✗ Error executing {step_name}: {str(e)}")
            self.pipeline_log['steps'].append({
                'step': step_num,
                'name': step_name,
                'status': 'error',
                'error': str(e)
            })
            return False
    
    def execute(self):
        """Execute complete pipeline"""
        logger.info("\n" + "🚀 "*30)
        logger.info("INSURANCE AI MASTER PIPELINE")
        logger.info("Complete End-to-End Training & Recommendation System")
        logger.info("🚀 "*30)
        
        all_success = True
        
        # Step 1: Download Insurance Data
        logger.info("\n📥 PHASE 1: DATA ACQUISITION")
        if not self.run_step(1, "Download Insurance PDFs & Data from Multiple Sources", 
                            "download_insurance_pdfs.py"):
            all_success = False
            logger.warning("⚠️  Data download failed, but continuing with available data...")
        else:
            logger.info("✓ Insurance data successfully downloaded from open sources")
        
        # Step 2: Train and Test Models
        logger.info("\n🤖 PHASE 2: MODEL TRAINING & TESTING")
        if not self.run_step(2, "Train Multiple ML Models & Generate Recommendations", 
                            "train_and_test_model.py"):
            all_success = False
            logger.error("✗ Model training failed!")
            return False
        else:
            logger.info("✓ Models trained and tested successfully")
        
        # Pipeline Complete
        total_duration = time.time() - self.start_time
        self.pipeline_log['end_time'] = datetime.now().isoformat()
        self.pipeline_log['total_duration'] = total_duration
        self.pipeline_log['status'] = 'success' if all_success else 'partial_success'
        
        self._print_final_summary(total_duration, all_success)
        self._save_pipeline_log()
        
        return all_success
    
    def _print_final_summary(self, total_duration: float, success: bool):
        """Print final pipeline summary"""
        logger.info("\n" + "="*70)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("="*70)
        
        for step in self.pipeline_log['steps']:
            status_icon = "✓" if step['status'] == 'success' else "✗"
            logger.info(f"{status_icon} Step {step['step']}: {step['name']}")
            logger.info(f"  Status: {step['status'].upper()}")
            logger.info(f"  Duration: {step.get('duration', 0):.2f}s")
        
        logger.info(f"\nTotal Pipeline Duration: {total_duration/60:.2f} minutes")
        logger.info(f"Overall Status: {'✅ SUCCESS' if success else '⚠️  PARTIAL SUCCESS'}")
        
        logger.info("\n" + "="*70)
        logger.info("📊 OUTPUT LOCATIONS:")
        logger.info("="*70)
        logger.info(f"• Datasets: {os.path.join(BASE_DIR, 'datasets')}")
        logger.info(f"• Trained Models: {os.path.join(BASE_DIR, 'models')}")
        logger.info(f"• Reports & Visualizations: {os.path.join(BASE_DIR, 'reports')}")
        logger.info(f"• Recommendations: {os.path.join(BASE_DIR, 'reports')}")
        
        logger.info("\n" + "="*70)
        logger.info("🎯 NEXT STEPS:")
        logger.info("="*70)
        logger.info("1. Review model performance in reports/model_comparison/")
        logger.info("2. Test recommendations with sample customer profiles")
        logger.info("3. Integrate with Django API endpoints")
        logger.info("4. Deploy to production and monitor accuracy")
        logger.info("5. Retrain models monthly with new data")
        
        logger.info("\n" + "✅ "*30)
    
    def _save_pipeline_log(self):
        """Save pipeline execution log"""
        log_path = os.path.join(REPORTS_DIR, f"pipeline_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_path, 'w') as f:
            json.dump(self.pipeline_log, f, indent=2, default=str)
        logger.info(f"\n📝 Pipeline log saved: {log_path}")


def print_welcome():
    """Print welcome message"""
    welcome = """
╔════════════════════════════════════════════════════════════════════════════╗
║                   🏢 INSURANCE AI SYSTEM - MASTER PIPELINE                 ║
║                                                                            ║
║  This system will:                                                         ║
║  ✓ Download real insurance data from multiple open sources                 ║
║  ✓ Process and clean the data                                             ║
║  ✓ Train multiple ML algorithms (10+ models)                              ║
║  ✓ Select the best performing model                                       ║
║  ✓ Generate personalized insurance recommendations                        ║
║  ✓ Create comprehensive reports and visualizations                        ║
║                                                                            ║
║  Estimated Duration: 15-30 minutes                                        ║
║  Required: Python 3.8+, 4GB RAM minimum                                   ║
╚════════════════════════════════════════════════════════════════════════════╝
    """
    print(welcome)


def main():
    """Main entry point"""
    print_welcome()
    
    logger.info("Starting Master Pipeline Execution...")
    
    pipeline = MasterPipeline()
    success = pipeline.execute()
    
    if success:
        print("\n✅ Pipeline execution completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Pipeline completed with errors!")
        sys.exit(1)


if __name__ == "__main__":
    main()
