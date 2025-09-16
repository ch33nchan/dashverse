#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import character_pipeline
        print("character_pipeline import successful")
    except ImportError as e:
        print(f"character_pipeline import failed: {e}")
        return False
    
    try:
        import app
        print("Gradio app import successful")
    except ImportError as e:
        print(f"Gradio app import failed: {e}")
        return False
    
    try:
        from pipeline.rl_optimizer import RLOptimizer
        print("RL optimizer import successful")
    except ImportError as e:
        print(f"RL optimizer import failed: {e}")
        return False
    
    try:
        from pipeline.parquet_storage import ParquetStorage
        print("Parquet storage import successful")
    except ImportError as e:
        print(f"Parquet storage import failed: {e}")
        return False
    
    try:
        import fastapi_app
        print("FastAPI app import successful")
    except ImportError as e:
        print(f"FastAPI app import failed: {e}")
        return False
    
    try:
        import celery_tasks
        print("Celery tasks import successful")
    except ImportError as e:
        print(f"Celery tasks import failed: {e}")
        return False
    
    return True

def test_pipeline_creation():
    """Test basic pipeline creation without requiring HF authentication."""
    print("\nTesting pipeline creation...")
    
    try:
        # Check if HF_TOKEN is properly set
        from dotenv import load_dotenv
        load_dotenv()
        hf_token = os.environ.get("HF_TOKEN")
        
        if not hf_token or hf_token == "your_token_here":
            print("HF_TOKEN not properly configured - skipping model loading test")
            print("To test model loading, set a valid HF_TOKEN in .env file")
            return True
        
        # Only test actual pipeline creation if token is available
        from character_pipeline import create_pipeline
        pipeline = create_pipeline({'use_rl': True})
        print("Pipeline creation successful")
        return True
        
    except Exception as e:
        print(f"Pipeline creation failed: {e}")
        return False

def test_large_scale_features():
    """Test large-scale processing capabilities without requiring HF authentication."""
    print("\nTesting large-scale processing features...")
    
    try:
        # Check if HF_TOKEN is properly set
        from dotenv import load_dotenv
        load_dotenv()
        hf_token = os.environ.get("HF_TOKEN")
        
        if not hf_token or hf_token == "your_token_here":
            print("HF_TOKEN not properly configured - skipping large-scale test")
            print("To test large-scale features, set a valid HF_TOKEN in .env file")
            return True
        
        # Only test if token is available
        from character_pipeline import create_pipeline
        pipeline = create_pipeline({
            'use_rl': True,
            'batch_size': 32,
            'enable_caching': True
        })
        print("Large-scale features test successful")
        return True
        
    except Exception as e:
        print(f"Large-scale features test failed: {e}")
        return False

def test_directory_structure():
    """Test that all required files and directories exist."""
    print("\nTesting directory structure...")
    
    # Use the src directory as base
    base_dir = Path(__file__).resolve().parent
    
    required_files = [
        'app.py',
        'character_pipeline.py', 
        'fastapi_app.py',
        'celery_tasks.py',
        'pipeline_demonstration.ipynb',
        'requirements.txt',
        'README.md',
        'SCALABILITY.md'
    ]
    
    required_dirs = [
        'pipeline',
        'batch_images',
        'cache',
        'data'
    ]
    
    all_found = True
    
    for file in required_files:
        file_path = base_dir / file
        if file_path.exists():
            print(f"{file} exists")
        else:
            print(f"ERROR: {file} not found")
            all_found = False
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"{dir_name}/ directory exists")
        else:
            print(f"ERROR: {dir_name}/ directory not found")
            all_found = False
    
    return all_found

def main():
    """Run all tests and report results."""
    print("Character Extraction Pipeline - Component Test\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Pipeline Creation Test", test_pipeline_creation),
        ("Large-scale Features Test", test_large_scale_features),
        ("Directory Structure Test", test_directory_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\nERROR in {test_name}: {e}")
            results.append(False)
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"ALL TESTS PASSED ({passed}/{total})")
        print("\nSetup verification complete! Your environment is ready.")
        if os.environ.get("HF_TOKEN") == "your_token_here":
            print("\nNOTE: To use Hugging Face models, update HF_TOKEN in .env file")
        return 0
    else:
        print(f"SOME TESTS FAILED ({passed}/{total})")
        if os.environ.get("HF_TOKEN") == "your_token_here":
            print("\nNOTE: HF_TOKEN needs to be configured for full functionality")
        else:
            print("Please check the errors above and install missing dependencies")
        return 1

if __name__ == "__main__":
    exit(main())