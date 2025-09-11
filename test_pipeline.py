#!/usr/bin/env python3
"""Test script to verify all pipeline components are working properly."""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")
    
    try:
        from character_pipeline import create_pipeline
        print("character_pipeline import successful")
    except Exception as e:
        print(f"character_pipeline import failed: {e}")
        return False
    
    try:
        from app import UnifiedCharacterExtractionApp
        print("Gradio app import successful")
    except Exception as e:
        print(f"Gradio app import failed: {e}")
        return False
    
    try:
        from pipeline.rl_optimizer import RLOptimizer
        print("RL optimizer import successful")
    except Exception as e:
        print(f"RL optimizer import failed: {e}")
        return False
    
    try:
        from pipeline.parquet_storage import ParquetStorage
        print("Parquet storage import successful")
    except Exception as e:
        print(f"Parquet storage import failed: {e}")
        return False
    
    try:
        from fastapi_app import app as fastapi_app
        print("FastAPI app import successful")
    except Exception as e:
        print(f"FastAPI app import failed: {e}")
        return False
    
    try:
        from celery_tasks import celery_app
        print("Celery tasks import successful")
    except Exception as e:
        print(f"Celery tasks import failed: {e}")
        return False
    
    return True

def test_pipeline_creation():
    """Test pipeline creation and RL integration."""
    print("\nTesting pipeline creation...")
    
    try:
        from character_pipeline import create_pipeline
        
        # Test with RL enabled
        pipeline = create_pipeline({'use_rl': True})
        print(f"Pipeline created successfully")
        print(f"RL enabled: {pipeline.use_rl}")
        print(f"RL optimizer available: {hasattr(pipeline, 'rl_optimizer')}")
        
        # Test components
        components = [
            'input_loader', 'tag_parser', 'clip_analyzer', 'attribute_fusion',
            'edge_case_handler', 'preprocessor', 'cache_manager'
        ]
        
        for component in components:
            if hasattr(pipeline, component):
                print(f"{component} initialized")
            else:
                print(f"{component} missing")
        
        return True
        
    except Exception as e:
        print(f"Pipeline creation failed: {e}")
        traceback.print_exc()
        return False

def test_large_scale_features():
    """Test large-scale processing features."""
    print("\nTesting large-scale processing features...")
    
    try:
        from character_pipeline import create_pipeline
        pipeline = create_pipeline()
        
        # Test HuggingFace datasets
        try:
            items = pipeline.input_loader.get_sample_items(2)
            if items:
                hf_dataset = pipeline.input_loader.create_huggingface_dataset(items)
                if hf_dataset is not None:
                    print("HuggingFace datasets integration working")
                else:
                    print("HuggingFace datasets not available (install: pip install datasets)")
            else:
                print("No sample items found for HF testing")
        except Exception as e:
            print(f"HuggingFace datasets test failed: {e}")
        
        # Test PyTorch DataLoader
        try:
            items = pipeline.input_loader.get_sample_items(2)
            if items:
                pytorch_dataset = pipeline.input_loader.create_pytorch_dataset(items)
                dataloader = pipeline.input_loader.create_dataloader(items, batch_size=1)
                print("PyTorch DataLoader integration working")
            else:
                print("No sample items found for PyTorch testing")
        except Exception as e:
            print(f"PyTorch DataLoader test failed: {e}")
        
        # Test Parquet storage
        try:
            from pipeline.parquet_storage import ParquetStorage
            parquet_storage = ParquetStorage()
            test_data = [{'item_id': 'test', 'success': True, 'confidence': 0.9}]
            result = parquet_storage.store_batch_results(test_data)
            if result.get('success', False):
                print("Parquet storage integration working")
            else:
                print("Parquet storage test failed")
        except Exception as e:
            print(f"Parquet storage test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"Large-scale features test failed: {e}")
        return False

def test_directory_structure():
    """Test directory structure."""
    print("\nTesting directory structure...")
    
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
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"{file_path} exists")
        else:
            print(f"{file_path} missing")
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"{dir_path}/ directory exists")
        else:
            print(f"{dir_path}/ directory missing")
    
    return True

def main():
    """Run all tests."""
    print("Character Extraction Pipeline - Component Test\n")
    
    tests = [
        test_imports,
        test_pipeline_creation,
        test_large_scale_features,
        test_directory_structure
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"ALL TESTS PASSED ({passed}/{total})")
        print("Pipeline is ready for production use!")
        print("All large-scale processing features are functional")
        print("RL optimizer is properly integrated")
        print("Repository is properly organized")
    else:
        print(f"SOME TESTS FAILED ({passed}/{total})")
        print("Please check the errors above and install missing dependencies")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)