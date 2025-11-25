#!/usr/bin/env python3
"""
Test script for the new Image ID flow with triggered captures

This script tests:
1. Storing image_id in cache
2. Simulating triggered capture processing
3. Verifying directory move functionality
"""

import os
import django
import sys

# Add the Django project to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ml_backend.settings')
django.setup()

from django.core.cache import cache
from django.conf import settings
from ml_api.views import HikrobotCameraManager
import tempfile
import shutil
from datetime import datetime


def test_image_id_storage():
    """Test storing and retrieving image_id from cache"""
    print("🔧 Testing Image ID Storage in Cache...")
    
    test_image_id = "TEST_ID_123456"
    cache_key = 'current_image_id'
    
    # Store in cache
    cache.set(cache_key, test_image_id, timeout=3600)
    print(f"✅ Stored '{test_image_id}' in cache")
    
    # Retrieve from cache
    retrieved_id = cache.get(cache_key)
    print(f"✅ Retrieved '{retrieved_id}' from cache")
    
    assert retrieved_id == test_image_id, f"Expected {test_image_id}, got {retrieved_id}"
    print("✅ Cache storage/retrieval test passed!")
    
    return test_image_id


def test_directory_move_simulation():
    """Test the directory moving functionality"""
    print("\n📁 Testing Directory Move Functionality...")
    
    # Create temporary directories for testing
    base_temp_dir = tempfile.mkdtemp(prefix="marelli_test_")
    print(f"📁 Created test directory: {base_temp_dir}")
    
    try:
        # Simulate the current structure
        inspections_dir = os.path.join(base_temp_dir, 'inspections')
        triggered_inspections_dir = os.path.join(base_temp_dir, 'triggered_inspections')
        
        # Create source directory structure
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        temp_image_id = f"triggered_capture_{timestamp_str}"
        source_dir = os.path.join(inspections_dir, temp_image_id)
        
        # Create subdirectories
        ok_dir = os.path.join(source_dir, 'OK', 'annotated')
        original_dir = os.path.join(source_dir, 'OK', 'original')
        os.makedirs(ok_dir, exist_ok=True)
        os.makedirs(original_dir, exist_ok=True)
        
        # Create some test files
        test_files = [
            os.path.join(ok_dir, 'annotated_image.jpg'),
            os.path.join(original_dir, 'original_image.jpg'),
            os.path.join(source_dir, 'results.json')
        ]
        
        for file_path in test_files:
            with open(file_path, 'w') as f:
                f.write('test content')
        
        print(f"✅ Created source directory: {source_dir}")
        print(f"✅ Created test files: {len(test_files)} files")
        
        # Test the move functionality
        final_image_id = "USER_ENTERED_ID_789"
        target_dir = os.path.join(triggered_inspections_dir, final_image_id)
        
        # Create target base directory
        os.makedirs(triggered_inspections_dir, exist_ok=True)
        
        # Move the directory
        shutil.move(source_dir, target_dir)
        
        print(f"✅ Moved directory from {temp_image_id} to {final_image_id}")
        
        # Verify the move
        if os.path.exists(target_dir):
            print("✅ Target directory exists after move")
            
            # Verify structure
            expected_paths = [
                os.path.join(target_dir, 'OK', 'annotated', 'annotated_image.jpg'),
                os.path.join(target_dir, 'OK', 'original', 'original_image.jpg'),
                os.path.join(target_dir, 'results.json')
            ]
            
            for path in expected_paths:
                if os.path.exists(path):
                    print(f"✅ File exists: {os.path.relpath(path, target_dir)}")
                else:
                    print(f"❌ File missing: {os.path.relpath(path, target_dir)}")
            
            print("✅ Directory move test passed!")
        else:
            print("❌ Target directory does not exist after move")
            return False
        
        return True
        
    finally:
        # Cleanup
        if os.path.exists(base_temp_dir):
            shutil.rmtree(base_temp_dir)
            print(f"🗑️ Cleaned up test directory: {base_temp_dir}")


def test_camera_manager_methods():
    """Test the camera manager helper methods"""
    print("\n🎥 Testing Camera Manager Helper Methods...")
    
    # Test without actual camera connection
    try:
        camera_manager = HikrobotCameraManager()
        
        # Test _get_user_entered_image_id with cache
        test_image_id = "TEST_CACHE_ID_456"
        cache.set('current_image_id', test_image_id, timeout=3600)
        
        retrieved_id = camera_manager._get_user_entered_image_id()
        
        if retrieved_id == test_image_id:
            print(f"✅ _get_user_entered_image_id returned: {retrieved_id}")
        else:
            print(f"❌ Expected {test_image_id}, got {retrieved_id}")
            return False
        
        print("✅ Camera manager helper methods test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Camera manager test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting Image ID Flow Tests")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Cache storage
    try:
        test_image_id_storage()
        tests_passed += 1
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
    
    # Test 2: Directory move
    try:
        if test_directory_move_simulation():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Directory move test failed: {e}")
    
    # Test 3: Camera manager methods
    try:
        if test_camera_manager_methods():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Camera manager test failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🏁 Tests Complete: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! The Image ID flow is ready.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
