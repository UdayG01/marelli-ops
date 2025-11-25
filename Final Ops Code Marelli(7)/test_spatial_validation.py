# test_spatial_validation.py
# Place this file in your project root directory for testing

import os
import sys
import django
from pathlib import Path

# Setup Django environment
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ml_backend.settings')
django.setup()

from ml_api.services import enhanced_nut_detection_service

def setup_reference_profile(reference_image_path):
    """
    Setup reference distance profile from your perfect reference image
    """
    print("=" * 60)
    print("SETTING UP REFERENCE DISTANCE PROFILE")
    print("=" * 60)
    
    if not os.path.exists(reference_image_path):
        print(f"ERROR: Reference image not found: {reference_image_path}")
        return False
    
    print(f"Processing reference image: {reference_image_path}")
    
    # Create reference profile
    result = enhanced_nut_detection_service.create_reference_from_image(reference_image_path)
    
    if result['success']:
        print("SUCCESS: Reference profile created!")
        print(f"Reference distances:")
        for dist in result['profile']['reference_distances']:
            print(f"  {dist['pair']}: {dist['distance']:.2f} pixels (tolerance: {dist['min_distance']:.2f} - {dist['max_distance']:.2f})")
        return True
    else:
        print(f"ERROR: {result['error']}")
        return False

def test_image_with_spatial_validation(test_image_path):
    """
    Test an image with spatial validation
    """
    print("\n" + "=" * 60)
    print("TESTING IMAGE WITH SPATIAL VALIDATION")
    print("=" * 60)
    
    if not os.path.exists(test_image_path):
        print(f"ERROR: Test image not found: {test_image_path}")
        return
    
    print(f"Processing test image: {test_image_path}")
    
    # Process image
    result = enhanced_nut_detection_service.process_image_with_id(
        image_path=test_image_path,
        image_id=f"TEST_{Path(test_image_path).stem}"
    )
    
    if result['success']:
        print("\nDETECTION RESULTS:")
        print(f"  Total detections: {result['detection_summary']['total_detections']}")
        print(f"  Present nuts: {result['decision']['present_count']}")
        print(f"  Missing nuts: {result['decision']['missing_count']}")
        print(f"  Overall status: {result['decision']['status']}")
        print(f"  Action: {result['decision']['action']}")
        print(f"  Box color: {result['decision']['box_color']}")
        
        # Spatial validation results
        spatial = result['decision']['spatial_validation']
        print(f"\nSPATIAL VALIDATION:")
        print(f"  Enabled: {spatial['enabled']}")
        if spatial['enabled']:
            print(f"  Validation passed: {spatial['passed']}")
            if spatial['details']:
                details = spatial['details']
                print(f"  Validation rate: {details['validation_rate']:.1f}%")
                print(f"  Passed pairs: {details['passed_pairs']}/{details['total_pairs']}")
                
                print(f"\n  Individual distance validations:")
                for validation in details['individual_results']:
                    status_symbol = "✓" if validation['passed'] else "✗"
                    print(f"    {status_symbol} {validation['pair']}: {validation['current_distance']:.2f} pixels "
                          f"(expected: {validation['reference_distance']:.2f}, deviation: {validation['deviation_percent']:+.1f}%)")
        
        print(f"\nAnnotated image saved to: {result['annotated_image_path']}")
        
    else:
        print(f"ERROR: {result['error']}")

def demonstrate_distance_calculation(image_path):
    """
    Demonstrate distance calculation and center marking
    """
    print("\n" + "=" * 60)
    print("DISTANCE CALCULATION DEMONSTRATION")
    print("=" * 60)
    
    # Load image and run detection
    detections = enhanced_nut_detection_service._run_detection(image_path)
    
    # Load image for center marking
    import cv2
    image = cv2.imread(image_path)
    
    # Calculate distances and mark centers
    distance_data = enhanced_nut_detection_service._calculate_nut_centers_and_distances(detections, image)
    
    # Print results
    print(f"Image: {image_path}")
    print(f"Total nuts detected: {distance_data['total_nuts']}")
    
    print(f"\nNUT CENTERS:")
    for center_info in distance_data['centers']:
        print(f"  Nut {center_info['nut_index']+1}: Center({center_info['center'][0]:.1f}, {center_info['center'][1]:.1f}) - {center_info['class_name']} (conf: {center_info['confidence']:.3f})")
    
    print(f"\nDISTANCES BETWEEN NUTS:")
    for dist_info in distance_data['distances']:
        print(f"  Nut{dist_info['nut1_index']+1} ↔ Nut{dist_info['nut2_index']+1}: {dist_info['distance']:.2f} pixels")
    
    # Save marked image
    if distance_data['marked_image'] is not None:
        output_path = f"marked_centers_{Path(image_path).stem}.jpg"
        cv2.imwrite(output_path, distance_data['marked_image'])
        print(f"\nMarked image with centers saved to: {output_path}")

def main():
    """
    Main testing function
    """
    print("SPATIAL VALIDATION SYSTEM - TESTING SCRIPT")
    print("This script will help you set up and test the spatial validation system.")
    
    # Step 1: Setup reference profile
    print("\n1. REFERENCE PROFILE SETUP")
    reference_image = input("Enter path to your reference image (with all 4 nuts correctly positioned): ").strip()
    
    if reference_image and os.path.exists(reference_image):
        setup_reference_profile(reference_image)
    else:
        print("Skipping reference setup - file not found or not provided")
    
    # Step 2: Test with current image
    print("\n2. TEST IMAGE VALIDATION")
    test_image = input("Enter path to test image: ").strip()
    
    if test_image and os.path.exists(test_image):
        demonstrate_distance_calculation(test_image)
        test_image_with_spatial_validation(test_image)
    else:
        print("Skipping test - file not found or not provided")

if __name__ == "__main__":
    main()