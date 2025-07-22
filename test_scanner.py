import os
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from document_scanner import test_scanner, simple_quadrilateral_detection, harris_corner_detection
from hyperparameter_tuning import hyperparameter_tuning, quick_hyperparameter_test
from analysis import analyze_results, visualize_top_results, visualize_quick_results


def main():
    # Test image paths
    test_images = [
        "/Users/anvay-coder/document-scanner/document_1_1752773781461.jpg",
        "/Users/anvay-coder/document-scanner/document_2_1752773778776.jpg"
    ]
    
    print("Document Scanner Test Suite")
    print("=" * 50)
    
    # Test basic document scanner
    print("\n1. Testing basic document scanner...")
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"Processing: {os.path.basename(img_path)}")
            original, corners_viz, scanned = test_scanner(img_path)
            if scanned is not None:
                print("✓ Document scanner completed successfully")
            break
    
    # Test simple quadrilateral detection
    print("\n2. Testing simple quadrilateral detection...")
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"Processing: {os.path.basename(img_path)}")
            quads, result_img = simple_quadrilateral_detection(img_path)
            print(f"Found {len(quads)} quadrilaterals")
            break
    
    # Test Harris corner detection
    print("\n3. Testing Harris corner detection...")
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"Processing: {os.path.basename(img_path)}")
            corner_img = harris_corner_detection(img_path)
            if corner_img is not None:
                print("✓ Harris corner detection completed")
            break
    
    # Test quick hyperparameter tuning
    print("\n4. Testing quick hyperparameter tuning...")
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"Running quick test on: {os.path.basename(img_path)}")
            results, best = quick_hyperparameter_test(img_path)
            print(f"✓ Quick test completed. Best result: {best['num_quadrilaterals']} quadrilaterals")
            break
    
    # Analyze results if available
    print("\n5. Analyzing results...")
    if os.path.exists("quick_test_results/hyperparameter_summary.json"):
        visualize_quick_results("quick_test_results")
        print("✓ Quick test analysis completed")
    
    print("\nTest suite completed!")


if __name__ == "__main__":
    main()
