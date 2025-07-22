import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import product
import json


def document_scanner_with_hyperparams(image_path, blur_kernel, canny_low, canny_high, 
                                     epsilon_factor, min_area, save_dir):
    """
    Document scanner with configurable hyperparameters
    
    Args:
        image_path: Path to input image
        blur_kernel: Gaussian blur kernel size (odd number)
        canny_low: Lower threshold for Canny edge detection
        canny_high: Upper threshold for Canny edge detection
        epsilon_factor: Factor for contour approximation (0.01-0.1)
        min_area: Minimum area threshold for quadrilaterals
        save_dir: Directory to save results
    
    Returns:
        Number of quadrilaterals found, result images
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return 0, None, None, None
    
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Preprocess with hyperparameters
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter quadrilaterals with hyperparameters
    quads = []
    
    for cnt in contours:
        # Approximate contour
        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Check if it's a quadrilateral
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > min_area:
                quads.append((area, approx))
    
    # Sort by area, largest first
    quads = sorted(quads, key=lambda x: x[0], reverse=True)
    
    # Create visualizations
    result_original = original.copy()
    result_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    result_contours = original.copy()
    
    # Draw all plausible quadrilaterals
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for i, (area, quad) in enumerate(quads[:5]):  # Show top 5 quads
        color = colors[i % len(colors)]
        cv2.drawContours(result_contours, [quad], -1, color, 3)
        
        # Label corners
        for j, corner in enumerate(quad):
            pt = tuple(corner[0])
            cv2.circle(result_contours, pt, 8, color, -1)
            cv2.putText(result_contours, f"{i+1}-{j+1}", (pt[0]+10, pt[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    
    # Save images
    cv2.imwrite(os.path.join(save_dir, "original.jpg"), result_original)
    cv2.imwrite(os.path.join(save_dir, "edges.jpg"), result_edges)
    cv2.imwrite(os.path.join(save_dir, "contours.jpg"), result_contours)
    cv2.imwrite(os.path.join(save_dir, "blurred.jpg"), blurred)
    
    # Save hyperparameters and results
    results = {
        "hyperparameters": {
            "blur_kernel": blur_kernel,
            "canny_low": canny_low,
            "canny_high": canny_high,
            "epsilon_factor": epsilon_factor,
            "min_area": min_area
        },
        "results": {
            "num_quadrilaterals": len(quads),
            "quad_areas": [float(area) for area, _ in quads],
            "image_size": img.shape[:2]
        }
    }
    
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return len(quads), result_original, result_edges, result_contours


def hyperparameter_tuning(image_path, base_output_dir="hyperparameter_results"):
    """
    Perform hyperparameter tuning for document scanner
    
    Args:
        image_path: Path to input image
        base_output_dir: Base directory for saving results
    
    Returns:
        Tuple of (results_summary, best_result)
    """
    # Define hyperparameter ranges
    hyperparams = {
        'blur_kernel': [3, 5, 7, 9],  # Gaussian blur kernel sizes
        'canny_low': [30, 50, 70, 100],  # Lower Canny threshold
        'canny_high': [100, 150, 200, 250],  # Upper Canny threshold
        'epsilon_factor': [0.01, 0.02, 0.03, 0.05],  # Contour approximation factor
        'min_area': [500, 1000, 2000, 5000]  # Minimum area threshold
    }
    
    # Create base output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Generate all combinations
    param_combinations = list(product(*hyperparams.values()))
    param_names = list(hyperparams.keys())
    
    print(f"Testing {len(param_combinations)} hyperparameter combinations...")
    
    results_summary = []
    
    for i, params in enumerate(param_combinations):
        # Create parameter dictionary
        param_dict = dict(zip(param_names, params))
        
        # Create directory name from parameters
        dir_name = f"blur{param_dict['blur_kernel']}_canny{param_dict['canny_low']}-{param_dict['canny_high']}_eps{param_dict['epsilon_factor']}_area{param_dict['min_area']}"
        save_dir = os.path.join(base_output_dir, dir_name)
        
        # Run scanner with these parameters
        num_quads, original, edges, contours = document_scanner_with_hyperparams(
            image_path, 
            param_dict['blur_kernel'],
            param_dict['canny_low'],
            param_dict['canny_high'],
            param_dict['epsilon_factor'],
            param_dict['min_area'],
            save_dir
        )
        
        # Store results
        results_summary.append({
            'combination': i + 1,
            'parameters': param_dict,
            'num_quadrilaterals': num_quads,
            'directory': dir_name
        })
        
        # Print progress
        if (i + 1) % 50 == 0:
            print(f"Completed {i + 1}/{len(param_combinations)} combinations")
    
    # Save summary results
    with open(os.path.join(base_output_dir, "hyperparameter_summary.json"), "w") as f:
        json.dump(results_summary, f, indent=2)
    
    # Find best parameters (most quadrilaterals found)
    best_result = max(results_summary, key=lambda x: x['num_quadrilaterals'])
    
    print(f"\nHyperparameter tuning completed!")
    print(f"Total combinations tested: {len(param_combinations)}")
    print(f"Best result: {best_result['num_quadrilaterals']} quadrilaterals found")
    print(f"Best parameters: {best_result['parameters']}")
    print(f"Best result directory: {best_result['directory']}")
    
    return results_summary, best_result


def quick_hyperparameter_test(image_path, base_output_dir="quick_test_results"):
    """
    Quick test with a smaller set of hyperparameters
    
    Args:
        image_path: Path to input image
        base_output_dir: Base directory for saving results
    
    Returns:
        Tuple of (results_summary, best_result)
    """
    # Define smaller hyperparameter ranges for quick testing
    hyperparams = {
        'blur_kernel': [3, 5, 7],  # 3 values
        'canny_low': [30, 50],     # 2 values
        'canny_high': [100, 150],  # 2 values
        'epsilon_factor': [0.02, 0.03],  # 2 values
        'min_area': [1000, 2000]   # 2 values
    }
    
    # Create base output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Generate all combinations (3×2×2×2×2 = 48 combinations)
    param_combinations = list(product(*hyperparams.values()))
    param_names = list(hyperparams.keys())
    
    print(f"Quick testing {len(param_combinations)} hyperparameter combinations...")
    
    results_summary = []
    
    for i, params in enumerate(param_combinations):
        # Create parameter dictionary
        param_dict = dict(zip(param_names, params))
        
        # Create directory name from parameters
        dir_name = f"blur{param_dict['blur_kernel']}_canny{param_dict['canny_low']}-{param_dict['canny_high']}_eps{param_dict['epsilon_factor']}_area{param_dict['min_area']}"
        save_dir = os.path.join(base_output_dir, dir_name)
        
        # Run scanner with these parameters
        num_quads, original, edges, contours = document_scanner_with_hyperparams(
            image_path, 
            param_dict['blur_kernel'],
            param_dict['canny_low'],
            param_dict['canny_high'],
            param_dict['epsilon_factor'],
            param_dict['min_area'],
            save_dir
        )
        
        # Store results
        results_summary.append({
            'combination': i + 1,
            'parameters': param_dict,
            'num_quadrilaterals': num_quads,
            'directory': dir_name
        })
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{len(param_combinations)} combinations")
    
    # Save summary results
    with open(os.path.join(base_output_dir, "hyperparameter_summary.json"), "w") as f:
        json.dump(results_summary, f, indent=2)
    
    # Find best parameters
    best_result = max(results_summary, key=lambda x: x['num_quadrilaterals'])
    
    print(f"\nQuick test completed!")
    print(f"Total combinations tested: {len(param_combinations)}")
    print(f"Best result: {best_result['num_quadrilaterals']} quadrilaterals found")
    print(f"Best parameters: {best_result['parameters']}")
    
    return results_summary, best_result
