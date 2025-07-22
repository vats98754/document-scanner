import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json


def analyze_results(base_output_dir="hyperparameter_results"):
    """
    Analyze and visualize hyperparameter tuning results
    
    Args:
        base_output_dir: Directory containing hyperparameter results
    
    Returns:
        Sorted results by performance
    """
    # Load summary
    summary_file = os.path.join(base_output_dir, "hyperparameter_summary.json")
    if not os.path.exists(summary_file):
        print("No summary file found. Run hyperparameter tuning first.")
        return []
    
    with open(summary_file, "r") as f:
        results = json.load(f)
    
    # Sort by number of quadrilaterals found
    sorted_results = sorted(results, key=lambda x: x['num_quadrilaterals'], reverse=True)
    
    print("Top 10 hyperparameter combinations:")
    print("=" * 80)
    for i, result in enumerate(sorted_results[:10]):
        print(f"{i+1}. Directory: {result['directory']}")
        print(f"   Quadrilaterals found: {result['num_quadrilaterals']}")
        print(f"   Parameters: {result['parameters']}")
        print()
    
    # Create visualization of parameter effects
    param_effects = {}
    for param in ['blur_kernel', 'canny_low', 'canny_high', 'epsilon_factor', 'min_area']:
        param_effects[param] = {}
        for result in results:
            value = result['parameters'][param]
            if value not in param_effects[param]:
                param_effects[param][value] = []
            param_effects[param][value].append(result['num_quadrilaterals'])
    
    # Plot parameter effects
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (param, values) in enumerate(param_effects.items()):
        if i < len(axes):
            param_values = sorted(values.keys())
            avg_quads = [np.mean(values[v]) for v in param_values]
            
            axes[i].bar(range(len(param_values)), avg_quads)
            axes[i].set_title(f'Effect of {param}')
            axes[i].set_xlabel(param)
            axes[i].set_ylabel('Avg. Quadrilaterals Found')
            axes[i].set_xticks(range(len(param_values)))
            axes[i].set_xticklabels(param_values)
    
    # Remove empty subplot
    if len(param_effects) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_output_dir, "parameter_effects.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    return sorted_results


def visualize_top_results(base_output_dir="hyperparameter_results", top_n=6):
    """
    Visualize results from the top N hyperparameter combinations
    
    Args:
        base_output_dir: Directory containing hyperparameter results
        top_n: Number of top results to visualize
    """
    # Check if summary file exists
    summary_file = os.path.join(base_output_dir, "hyperparameter_summary.json")
    if not os.path.exists(summary_file):
        print(f"Summary file not found: {summary_file}")
        print("Please run hyperparameter tuning first!")
        return
    
    # Load summary
    with open(summary_file, "r") as f:
        results = json.load(f)
    
    # Sort by number of quadrilaterals found
    sorted_results = sorted(results, key=lambda x: x['num_quadrilaterals'], reverse=True)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, result in enumerate(sorted_results[:top_n]):
        if i < len(axes):
            # Load contours image
            img_path = os.path.join(base_output_dir, result['directory'], "contours.jpg")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                axes[i].imshow(img_rgb)
                axes[i].set_title(f"Rank {i+1}: {result['num_quadrilaterals']} quads\n"
                                f"blur={result['parameters']['blur_kernel']}, "
                                f"canny={result['parameters']['canny_low']}-{result['parameters']['canny_high']}\n"
                                f"eps={result['parameters']['epsilon_factor']}, "
                                f"area={result['parameters']['min_area']}")
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f"Image not found\n{result['directory']}", 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_output_dir, "top_results_visualization.png"), dpi=300, bbox_inches='tight')
    plt.show()


def visualize_quick_results(base_output_dir="quick_test_results"):
    """
    Visualize results from the quick hyperparameter test
    
    Args:
        base_output_dir: Directory containing quick test results
    """
    summary_file = os.path.join(base_output_dir, "hyperparameter_summary.json")
    if not os.path.exists(summary_file):
        print(f"Summary file not found: {summary_file}")
        return
    
    with open(summary_file, "r") as f:
        results = json.load(f)
    
    # Sort by number of quadrilaterals found
    sorted_results = sorted(results, key=lambda x: x['num_quadrilaterals'], reverse=True)
    
    print("Top 5 results from quick test:")
    print("=" * 60)
    for i, result in enumerate(sorted_results[:5]):
        print(f"{i+1}. Quadrilaterals: {result['num_quadrilaterals']}")
        print(f"   Parameters: {result['parameters']}")
        print(f"   Directory: {result['directory']}")
        print()
    
    # Create visualization of top 6 results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, result in enumerate(sorted_results[:6]):
        if i < len(axes):
            # Load contours image
            img_path = os.path.join(base_output_dir, result['directory'], "contours.jpg")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                axes[i].imshow(img_rgb)
                axes[i].set_title(f"Rank {i+1}: {result['num_quadrilaterals']} quads\n"
                                f"blur={result['parameters']['blur_kernel']}, "
                                f"canny={result['parameters']['canny_low']}-{result['parameters']['canny_high']}\n"
                                f"eps={result['parameters']['epsilon_factor']}, "
                                f"area={result['parameters']['min_area']}")
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, "Image not found", 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_output_dir, "quick_test_visualization.png"), dpi=300, bbox_inches='tight')
    plt.show()


def compare_hyperparameter_effects(base_output_dir="hyperparameter_results"):
    """
    Create detailed comparison of hyperparameter effects
    
    Args:
        base_output_dir: Directory containing hyperparameter results
    """
    summary_file = os.path.join(base_output_dir, "hyperparameter_summary.json")
    if not os.path.exists(summary_file):
        print("No summary file found. Run hyperparameter tuning first.")
        return
    
    with open(summary_file, "r") as f:
        results = json.load(f)
    
    # Create parameter effect analysis
    param_stats = {}
    for param in ['blur_kernel', 'canny_low', 'canny_high', 'epsilon_factor', 'min_area']:
        param_stats[param] = {}
        for result in results:
            value = result['parameters'][param]
            if value not in param_stats[param]:
                param_stats[param][value] = []
            param_stats[param][value].append(result['num_quadrilaterals'])
    
    # Calculate statistics
    print("Parameter Effect Analysis:")
    print("=" * 50)
    
    for param, values in param_stats.items():
        print(f"\n{param.upper()}:")
        for value, quad_counts in values.items():
            mean_quads = np.mean(quad_counts)
            std_quads = np.std(quad_counts)
            print(f"  {value}: {mean_quads:.2f} ± {std_quads:.2f} quadrilaterals")
    
    return param_stats
