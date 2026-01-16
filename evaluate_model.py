import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from predict import predict_from_payload
from datetime import datetime

# Use a clean style for academic presentations
plt.style.use('seaborn-v0_8-muted') 

def calculate_gini_coefficient(allocations):
    if len(allocations) == 0 or np.sum(allocations) == 0:
        return 0.0
    allocs = np.sort(allocations)
    n = len(allocs)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * allocs)) / (n * np.sum(allocs))

def plot_evaluation_results(metrics, allocations_data, output_folder="plots"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    filename = metrics['file'].replace(".json", "")
    ids = [item['RecipientId'] for item in allocations_data]
    rl_vals = [item['rl_allocation'] for item in allocations_data]
    xgb_vals = [item['xgb_reference'] for item in allocations_data]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Evaluation Report: {metrics['file']}\nEfficiency: {metrics['budget_efficiency']:.1f}% | Gini: {metrics['gini_coefficient']:.3f}", fontsize=16)

    # 1. Bar Chart: RL vs XGB Comparison
    x = np.arange(len(ids))
    width = 0.35
    axes[0].bar(x - width/2, xgb_vals, width, label='Human (XGB) Ref', color='gray', alpha=0.5)
    axes[0].bar(x + width/2, rl_vals, width, label='RL Allocation', color='teal')
    axes[0].set_title("Allocation Comparison")
    axes[0].set_xlabel("Recipient ID")
    axes[0].set_ylabel("Amount ($)")
    axes[0].legend()

    # 2. Scatter Plot: Correlation Proof
    axes[1].scatter(xgb_vals, rl_vals, color='darkorange', s=100, edgecolors='white')
    # Add trend line
    z = np.polyfit(xgb_vals, rl_vals, 1)
    p = np.poly1d(z)
    axes[1].plot(xgb_vals, p(xgb_vals), "r--", alpha=0.5)
    axes[1].set_title(f"Alignment Correlation: {metrics['correlation_score_allocation']:.3f}")
    axes[1].set_xlabel("Human/XGB Suggestion")
    axes[1].set_ylabel("RL Final Decision")

    # 3. Lorenz Curve: Fairness Visual
    sorted_allocs = np.sort(rl_vals)
    cumsum_allocs = np.cumsum(sorted_allocs) / np.sum(sorted_allocs)
    cumsum_allocs = np.insert(cumsum_allocs, 0, 0)
    axes[2].plot(np.linspace(0, 1, len(cumsum_allocs)), cumsum_allocs, label='RL Distribution', color='purple', lw=2)
    axes[2].plot([0, 1], [0, 1], 'k--', label='Perfect Equality', alpha=0.3)
    axes[2].fill_between(np.linspace(0, 1, len(cumsum_allocs)), np.linspace(0, 1, len(cumsum_allocs)), cumsum_allocs, color='purple', alpha=0.1)
    axes[2].set_title(f"Lorenz Curve (Gini: {metrics['gini_coefficient']:.3f})")
    axes[2].set_xlabel("Cumulative Population %")
    axes[2].set_ylabel("Cumulative Allocation %")
    axes[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_folder}/{filename}_metrics.png", dpi=300)
    plt.close()

def evaluate_single_file(filepath):
    with open(filepath, 'r') as f:
        payload = json.load(f)
    
    results = predict_from_payload(payload)
    allocations = [item['rl_allocation'] for item in results['allocations']]
    xgb_scores = [item['xgb_reference'] for item in results['allocations']]
    
    params = payload.get('params', {})
    budget = params.get('budget', 0)
    
    # Calculate correlation
    if len(xgb_scores) > 1 and np.std(xgb_scores) > 1e-6 and np.std(allocations) > 1e-6:
        correlation = float(np.corrcoef(xgb_scores, allocations)[0,1])
    else:
        correlation = 0.0
    
    metrics = {
        'file': os.path.basename(filepath),
        'budget_total': budget,
        'budget_used': results['summary']['total_allocated'],
        'budget_efficiency': (results['summary']['total_allocated'] / budget * 100) if budget > 0 else 0,
        'people_helped': results['summary']['people_helped'],
        'gini_coefficient': calculate_gini_coefficient(allocations),
        'correlation_score_allocation': correlation
    }
    
    return metrics, results['allocations']

def generate_full_presentation_report(test_files):
    print("🚀 Starting Professional Evaluation...")
    all_metrics = []
    
    for filepath in test_files:
        try:
            metrics, alloc_data = evaluate_single_file(filepath)
            all_metrics.append(metrics)
            
            # Generate the charts!
            plot_evaluation_results(metrics, alloc_data)
            print(f"✅ Processed {metrics['file']} -> Chart saved in /plots")
            
        except Exception as e:
            print(f"❌ Error in {filepath}: {e}")

    # Final summary to console
    print("\n" + "="*40)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*40)
    print(f"Avg Efficiency: {np.mean([m['budget_efficiency'] for m in all_metrics]):.2f}%")
    print(f"Avg Alignment (Corr): {np.mean([m['correlation_score_allocation'] for m in all_metrics]):.3f}")
    print(f"Avg Fairness (Gini): {np.mean([m['gini_coefficient'] for m in all_metrics]):.3f}")
    print("="*40)

if __name__ == "__main__":
    test_files = ["in/example_predict.json", "in/test_case_1.json"]
    generate_full_presentation_report(test_files)