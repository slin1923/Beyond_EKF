import matplotlib.pyplot as plt
import numpy as np

# Dataset 1
data1 = {
    'WLS': {'Runtime': 25.23, 'RMSE': 38.36, 'hyperparam': None},
    'EKF': {'Runtime': 22.62, 'RMSE': 43.33, 'hyperparam': None},
    'UKF_0.1': {'Runtime': 26.44, 'RMSE': 220.74, 'hyperparam': 'λ=0.1'},
    'UKF_1': {'Runtime': 26.07, 'RMSE': 243.05, 'hyperparam': 'λ=1'},
    'UKF_10': {'Runtime': 27.49, 'RMSE': 272.41, 'hyperparam': 'λ=10'},
    'HINF_0.04': {'Runtime': 19.22, 'RMSE': 84.62, 'hyperparam': 'γ=0.04'},
    'HINF_0.2': {'Runtime': 24.33, 'RMSE': 46.54, 'hyperparam': 'γ=0.2'},
    'HINF_1': {'Runtime': 21.27, 'RMSE': 43.06, 'hyperparam': 'γ=1'},
    'PF_100': {'Runtime': 18.27, 'RMSE': 129.8, 'hyperparam': 'Np=100'},
    'PF_1000': {'Runtime': 25.3, 'RMSE': 55.98, 'hyperparam': 'Np=1000'},
    'PF_5000': {'Runtime': 46.61, 'RMSE': 42.82, 'hyperparam': 'Np=5000'}
}

# Dataset 2
data2 = {
    'WLS': {'Runtime': 19.48, 'RMSE': 10.05, 'hyperparam': None},
    'EKF': {'Runtime': 17.64, 'RMSE': 30.18, 'hyperparam': None},
    'UKF_0.1': {'Runtime': 20.26, 'RMSE': 123.45, 'hyperparam': 'λ=0.1'},
    'UKF_1': {'Runtime': 20.78, 'RMSE': 99.68, 'hyperparam': 'λ=1'},
    'UKF_10': {'Runtime': 23.31, 'RMSE': 139.59, 'hyperparam': 'λ=10'},
    'HINF_0.04': {'Runtime': 13.93, 'RMSE': 96.97, 'hyperparam': 'γ=0.04'},
    'HINF_0.2': {'Runtime': 16.93, 'RMSE': 36.39, 'hyperparam': 'γ=0.2'},
    'HINF_1': {'Runtime': 18.85, 'RMSE': 30.39, 'hyperparam': 'γ=1'},
    'PF_100': {'Runtime': 19.31, 'RMSE': 126.01, 'hyperparam': 'Np=100'},
    'PF_1000': {'Runtime': 26.6, 'RMSE': 42.41, 'hyperparam': 'Np=1000'},
    'PF_5000': {'Runtime': 42.32, 'RMSE': 20.2, 'hyperparam': 'Np=5000'}
}

# Dataset 3
data3 = {
    'WLS': {'Runtime': 13.55, 'RMSE': 20.17, 'hyperparam': None},
    'EKF': {'Runtime': 10.89, 'RMSE': 36.12, 'hyperparam': None},
    'UKF_0.1': {'Runtime': 14.72, 'RMSE': 125.23, 'hyperparam': 'λ=0.1'},
    'UKF_1': {'Runtime': 15.99, 'RMSE': 105.9, 'hyperparam': 'λ=1'},
    'UKF_10': {'Runtime': 15.79, 'RMSE': 147.28, 'hyperparam': 'λ=10'},
    'HINF_0.04': {'Runtime': 10.51, 'RMSE': 46.51, 'hyperparam': 'γ=0.04'},
    'HINF_0.2': {'Runtime': 11.43, 'RMSE': 37.14, 'hyperparam': 'γ=0.2'},
    'HINF_1': {'Runtime': 12.46, 'RMSE': 36.09, 'hyperparam': 'γ=1'},
    'PF_100': {'Runtime': 15.89, 'RMSE': 129.3, 'hyperparam': 'Np=100'},
    'PF_1000': {'Runtime': 20.11, 'RMSE': 44.31, 'hyperparam': 'Np=1000'},
    'PF_5000': {'Runtime': 34.74, 'RMSE': 26.21, 'hyperparam': 'Np=5000'}
}

def normalize_data(data):
    """Return data without normalization"""
    # Just pass through the original values
    normalized = {}
    for key, val in data.items():
        normalized[key] = {
            'Runtime_norm': val['Runtime'],
            'RMSE_norm': val['RMSE'],
            'hyperparam': val['hyperparam']
        }
    
    return normalized

def plot_dataset(data, dataset_num):
    """Create a bar chart for one dataset"""
    norm_data = normalize_data(data)
    
    # Define algorithm groups and colors
    alg_order = ['WLS', 'EKF', 'UKF_0.1', 'UKF_1', 'UKF_10', 
                 'HINF_0.04', 'HINF_0.2', 'HINF_1', 
                 'PF_100', 'PF_1000', 'PF_5000']
    
    colors = {
        'WLS': '#1f77b4',
        'EKF': '#ff7f0e',
        'UKF': '#2ca02c',
        'HINF': '#d62728',
        'PF': '#9467bd'
    }
    
    # Prepare data for plotting
    x_pos = []
    runtime_vals = []
    rmse_vals = []
    bar_colors = []
    labels = []
    
    pos = 0
    for i, key in enumerate(alg_order):
        # Add small gap between different algorithm families
        if i > 0:
            prev_alg = alg_order[i-1].split('_')[0]
            curr_alg = key.split('_')[0]
            if prev_alg != curr_alg:
                pos += 0.5
        
        x_pos.append(pos)
        runtime_vals.append(norm_data[key]['Runtime_norm'])
        rmse_vals.append(norm_data[key]['RMSE_norm'])
        
        # Determine color based on algorithm family
        alg_family = key.split('_')[0]
        bar_colors.append(colors[alg_family])
        
        # Create label
        if norm_data[key]['hyperparam']:
            labels.append(f"{key.split('_')[0]}\n{norm_data[key]['hyperparam']}")
        else:
            labels.append(key)
        
        pos += 1
    
    # Create figure with larger size for readability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot Runtime
    ax1.bar(x_pos, runtime_vals, color=bar_colors, width=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=13)
    ax1.set_ylabel('Runtime (s)', fontsize=15)
    ax1.set_title(f'Dataset {dataset_num}: Runtime', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=13)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_axisbelow(True)
    
    # Add legend to top left of runtime plot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['WLS'], label='WLS'),
        Patch(facecolor=colors['EKF'], label='EKF'),
        Patch(facecolor=colors['UKF'], label='UKF'),
        Patch(facecolor=colors['HINF'], label='H-infinity'),
        Patch(facecolor=colors['PF'], label='Particle Filter')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=11, 
               frameon=True, fancybox=True, shadow=True)
    
    # Plot RMSE
    ax2.bar(x_pos, rmse_vals, color=bar_colors, width=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=13)
    ax2.set_ylabel('RMSE (m)', fontsize=15)
    ax2.set_title(f'Dataset {dataset_num}: RMSE', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=13)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    return fig

# Create plots for all three datasets and save as PNG
fig1 = plot_dataset(data1, 1)
fig1.savefig('dataset1_performance.png', dpi=300, bbox_inches='tight')
print("Saved: dataset1_performance.png")

fig2 = plot_dataset(data2, 2)
fig2.savefig('dataset2_performance.png', dpi=300, bbox_inches='tight')
print("Saved: dataset2_performance.png")

fig3 = plot_dataset(data3, 3)
fig3.savefig('dataset3_performance.png', dpi=300, bbox_inches='tight')
print("Saved: dataset3_performance.png")

plt.close('all')
print("\nAll plots saved successfully!")