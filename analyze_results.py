import os
import glob
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from src.utils.plot_style import apply_paper_style, PALETTE, save_plot

# Apply Publication Quality Style
apply_paper_style()

RESULTS_DIR = "results/metrics"
FIGURES_DIR = "results/figures"

def load_jsonl_data(filepath):
    """Reads a JSONL file into a Pandas DataFrame."""
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Warning: File not found {filepath}")
        return pd.DataFrame()
        
    return pd.DataFrame(data)

def find_latest_log(exp_group_dir):
    """Finds the most recent .jsonl file in a specific experiment directory."""
    search_path = os.path.join(exp_group_dir, "*.jsonl")
    files = glob.glob(search_path)
    if not files:
        return None
    # Sort by modification time
    return max(files, key=os.path.getmtime)

# ==============================================================================
# 1. Quantization Cliff Analysis (Recreating Paper Figure 3 / Appendix C)
# ==============================================================================
def analyze_cliff():
    print(">>> Generating Cliff Analysis Plot...")
    base_dir = os.path.join(RESULTS_DIR, "A5_fp8_cliff")
    
    # We expect subdirectories or files like "A5_FP8_Cliff_Delta1e-09_..."
    # Since run_matrix creates distinct IDs but might map to same output dir depending on config,
    # let's assume run_matrix saves all runs into the same base output_dir with distinct filenames 
    # OR distinct directories. Based on run_matrix.py, it uses config['output_dir'].
    # Let's scan for all jsonl files in the A5 directory.
    
    log_files = glob.glob(os.path.join(base_dir, "*.jsonl"))
    if not log_files:
        print(f"  [Skipped] No logs found in {base_dir}")
        return

    summary_data = []

    for log_file in log_files:
        # Extract Delta from filename (e.g., ..._Deltan1e09_...)
        match = re.search(r"Delta(\d+e-\d+|1e-09|1e-\d+)", os.path.basename(log_file).replace('n', '-'))
        # Alternative regex if 'n' replacement happened in run_matrix: "Delta1e-09"
        
        if not match:
            # Fallback: try parsing metadata
            meta_file = log_file.replace('.jsonl', '_meta.json')
            if os.path.exists(meta_file):
                with open(meta_file) as f:
                    cfg = json.load(f)
                    delta = float(cfg['adanys']['delta'])
            else:
                continue
        else:
            delta = float(match.group(1))

        df = load_jsonl_data(log_file)
        if df.empty: continue

        # Metrics at the end of training
        final_loss = df['loss'].iloc[-1]
        max_jitter = df['jitter'].max() if 'jitter' in df.columns else 0.0
        # Check if divergence happened (Loss > 100 or NaN)
        is_diverged = final_loss > 100 or np.isnan(final_loss)
        
        summary_data.append({
            'delta': delta,
            'final_loss': final_loss,
            'max_jitter': max_jitter,
            'is_diverged': is_diverged
        })

    if not summary_data:
        return

    res_df = pd.DataFrame(summary_data).sort_values('delta')

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Left Axis: Loss
    ax1.set_xlabel(r'Kernel Regularization $\delta$ (Log Scale)')
    ax1.set_ylabel('Final Training Loss', color=PALETTE['blue'])
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    line1 = ax1.plot(res_df['delta'], res_df['final_loss'], marker='o', 
                     color=PALETTE['blue'], label='Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=PALETTE['blue'])

    # Right Axis: Jitter
    ax2 = ax1.twinx()
    ax2.set_ylabel('Max Jitter Injected (Diagnosis)', color=PALETTE['red'])
    # Jitter is 0 or >0. Use linear or symlog. 
    # Since jitter spikes to 1e-2, linear is fine, or log if range is huge.
    line2 = ax2.plot(res_df['delta'], res_df['max_jitter'], marker='x', 
                     linestyle='--', color=PALETTE['red'], label='Jitter Needed')
    ax2.tick_params(axis='y', labelcolor=PALETTE['red'])
    ax2.set_ylim(-0.001, 0.02) # Cap to visualize the 0 vs >0 clearly

    # Annotation: The Cliff
    cliff_x = 1e-7 # Theoretical expectation
    ax1.axvline(x=cliff_x, color='gray', linestyle=':', alpha=0.5)
    ax1.text(cliff_x * 0.1, ax1.get_ylim()[1]*0.5, "Quantization\nCliff", 
             color='red', fontweight='bold', ha='center')

    plt.title('The Quantization Cliff: $\delta$ vs. Numerical Stability')
    
    # Combined Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center')

    save_plot(fig, "Figure2_Quantization_Cliff")


# ==============================================================================
# 2. Stability Analysis (Recreating Figure 2b)
# ==============================================================================
def analyze_stability():
    print(">>> Generating Stability Analysis Plot...")
    
    # 1. Load Unstable Run (A2 logic)
    # Note: run_matrix saves these into specific dirs based on ID or config output_dir
    # We look for the "TriggersOFF" run generated by the matrix loop
    unstable_file = find_latest_log("results/metrics/A3_lazy_stable") # Assuming matrix saves here with suffix
    # Wait, run_matrix appends suffix to ID, but output_dir is from config. 
    # If run_matrix doesn't change output_dir dynamically, files are mixed.
    # Let's search by filename pattern in the A3 directory.
    
    base_dir = "results/metrics/A3_lazy_stable"
    files = glob.glob(os.path.join(base_dir, "*.jsonl"))
    
    df_stable = pd.DataFrame()
    df_unstable = pd.DataFrame()
    
    for f in files:
        if "TriggersON" in f:
            df_stable = load_jsonl_data(f)
        elif "TriggersOFF" in f:
            df_unstable = load_jsonl_data(f)
            
    if df_stable.empty or df_unstable.empty:
        print("  [Skipped] Missing Stable/Unstable logs. Run 'run_matrix.py' with A3.")
        return

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot Unstable
    ax.plot(df_unstable['wall_clock_real'], df_unstable['loss'], 
            color=PALETTE['orange'], linestyle='--', label=r'Lazy ($\tau=50$) No Triggers')
    
    # Plot Stable
    ax.plot(df_stable['wall_clock_real'], df_stable['loss'], 
            color=PALETTE['red'], linewidth=2, label=r'AdaNys ($\tau=50$) + Triggers')
    
    # Mark Trigger Events
    triggers = df_stable[df_stable['trigger_fired'] == True]
    if not triggers.empty:
        ax.scatter(triggers['wall_clock_real'], triggers['loss'], 
                   color='black', marker='v', s=50, zorder=5, label='Force Update')
        
        # Annotate first trigger
        first_t = triggers.iloc[0]
        ax.annotate(f"Trigger: {first_t['trigger_reason']}", 
                    xy=(first_t['wall_clock_real'], first_t['loss']),
                    xytext=(first_t['wall_clock_real']+5, first_t['loss']+0.2),
                    arrowprops=dict(facecolor='black', arrowstyle='->'))

    ax.set_xlabel('Simulated Wall-Clock Time (s)')
    ax.set_ylabel('Training Loss (Log Scale)')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('Stability Triggers: Recovering from Hessian Staleness')
    
    save_plot(fig, "Figure3_Stability_Triggers")


# ==============================================================================
# 3. System Efficiency (Bandwidth Breakdown)
# ==============================================================================
def analyze_system_efficiency():
    print(">>> Generating System Efficiency Plot...")
    
    # Load Baseline (A0)
    file_a0 = find_latest_log("results/metrics/A0_baseline")
    # Load Ours (A3 or A4)
    file_ours = find_latest_log("results/metrics/A3_lazy_stable") # Representative run
    
    if not file_a0 or not file_ours:
        print("  [Skipped] Missing logs for System Analysis.")
        return
        
    df_a0 = load_jsonl_data(file_a0)
    df_ours = load_jsonl_data(file_ours)
    
    # Extract cumulative bytes at step 100 (arbitrary snapshot)
    step_snapshot = min(len(df_a0), len(df_ours)) - 1
    
    # Baseline: SGD sends Gradients (O(d)) every step
    # run_matrix logic puts this in 'total_effective_bytes'
    bytes_sgd = df_a0.iloc[step_snapshot].get('total_comm_bytes', 0)
    
    # Ours: Sends Gram (O(m^2)) + Projections (O(m))
    bytes_ours = df_ours.iloc[step_snapshot].get('total_comm_bytes', 0)
    
    # Convert to GB
    gb_sgd = bytes_sgd / 1e9
    gb_ours = bytes_ours / 1e9
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(6, 5))
    
    labels = ['Dist. SGD / AdamW', 'AdaNys-Prox (Ours)']
    values = [gb_sgd, gb_ours]
    colors = [PALETTE['gray'], PALETTE['red']]
    
    bars = ax.bar(labels, values, color=colors, width=0.6)
    
    # Log scale if difference is huge (likely)
    ax.set_yscale('log')
    
    # Add text labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., 1.05*height,
                f'{val:.4f} GB',
                ha='center', va='bottom', fontweight='bold')
                
    ax.set_ylabel('Total Comm. Volume (GB) [Log Scale]')
    ax.set_title(f'Communication Overhead (Simulated @ Step {step_snapshot})')
    
    # Annotation for Reduction
    if gb_ours > 0:
        reduction = gb_sgd / gb_ours
        ax.text(0.5, 0.5, f"{reduction:.0f}x Reduction", 
                transform=ax.transAxes, ha='center', color='red', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))

    save_plot(fig, "Figure1_System_Efficiency")

if __name__ == "__main__":
    print("="*60)
    print("📊 Generating Platinum Report Graphics")
    print("="*60)
    
    analyze_cliff()
    analyze_stability()
    analyze_system_efficiency()
    
    print("\nDone. Check results/figures/ for PDF output.")