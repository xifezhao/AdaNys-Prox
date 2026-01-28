import matplotlib.pyplot as plt
import seaborn as sns
import os

def apply_paper_style():
    """
    Configures Matplotlib to generate publication-quality figures (ICML/NeurIPS style).
    Call this at the beginning of any visualization script.
    """
    # Use seaborn whitegrid as a base for readability
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        # Fallback for older matplotlib versions
        plt.style.use('seaborn-whitegrid')

    # Global rcParams overrides for high-quality vector output
    plt.rcParams.update({
        # Fonts
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        
        # Lines & Markers
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'axes.linewidth': 1.0,
        
        # Saving
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'pdf.fonttype': 42, # TrueType fonts for editable PDFs
        'ps.fonttype': 42,
        
        # Grid
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })

# ==============================================================================
# Consistent Color & Style Mappings (For Experiment Matrix A0-A5)
# ==============================================================================

# Distinct Colorblind-Friendly Palette
PALETTE = {
    'red': '#D62728',    # AdaNys (Ours) - Primary Highlight
    'blue': '#1F77B4',   # Strong Baseline (FP32 Ideal)
    'gray': '#7F7F7F',   # Weak Baseline (SGD/AdamW)
    'orange': '#FF7F0E', # Unstable / Warning (Cliff)
    'green': '#2CA02C',  # Alternative / Control
    'purple': '#9467BD',
}

# Mapping Experiment IDs to Visual Styles
# Keys match the config filenames or logical group names
EXP_STYLE = {
    # A0: Baselines (Neutral/Background)
    'A0_Baseline': {
        'color': PALETTE['gray'],
        'linestyle': '--',
        'marker': 's', # Square
        'label': 'AdamW / SGD (Baseline)'
    },
    
    # A1: Mathematical Ideal (The Upper Bound)
    'A1_Nystrom_FP32': {
        'color': PALETTE['blue'],
        'linestyle': '-.',
        'marker': '^', # Triangle Up
        'label': 'Nyström (FP32, Atomic)'
    },
    
    # A2: Unstable (The Problem)
    'A2_Lazy_Unstable': {
        'color': PALETTE['orange'],
        'linestyle': '-',
        'marker': 'x',
        'label': 'Lazy (No Triggers)'
    },
    
    # A3: Stable (The Solution - Ours)
    'A3_Lazy_Stable': {
        'color': PALETTE['red'],
        'linestyle': '-',
        'marker': 'o', # Circle
        'linewidth': 2.5, # Thicker line for emphasis
        'label': 'AdaNys-Prox (Ours)'
    },
    
    # A4: FP8 Safe (The Reality)
    'A4_FP8_Safe': {
        'color': PALETTE['purple'],
        'linestyle': '-',
        'marker': 'D', # Diamond
        'label': 'AdaNys (FP8 Safe)'
    },
    
    # A5: The Cliff (Failure Mode)
    'A5_FP8_Cliff': {
        'color': '#8B0000', # Dark Red
        'linestyle': ':',
        'marker': 'v',
        'label': 'Quantization Cliff'
    }
}

def get_style(exp_id):
    """Retrieves style dict for a given experiment ID key."""
    # Fuzzy matching for robustness
    for key in EXP_STYLE:
        if key in exp_id or exp_id in key:
            return EXP_STYLE[key]
    
    # Fallback style
    return {
        'color': 'black',
        'linestyle': '-',
        'marker': None,
        'label': exp_id
    }

def save_plot(fig, filename, output_dir='results/figures'):
    """Helper to save figures in both PDF (Vector) and PNG (Preview) formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save PDF for LaTeX
    pdf_path = os.path.join(output_dir, f"{filename}.pdf")
    fig.savefig(pdf_path, format='pdf')
    
    # Save PNG for Slides/Preview
    png_path = os.path.join(output_dir, f"{filename}.png")
    fig.savefig(png_path, format='png', dpi=300)
    
    print(f"Saved plot to: {pdf_path}")