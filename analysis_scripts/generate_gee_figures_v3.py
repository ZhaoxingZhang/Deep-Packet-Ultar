#!/usr/bin/env python3
"""
Generate GEE Architecture Figures - Version 3

Professional academic figures inspired by DeepSeek architecture diagram.
Key design principles:
- Clean, modular layout with clear separation
- Professional color scheme (based on reference image analysis)
- Precise alignment and spacing
- Clear visual hierarchy
- High readability at publication resolution
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Configuration based on reference image
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 8
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.unicode_minus'] = False

# Professional color scheme (inspired by DeepSeek and reference analysis)
COLORS = {
    # Component colors (soft, academic)
    'baseline': '#E8D4B8',      # Warm tan - for baseline components
    'expert_light': '#FDE8C6',  # Light orange - for routed experts
    'expert_shared': '#FEF0D4', # Very light - for shared experts
    'gating': '#D4E5F7',        # Soft blue - for gating/attention
    'data': '#FEFEFE',          # White - for data/I/O
    'special': '#FFD54F',       # Golden - for special components

    # Structural colors
    'border': '#2072B8',        # Deep blue - module boundaries
    'border_dark': '#15548A',   # Darker blue - stronger borders
    'text_main': '#000000',     # Black - primary text
    'text_sub': '#444444',      # Dark gray - secondary text
    'arrow': '#333333',         # Dark gray - connections
}

def setup_figure(figsize=(14, 10)):
    """Setup figure with proper margins"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, figsize[0])
    ax.set_ylim(0, figsize[1])
    ax.axis('off')
    return fig, ax

def draw_module_boundary(ax, x, y, width, height, title, subtitle=''):
    """
    Draw a module boundary with dashed border (like DeepSeek diagram)
    This creates the blue dashed boxes that group related components
    """
    # Main boundary box
    boundary = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=6",
                             edgecolor=COLORS['border'],
                             facecolor='none',
                             linewidth=1.8,
                             linestyle='--')
    ax.add_patch(boundary)

    # Title
    ax.text(x + width/2, y + height - 0.35, title,
            ha='center', va='top',
            fontsize=10, fontweight='bold',
            color=COLORS['border'])

    # Optional subtitle
    if subtitle:
        ax.text(x + width/2, y + height - 0.65, subtitle,
                ha='center', va='top',
                fontsize=7, style='italic',
                color=COLORS['text_sub'])

    return y + height - 0.8

def draw_component_box(ax, x, y, width, height, text,
                       color, fontsize=8, bold=False):
    """
    Draw a component/layer box (the colored rectangles inside modules)
    """
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=3",
                         edgecolor='black',
                         facecolor=color,
                         linewidth=1.0)
    ax.add_patch(box)

    fontweight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=fontsize,
            fontweight=fontweight,
            color=COLORS['text_main'])

def draw_data_sequence(ax, x, y, n_items, label=''):
    """
    Draw a horizontal sequence of circles representing vector/tensor data
    Like the hidden state representations in DeepSeek diagram
    """
    radius = 0.10
    spacing = 0.30
    total_width = n_items * spacing

    for i in range(n_items):
        circle = Circle((x + i*spacing, y), radius,
                       edgecolor='#555555',
                       facecolor=COLORS['data'],
                       linewidth=0.8)
        ax.add_patch(circle)

    if label:
        ax.text(x + total_width/2, y - 0.28, label,
                ha='center', va='top',
                fontsize=7, style='italic',
                color=COLORS['text_sub'])

    return total_width

def draw_connection(ax, x1, y1, x2, y2, label='', color=None):
    """
    Draw an arrow connection between components
    """
    if color is None:
        color = COLORS['arrow']

    # Calculate control points for smooth curve
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx**2 + dy**2)

    # Use slight curvature for aesthetic
    if dist > 3:
        connection = "arc3,rad=0.05"
    else:
        connection = "arc3,rad=0"

    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->',
                           mutation_scale=12,
                           connectionstyle=connection,
                           color=color,
                           linewidth=1.5)
    ax.add_patch(arrow)

    if label:
        mid_x, mid_y = (x1 + x2)/2, (y1 + y2)/2
        # Background for label readability
        ax.text(mid_x, mid_y + 0.18, label,
                ha='center', va='bottom',
                fontsize=6.5,
                bbox=dict(boxstyle="round,pad=2.5",
                         facecolor='white',
                         edgecolor='none',
                         alpha=0.85))

# =============================================================================
# Figure 3-1: Architecture Overview
# =============================================================================

def create_figure_3_1():
    """Figure 3-1: GEE Architecture Overview - Clean and Professional"""
    fig, ax = setup_figure()

    # Title
    ax.text(7, 9.7, 'Figure 3-1: GEE Architecture Overview',
            ha='center', va='top', fontsize=14, fontweight='bold')

    # === Input ===
    draw_component_box(ax, 0.5, 4.25, 1.3, 1.3,
                      'Input\nPacket',
                      COLORS['data'], fontsize=9)
    ax.text(1.15, 4.35, '(1500 B)',
            ha='center', fontsize=7, color=COLORS['text_sub'])

    # === Baseline Model Module (Top) ===
    draw_module_boundary(ax, 2.3, 7.0, 3.2, 2.8,
                        'Baseline Model',
                        'Trained on all 6 classes')

    # ResNet block
    draw_component_box(ax, 2.5, 9.1, 2.8, 0.7,
                      'ResNet1d',
                      COLORS['baseline'], fontsize=10, bold=True)
    ax.text(3.9, 9.45, '(16 BasicBlocks)',
            ha='center', fontsize=7, color=COLORS['text_sub'])

    # FC layers
    draw_component_box(ax, 2.5, 8.1, 2.8, 0.55,
                      'FC: 200 -> 100 -> 50 -> 6',
                      COLORS['baseline'], fontsize=9)

    # Output probability vector
    draw_data_sequence(ax, 2.7, 7.4, 6, 'p_baseline')

    # === Expert Model Module (Bottom) ===
    draw_module_boundary(ax, 2.3, 3.2, 3.2, 3.2,
                        'Minority Expert Model',
                        'Trained on classes 5 & 7 only')

    # ResNet block
    draw_component_box(ax, 2.5, 5.7, 2.8, 0.7,
                      'ResNet1d',
                      COLORS['expert_light'], fontsize=10, bold=True)
    ax.text(3.9, 6.05, '(16 BasicBlocks)',
            ha='center', fontsize=7, color=COLORS['text_sub'])

    # FC layers
    draw_component_box(ax, 2.5, 4.7, 2.8, 0.55,
                      'FC: 200 -> 100 -> 50 -> 2',
                      COLORS['expert_light'], fontsize=9)

    # Output probability vector (2 classes)
    draw_data_sequence(ax, 2.7, 4.0, 2, 'p_expert')

    # Note about extension
    ax.text(3.9, 3.6,
            'Zero-extended to ℝ⁶',
            ha='center', fontsize=7, style='italic',
            color='#D32F2F')

    # === Gating Network Module (Right) ===
    draw_module_boundary(ax, 6.2, 4.3, 3.0, 4.2,
                        'Gating Network',
                        'Learns optimal fusion strategy')

    # Layers (vertical stack)
    draw_component_box(ax, 6.4, 7.7, 2.6, 0.55,
                      'Concatenate',
                      COLORS['gating'], fontsize=9)
    ax.text(7.7, 7.97, 'Input: 12 dims',
            ha='center', fontsize=6.5, color=COLORS['text_sub'])

    draw_component_box(ax, 6.4, 6.95, 2.6, 0.55,
                      'Linear(128)',
                      COLORS['gating'], fontsize=9)
    draw_component_box(ax, 6.4, 6.2, 2.6, 0.55,
                      'ReLU + Dropout',
                      COLORS['gating'], fontsize=8)

    draw_component_box(ax, 6.4, 5.45, 2.6, 0.55,
                      'Linear(64)',
                      COLORS['gating'], fontsize=9)
    draw_component_box(ax, 6.4, 4.7, 2.6, 0.55,
                      'ReLU + Dropout',
                      COLORS['gating'], fontsize=8)

    draw_component_box(ax, 6.4, 4.0, 2.6, 0.5,
                      'Linear(6)',
                      COLORS['gating'], fontsize=9)

    # Weighted CE loss highlight
    draw_component_box(ax, 6.4, 4.55, 2.6, 0.45,
                      'Weighted CE Loss',
                      COLORS['special'], fontsize=8, bold=True)

    # === Output ===
    draw_component_box(ax, 9.8, 5.5, 1.5, 1.3,
                      'Final\nPrediction',
                      COLORS['data'], fontsize=9)
    draw_data_sequence(ax, 10.0, 5.1, 6, 'y_pred')

    # === Connection Arrows ===

    # Input to Baseline
    draw_connection(ax, 1.8, 4.9, 2.5, 9.1)

    # Input to Expert
    draw_connection(ax, 1.8, 4.9, 2.5, 5.7)

    # Baseline to Gating (top path)
    draw_connection(ax, 5.1, 8.0, 6.4, 7.7)

    # Expert to Gating (bottom path, with note)
    draw_connection(ax, 5.1, 5.3, 6.4, 5.7,
                   label='Extended')

    # Gating to Output
    draw_connection(ax, 9.2, 6.4, 9.8, 6.15)

    # === Legend (bottom) ===
    ax.text(0.5, 1.8, 'Legend:',
            fontsize=9, fontweight='bold')

    legend_items = [
        (COLORS['baseline'], 'Baseline Component'),
        (COLORS['expert_light'], 'Expert Component'),
        (COLORS['gating'], 'Gating Component'),
    ]

    for i, (color, label) in enumerate(legend_items):
        y = 1.5 - i * 0.3
        # Color swatch
        rect = Rectangle((1.4, y - 0.08), 0.22, 0.18,
                        facecolor=color, edgecolor='black', linewidth=0.8)
        ax.add_patch(rect)
        # Label
        ax.text(1.7, y, label, ha='left', va='center', fontsize=7.5)

    # === Key Innovations (bottom right) ===
    ax.text(6.5, 1.8, 'Key Innovations:',
            fontsize=9, fontweight='bold')

    innovations = [
        'Learnable fusion via gating network',
        'Weighted CE for class balance',
        'Incremental architecture design',
    ]

    for i, text in enumerate(innovations):
        y = 1.5 - i * 0.25
        ax.text(6.5, y, f'• {text}',
                ha='left', va='center', fontsize=7.5)

    plt.tight_layout()
    return fig

# =============================================================================
# Figure 3-2: Detailed Component Architecture
# =============================================================================

def create_figure_3_2():
    """Figure 3-2: Detailed Architecture - Layer-by-layer breakdown"""
    fig, ax = setup_figure(figsize=(16, 8))

    # Title
    ax.text(8, 7.5, 'Figure 3-2: Detailed Component Architecture',
            ha='center', va='top', fontsize=14, fontweight='bold')

    # === Baseline Model (Left) ===
    draw_module_boundary(ax, 0.5, 5.2, 4.6, 2.0,
                        'Baseline Model: ResNet1d',
                        'Output: 6 classes')

    # Layer sequence (left to right)
    draw_component_box(ax, 0.7, 6.7, 0.9, 0.45,
                      'Input',
                      COLORS['data'], fontsize=8)

    draw_component_box(ax, 1.8, 6.7, 1.0, 0.45,
                      'Conv1d',
                      COLORS['baseline'], fontsize=8)
    draw_connection(ax, 1.6, 6.92, 1.8, 6.92)

    draw_component_box(ax, 3.0, 6.7, 1.6, 0.45,
                      'BasicBlock×16',
                      COLORS['baseline'], fontsize=9, bold=True)
    draw_connection(ax, 2.8, 6.92, 3.0, 6.92)

    # Second row
    draw_component_box(ax, 0.7, 6.0, 1.1, 0.45,
                      'GlobalPool',
                      COLORS['baseline'], fontsize=8)

    draw_component_box(ax, 2.0, 6.0, 0.9, 0.45,
                      'FC(200)',
                      COLORS['baseline'], fontsize=8)
    draw_connection(ax, 1.8, 6.22, 2.0, 6.22)

    draw_component_box(ax, 3.1, 6.0, 0.9, 0.45,
                      'FC(100)',
                      COLORS['baseline'], fontsize=8)
    draw_connection(ax, 2.9, 6.22, 3.1, 6.22)

    # Third row
    draw_component_box(ax, 0.7, 5.3, 1.1, 0.45,
                      'FC(50)',
                      COLORS['baseline'], fontsize=8)

    draw_component_box(ax, 2.0, 5.3, 0.9, 0.45,
                      'FC(6)',
                      COLORS['baseline'], fontsize=8)
    draw_connection(ax, 1.8, 5.52, 2.0, 5.52)

    draw_component_box(ax, 3.1, 5.3, 1.6, 0.45,
                      'Softmax',
                      COLORS['data'], fontsize=8)
    draw_connection(ax, 2.9, 5.52, 3.1, 5.52)

    # === Expert Model (Middle) ===
    draw_module_boundary(ax, 5.7, 5.2, 4.6, 2.0,
                        'Expert Model: ResNet1d',
                        'Output: 2 classes (minority)')

    # Layer sequence
    draw_component_box(ax, 5.9, 6.7, 0.9, 0.45,
                      'Input',
                      COLORS['data'], fontsize=8)

    draw_component_box(ax, 7.0, 6.7, 1.0, 0.45,
                      'Conv1d',
                      COLORS['expert_light'], fontsize=8)
    draw_connection(ax, 6.8, 6.92, 7.0, 6.92)

    draw_component_box(ax, 8.2, 6.7, 1.6, 0.45,
                      'BasicBlock×16',
                      COLORS['expert_light'], fontsize=9, bold=True)
    draw_connection(ax, 8.0, 6.92, 8.2, 6.92)

    # Second row
    draw_component_box(ax, 5.9, 6.0, 1.1, 0.45,
                      'GlobalPool',
                      COLORS['expert_light'], fontsize=8)

    draw_component_box(ax, 7.2, 6.0, 0.9, 0.45,
                      'FC(200)',
                      COLORS['expert_light'], fontsize=8)
    draw_connection(ax, 7.0, 6.22, 7.2, 6.22)

    draw_component_box(ax, 8.3, 6.0, 0.9, 0.45,
                      'FC(100)',
                      COLORS['expert_light'], fontsize=8)
    draw_connection(ax, 8.1, 6.22, 8.3, 6.22)

    # Third row
    draw_component_box(ax, 5.9, 5.3, 1.1, 0.45,
                      'FC(50)',
                      COLORS['expert_light'], fontsize=8)

    draw_component_box(ax, 7.2, 5.3, 0.9, 0.45,
                      'FC(2)',
                      COLORS['expert_light'], fontsize=8)
    draw_connection(ax, 7.0, 5.52, 7.2, 5.52)

    draw_component_box(ax, 8.3, 5.3, 1.6, 0.45,
                      'Softmax',
                      COLORS['data'], fontsize=8)
    draw_connection(ax, 8.1, 5.52, 8.3, 5.52)

    # === Gating Network (Right) ===
    draw_module_boundary(ax, 10.9, 5.2, 4.8, 2.0,
                        'Gating Network: MLP',
                        '~2,500 parameters')

    # Layer sequence
    draw_component_box(ax, 11.1, 6.7, 2.0, 0.45,
                      'Concat(12)',
                      COLORS['gating'], fontsize=8)

    draw_component_box(ax, 13.3, 6.7, 2.0, 0.45,
                      'Linear(128)',
                      COLORS['gating'], fontsize=8)
    draw_connection(ax, 13.1, 6.92, 13.3, 6.92)

    # Second row
    draw_component_box(ax, 11.1, 6.0, 2.0, 0.45,
                      'ReLU + Dropout',
                      COLORS['gating'], fontsize=8)
    draw_connection(ax, 12.1, 6.22, 11.1, 6.22)

    draw_component_box(ax, 13.3, 6.0, 2.0, 0.45,
                      'Linear(64)',
                      COLORS['gating'], fontsize=8)
    draw_connection(ax, 13.1, 6.22, 13.3, 6.22)

    # Third row
    draw_component_box(ax, 11.1, 5.3, 2.0, 0.45,
                      'ReLU + Dropout',
                      COLORS['gating'], fontsize=8)
    draw_connection(ax, 12.1, 5.52, 11.1, 5.52)

    draw_component_box(ax, 13.3, 5.3, 2.0, 0.45,
                      'Linear(6)',
                      COLORS['gating'], fontsize=8)
    draw_connection(ax, 13.1, 5.52, 13.3, 5.52)

    # Loss function
    draw_component_box(ax, 11.1, 4.6, 4.2, 0.4,
                      'Weighted CrossEntropy Loss',
                      COLORS['special'], fontsize=8, bold=True)
    ax.text(13.2, 4.8,
            'w_i = N / (K × N_i)',
            ha='center', fontsize=7, style='italic')

    # === Legend (bottom) ===
    ax.text(0.5, 4.0, 'Layer Types:',
            fontsize=9, fontweight='bold')

    types = [
        (COLORS['baseline'], 'Baseline Layer'),
        (COLORS['expert_light'], 'Expert Layer'),
        (COLORS['gating'], 'Gating Layer'),
        (COLORS['data'], 'Data / I/O'),
    ]

    for i, (color, label) in enumerate(types):
        y = 3.7 - i * 0.25
        rect = Rectangle((1.5, y - 0.08), 0.20, 0.18,
                        facecolor=color, edgecolor='black', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(1.8, y, label, ha='left', va='center', fontsize=7.5)

    # === Notes (bottom right) ===
    ax.text(6.5, 4.0, 'Architecture Notes:',
            fontsize=9, fontweight='bold')

    notes = [
        '• Baseline & Expert share ResNet1d backbone',
        '• Expert output zero-padded: [p5, p7, 0, 0, 0, 0]',
        '• Gating network learns: α·p_baseline + (1-α)·p_expert',
        '• Weighted CE amplifies minority class gradients',
    ]

    for i, text in enumerate(notes):
        y = 3.7 - i * 0.22
        ax.text(6.5, y, text, ha='left', va='center', fontsize=7)

    plt.tight_layout()
    return fig

# =============================================================================
# Figure 3-3: Complete Framework
# =============================================================================

def create_figure_3_3():
    """Figure 3-3: Complete Framework - Architecture + Training + Results"""
    fig, ax = setup_figure(figsize=(14, 9))

    # Title
    ax.text(7.5, 9.8, 'Figure 3-3: Complete GEE Framework',
            ha='center', va='top', fontsize=14, fontweight='bold')

    # === TOP SECTION: Architecture ===

    # Input
    draw_component_box(ax, 0.5, 6.8, 1.1, 1.0,
                      'Input',
                      COLORS['data'], fontsize=9)

    # Baseline (top)
    draw_module_boundary(ax, 2.0, 7.5, 2.6, 1.5, 'Baseline')
    draw_component_box(ax, 2.2, 8.5, 2.2, 0.5,
                      'ResNet1d',
                      COLORS['baseline'], fontsize=9)
    draw_component_box(ax, 2.2, 7.9, 2.2, 0.4,
                      'FC → Softmax',
                      COLORS['baseline'], fontsize=8)
    draw_data_sequence(ax, 2.3, 7.5, 6, 'p_base')

    # Expert (bottom)
    draw_module_boundary(ax, 2.0, 5.2, 2.6, 1.8, 'Expert')
    draw_component_box(ax, 2.2, 6.5, 2.2, 0.5,
                      'ResNet1d',
                      COLORS['expert_light'], fontsize=9)
    draw_component_box(ax, 2.2, 5.9, 2.2, 0.4,
                      'FC → Softmax',
                      COLORS['expert_light'], fontsize=8)
    draw_data_sequence(ax, 2.3, 5.5, 2, 'p_exp')
    ax.text(3.3, 5.2, '+ zero-pad to ℝ⁶',
            ha='center', fontsize=6.5, style='italic', color='#D32F2F')

    # Gating (right)
    draw_module_boundary(ax, 5.2, 5.8, 2.4, 2.8, 'Gating')
    draw_component_box(ax, 5.4, 8.0, 2.0, 0.4,
                      'Concat',
                      COLORS['gating'], fontsize=8)
    draw_component_box(ax, 5.4, 7.5, 2.0, 0.4,
                      'Linear(128)',
                      COLORS['gating'], fontsize=8)
    draw_component_box(ax, 5.4, 7.0, 2.0, 0.4,
                      'Linear(64)',
                      COLORS['gating'], fontsize=8)
    draw_component_box(ax, 5.4, 6.5, 2.0, 0.4,
                      'Linear(6)',
                      COLORS['gating'], fontsize=8)

    # Output
    draw_component_box(ax, 8.0, 6.7, 1.3, 0.9,
                      'Output',
                      COLORS['data'], fontsize=9)
    draw_data_sequence(ax, 8.1, 6.3, 6, 'y_pred')

    # Arrows
    draw_connection(ax, 1.6, 7.3, 2.2, 8.2)
    draw_connection(ax, 1.6, 7.3, 2.2, 6.2)
    draw_connection(ax, 4.6, 8.0, 5.4, 8.0)
    draw_connection(ax, 4.6, 6.2, 5.4, 7.0)
    draw_connection(ax, 7.6, 7.15, 8.0, 7.15)

    # === MIDDLE SECTION: Training Strategy ===

    ax.text(0.5, 4.8, 'Training Strategy:',
            fontsize=10, fontweight='bold', color=COLORS['border'])

    # Training boxes
    draw_component_box(ax, 0.5, 4.2, 3.0, 0.5,
                      'Baseline: All 6 classes (N=16,287)',
                      COLORS['baseline'], fontsize=8)

    draw_component_box(ax, 3.7, 4.2, 2.8, 0.5,
                      'Expert: Classes 5,7 (N=280)',
                      COLORS['expert_light'], fontsize=8)

    draw_component_box(ax, 6.7, 4.2, 3.2, 0.5,
                      'Gating: Weighted CE Loss',
                      COLORS['gating'], fontsize=8)

    # === BOTTOM SECTION: Class Distribution ===

    ax.text(0.5, 3.3, 'Class Distribution & Weights:',
            fontsize=10, fontweight='bold', color=COLORS['border'])

    # Bar chart
    classes = ['C5', 'C6', 'C7', 'C8', 'C9', 'C10']
    samples = [215, 1034, 65, 4408, 1089, 9476]
    weights = [12.63, 2.62, 41.75, 0.62, 2.49, 0.29]

    for i, (cls, n, w) in enumerate(zip(classes, samples, weights)):
        x = 0.5 + i * 0.95
        bar_height = n / 200.0  # Scale

        is_minority = cls in ['C5', 'C7']
        color = COLORS['expert_light'] if is_minority else COLORS['baseline']

        # Bar
        rect = Rectangle((x, 2.5 - bar_height), 0.65, bar_height,
                        facecolor=color, edgecolor='black', linewidth=0.8)
        ax.add_patch(rect)

        # Labels
        ax.text(x + 0.325, 2.45, cls, ha='center', fontsize=8)
        ax.text(x + 0.325, 2.45 - bar_height - 0.12, f'{n}',
                ha='center', fontsize=7)

        # Weights for minority
        if is_minority:
            ax.text(x + 0.325, 2.75, f'w={w:.2f}',
                    ha='center', fontsize=7,
                    color='#D32F2F', fontweight='bold')

    # === RIGHT SIDE: Performance & Logic ===

    # Performance metrics
    ax.text(10.5, 4.8, 'Performance:',
            fontsize=10, fontweight='bold', color=COLORS['border'])

    metrics = [
        ('Baseline Macro-F1:', '0.63', False),
        ('GEE Macro-F1:', '0.83', False),
        ('Improvement:', '+0.20', True),
    ]

    for i, (label, value, highlight) in enumerate(metrics):
        y = 4.2 - i * 0.3
        ax.text(10.5, y, label, fontsize=8.5, va='center')
        color = '#D32F2F' if highlight else 'black'
        ax.text(14.2, y, value, fontsize=8.5, va='center',
                fontweight='bold', color=color)

    # Decision logic
    ax.text(10.5, 2.9, 'Gating Decision:',
            fontsize=10, fontweight='bold', color=COLORS['border'])

    draw_component_box(ax, 10.5, 2.4, 4.5, 0.35,
                      'Majority class → Trust Baseline',
                      COLORS['baseline'], fontsize=8)

    draw_component_box(ax, 10.5, 1.9, 4.5, 0.35,
                      'Minority class → Trust Expert',
                      COLORS['expert_light'], fontsize=8)

    # Legend
    ax.text(10.5, 1.5, 'Legend:',
            fontsize=9, fontweight='bold')

    legend_items = [
        (COLORS['baseline'], 'Baseline/Majority'),
        (COLORS['expert_light'], 'Expert/Minority'),
        (COLORS['gating'], 'Gating Component'),
        ('#D32F2F', 'Key Metric'),
    ]

    for i, (color, label) in enumerate(legend_items):
        y = 1.2 - i * 0.2
        rect = Rectangle((10.5, y - 0.07), 0.18, 0.15,
                        facecolor=color, edgecolor='black', linewidth=0.6)
        ax.add_patch(rect)
        ax.text(10.75, y, label, ha='left', va='center', fontsize=7)

    plt.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.03)
    return fig

# =============================================================================
# Main
# =============================================================================

def main():
    import os

    output_dir = 'thesis/figures'
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("Generating GEE Architecture Figures v3.0")
    print("Professional Academic Edition")
    print("="*60)

    # Figure 3-1
    print("\n[1/3] Figure 3-1: Architecture Overview...")
    fig1 = create_figure_3_1()
    fig1.savefig(f'{output_dir}/figure_3_1_gee_architecture_overview.png',
                 bbox_inches='tight', dpi=300, facecolor='white')
    fig1.savefig(f'{output_dir}/figure_3_1_gee_architecture_overview.pdf',
                 bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig1)
    print("      ✓ Saved")

    # Figure 3-2
    print("\n[2/3] Figure 3-2: Detailed Architecture...")
    fig2 = create_figure_3_2()
    fig2.savefig(f'{output_dir}/figure_3_2_detailed_component_architecture.png',
                 bbox_inches='tight', dpi=300, facecolor='white')
    fig2.savefig(f'{output_dir}/figure_3_2_detailed_component_architecture.pdf',
                 bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig2)
    print("      ✓ Saved")

    # Figure 3-3
    print("\n[3/3] Figure 3-3: Complete Framework...")
    fig3 = create_figure_3_3()
    fig3.savefig(f'{output_dir}/figure_3_3_complete_gee_framework.png',
                 bbox_inches='tight', dpi=300, facecolor='white')
    fig3.savefig(f'{output_dir}/figure_3_3_complete_gee_framework.pdf',
                 bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig3)
    print("      ✓ Saved")

    print("\n" + "="*60)
    print("All figures generated successfully!")
    print("="*60)
    print(f"\nOutput: {output_dir}/")
    print("\nGenerated files:")
    print("  • figure_3_1_gee_architecture_overview.png/pdf")
    print("  • figure_3_2_detailed_component_architecture.png/pdf")
    print("  • figure_3_3_complete_gee_framework.png/pdf")

if __name__ == '__main__':
    main()
