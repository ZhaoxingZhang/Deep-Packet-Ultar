#!/usr/bin/env python3
"""
Generate GEE Architecture Figures - Version 2

Improved version with clearer layout, proper labels, and better data flow visualization.
Reference style: DeepSeek neural network architecture diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

# Use DejaVu fonts for better character support
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.unicode_minus'] = False

# Color scheme
COLORS = {
    'baseline': '#E8DCCA',      # Tan for baseline
    'expert': '#C8E6C9',        # Light green for expert
    'gating': '#BBDEFB',        # Light blue for gating
    'module_border': '#1976D2', # Blue for module boundaries
    'data': '#EEEEEE',          # Light gray for data
    'arrow': '#333333',         # Dark gray for arrows
    'text_main': '#000000',     # Black for main text
    'text_sub': '#555555',      # Gray for subtext
}

def setup_figure(figsize=(14, 10)):
    """Setup figure with white background"""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(0, figsize[0])
    ax.set_ylim(0, figsize[1])
    ax.axis('off')
    return fig, ax

def draw_module_box(ax, x, y, width, height, title, color_code):
    """Draw a module boundary box"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=8",
                         edgecolor=color_code,
                         facecolor='none',
                         linewidth=2,
                         linestyle='--')
    ax.add_patch(box)

    # Title
    ax.text(x + width/2, y + height - 0.25, title,
            ha='center', va='top', fontsize=11, fontweight='bold',
            color=color_code)

def draw_layer_box(ax, x, y, width, height, text, color,
                   fontsize=9, fontweight='normal'):
    """Draw a layer/operation box"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=4",
                         edgecolor='black',
                         facecolor=color,
                         linewidth=1.2)
    ax.add_patch(box)

    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=fontsize,
            color=COLORS['text_main'], fontweight=fontweight)

def draw_vector_circles(ax, x, y, n_circles, label, color=COLORS['data']):
    """Draw vector as sequence of circles"""
    radius = 0.12
    spacing = 0.35
    total_width = n_circles * spacing

    for i in range(n_circles):
        circle = Circle((x + i*spacing, y), radius,
                       edgecolor='black', facecolor=color, linewidth=0.8)
        ax.add_patch(circle)

    ax.text(x + total_width/2, y - 0.35, label,
            ha='center', va='top', fontsize=8, style='italic')

    return total_width

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    """Draw a connection arrow"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=15,
                           connectionstyle="arc3,rad=0",
                           color=COLORS['arrow'], linewidth=1.8)
    ax.add_patch(arrow)

    if label:
        mid_x, mid_y = (x1 + x2)/2, (y1 + y2)/2
        ax.text(mid_x, mid_y + 0.25, label,
                ha='center', va='bottom', fontsize=7,
                bbox=dict(boxstyle="round,pad=2", facecolor='white',
                         edgecolor='none', alpha=0.8))

def add_info_box(ax, x, y, text, fontsize=7):
    """Add information annotation"""
    ax.text(x, y, text, ha='left', va='top', fontsize=fontsize,
            style='italic', color=COLORS['text_sub'],
            bbox=dict(boxstyle="round,pad=4", facecolor='white',
                     edgecolor='gray', linewidth=0.5, alpha=0.9))

# =============================================================================
# Figure 3-1: Architecture Overview
# =============================================================================

def create_figure_3_1():
    """Figure 3-1: Clean architecture overview"""
    fig, ax = setup_figure()

    # Title
    ax.text(7, 9.7, 'Figure 3-1: GEE Architecture Overview',
            ha='center', va='top', fontsize=15, fontweight='bold')

    # ===== Input =====
    draw_layer_box(ax, 0.5, 4.2, 1.3, 1.2,
                  'Input\nPacket\n(1500 B)',
                  COLORS['data'], fontsize=9)

    # ===== Baseline Model (Top) =====
    draw_module_box(ax, 2.5, 6.5, 3.2, 3.2,
                   'Baseline Model', COLORS['module_border'])
    draw_layer_box(ax, 2.7, 8.8, 2.8, 0.9,
                  'ResNet1d\n(16 Blocks)',
                  COLORS['baseline'], fontsize=10)
    draw_layer_box(ax, 2.7, 7.7, 2.8, 0.7,
                  'FC: 200 -> 100 -> 50 -> 6',
                  COLORS['baseline'], fontsize=9)
    draw_vector_circles(ax, 2.9, 7.1, 6, 'p_baseline', COLORS['data'])
    add_info_box(ax, 2.7, 6.7, 'Trained on all 6 classes')

    # ===== Expert Model (Bottom) =====
    draw_module_box(ax, 2.5, 2.5, 3.2, 3.2,
                   'Minority Expert Model', COLORS['module_border'])
    draw_layer_box(ax, 2.7, 4.8, 2.8, 0.9,
                  'ResNet1d\n(16 Blocks)',
                  COLORS['expert'], fontsize=10)
    draw_layer_box(ax, 2.7, 3.7, 2.8, 0.7,
                  'FC: 200 -> 100 -> 50 -> 2',
                  COLORS['expert'], fontsize=9)
    draw_vector_circles(ax, 2.9, 3.1, 2, 'p_expert', COLORS['data'])
    add_info_box(ax, 2.7, 2.7, 'Trained on classes 5, 7')

    # ===== Gating Network (Right) =====
    draw_module_box(ax, 6.5, 4.0, 3.0, 4.2,
                   'Gating Network', COLORS['module_border'])
    draw_layer_box(ax, 6.7, 7.3, 2.6, 0.6,
                  'Concatenate',
                  COLORS['gating'], fontsize=9)
    draw_layer_box(ax, 6.7, 6.5, 2.6, 0.6,
                  'Linear(128) + ReLU',
                  COLORS['gating'], fontsize=9)
    draw_layer_box(ax, 6.7, 5.7, 2.6, 0.6,
                  'Linear(64) + ReLU',
                  COLORS['gating'], fontsize=9)
    draw_layer_box(ax, 6.7, 4.9, 2.6, 0.6,
                  'Linear(6)',
                  COLORS['gating'], fontsize=9)
    draw_layer_box(ax, 6.7, 4.1, 2.6, 0.5,
                  'Weighted CE Loss',
                  COLORS['gating'], fontsize=8, fontweight='bold')

    # ===== Output =====
    draw_layer_box(ax, 10.2, 5.5, 1.8, 1.2,
                  'Final\nOutput\n6 classes',
                  COLORS['data'], fontsize=9)
    draw_vector_circles(ax, 10.4, 5.1, 6, 'y_pred', COLORS['data'])

    # ===== Arrows =====
    # Input to both models
    draw_arrow(ax, 1.8, 4.8, 2.7, 7.0)
    draw_arrow(ax, 1.8, 4.8, 2.7, 2.9)

    # Models to Gating
    draw_arrow(ax, 5.3, 7.5, 6.7, 7.6)
    draw_arrow(ax, 5.3, 3.5, 6.7, 7.0, label='Extended to R6')

    # Gating to Output
    draw_arrow(ax, 9.5, 6.1, 10.2, 6.1)

    # ===== Legend =====
    ax.text(0.5, 1.2, 'Legend:', fontsize=10, fontweight='bold')

    legend_items = [
        (COLORS['baseline'], 'Baseline Component'),
        (COLORS['expert'], 'Expert Component'),
        (COLORS['gating'], 'Gating Component'),
    ]

    for i, (color, label) in enumerate(legend_items):
        rect = Rectangle((1.5, 1.0 - i*0.35), 0.25, 0.25,
                        facecolor=color, edgecolor='black', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(1.85, 1.12 - i*0.35, label, ha='left', va='center', fontsize=8)

    # ===== Key Points =====
    ax.text(6.5, 1.2, 'Key Innovations:', fontsize=10, fontweight='bold')
    ax.text(6.5, 0.85, u'\u2022 Learnable fusion via gating network', fontsize=8, va='top')
    ax.text(6.5, 0.55, u'\u2022 Weighted cross-entropy for class balance', fontsize=8, va='top')
    ax.text(6.5, 0.25, u'\u2022 Incremental architecture for minority classes', fontsize=8, va='top')

    plt.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02)
    return fig

# =============================================================================
# Figure 3-2: Detailed Architecture
# =============================================================================

def create_figure_3_2():
    """Figure 3-2: Detailed component architecture"""
    fig, ax = setup_figure(figsize=(16, 9))

    # Title
    ax.text(8, 8.5, 'Figure 3-2: Detailed Component Architecture',
            ha='center', va='top', fontsize=15, fontweight='bold')

    # ===== Baseline Model =====
    draw_module_box(ax, 0.5, 5.5, 4.5, 2.8,
                   'Baseline Model', COLORS['module_border'])

    draw_layer_box(ax, 0.7, 7.8, 1.0, 0.5,
                  'Input',
                  COLORS['data'], fontsize=8)
    draw_layer_box(ax, 1.9, 7.8, 1.2, 0.5,
                  'Conv1d',
                  COLORS['baseline'], fontsize=8)
    draw_arrow(ax, 1.7, 8.05, 1.9, 8.05)

    draw_layer_box(ax, 3.3, 7.8, 1.4, 0.5,
                  'BasicBlock x16',
                  COLORS['baseline'], fontsize=9)
    draw_arrow(ax, 3.1, 8.05, 3.3, 8.05)

    draw_layer_box(ax, 0.7, 7.0, 1.3, 0.5,
                  'GlobalPool',
                  COLORS['baseline'], fontsize=8)
    draw_layer_box(ax, 2.2, 7.0, 1.0, 0.5,
                  'FC(200)',
                  COLORS['baseline'], fontsize=8)
    draw_layer_box(ax, 3.4, 7.0, 1.0, 0.5,
                  'FC(100)',
                  COLORS['baseline'], fontsize=8)
    draw_arrow(ax, 2.0, 7.25, 2.2, 7.25)
    draw_arrow(ax, 3.2, 7.25, 3.4, 7.25)

    draw_layer_box(ax, 0.7, 6.2, 1.3, 0.5,
                  'FC(50)',
                  COLORS['baseline'], fontsize=8)
    draw_layer_box(ax, 2.2, 6.2, 1.0, 0.5,
                  'FC(6)',
                  COLORS['baseline'], fontsize=8)
    draw_layer_box(ax, 3.4, 6.2, 1.4, 0.5,
                  'Softmax',
                  COLORS['data'], fontsize=8)
    draw_arrow(ax, 2.0, 6.45, 2.2, 6.45)
    draw_arrow(ax, 3.2, 6.45, 3.4, 6.45)

    add_info_box(ax, 0.7, 5.7, 'Output: p_baseline in R^6')

    # ===== Expert Model =====
    draw_module_box(ax, 5.5, 5.5, 4.5, 2.8,
                   'Expert Model', COLORS['module_border'])

    draw_layer_box(ax, 5.7, 7.8, 1.0, 0.5,
                  'Input',
                  COLORS['data'], fontsize=8)
    draw_layer_box(ax, 6.9, 7.8, 1.2, 0.5,
                  'Conv1d',
                  COLORS['expert'], fontsize=8)
    draw_arrow(ax, 6.7, 8.05, 6.9, 8.05)

    draw_layer_box(ax, 8.3, 7.8, 1.4, 0.5,
                  'BasicBlock x16',
                  COLORS['expert'], fontsize=9)
    draw_arrow(ax, 8.1, 8.05, 8.3, 8.05)

    draw_layer_box(ax, 5.7, 7.0, 1.3, 0.5,
                  'GlobalPool',
                  COLORS['expert'], fontsize=8)
    draw_layer_box(ax, 7.2, 7.0, 1.0, 0.5,
                  'FC(200)',
                  COLORS['expert'], fontsize=8)
    draw_layer_box(ax, 8.4, 7.0, 1.0, 0.5,
                  'FC(100)',
                  COLORS['expert'], fontsize=8)
    draw_arrow(ax, 7.0, 7.25, 7.2, 7.25)
    draw_arrow(ax, 8.2, 7.25, 8.4, 7.25)

    draw_layer_box(ax, 5.7, 6.2, 1.3, 0.5,
                  'FC(50)',
                  COLORS['expert'], fontsize=8)
    draw_layer_box(ax, 7.2, 6.2, 1.0, 0.5,
                  'FC(2)',
                  COLORS['expert'], fontsize=8)
    draw_layer_box(ax, 8.4, 6.2, 1.4, 0.5,
                  'Softmax',
                  COLORS['data'], fontsize=8)
    draw_arrow(ax, 7.0, 6.45, 7.2, 6.45)
    draw_arrow(ax, 8.2, 6.45, 8.4, 6.45)

    add_info_box(ax, 5.7, 5.7, 'Output: p_expert in R^2 (minority classes only)')

    # ===== Gating Network =====
    draw_module_box(ax, 10.5, 5.5, 5.0, 2.8,
                   'Gating Network', COLORS['module_border'])

    draw_layer_box(ax, 10.7, 7.8, 2.2, 0.5,
                  'Concat(12)',
                  COLORS['gating'], fontsize=8)
    draw_layer_box(ax, 13.1, 7.8, 2.2, 0.5,
                  'Linear(128)',
                  COLORS['gating'], fontsize=8)
    draw_arrow(ax, 12.9, 8.05, 13.1, 8.05)

    draw_layer_box(ax, 10.7, 7.0, 2.2, 0.5,
                  'ReLU + Dropout',
                  COLORS['gating'], fontsize=8)
    draw_layer_box(ax, 13.1, 7.0, 2.2, 0.5,
                  'Linear(64)',
                  COLORS['gating'], fontsize=8)
    draw_arrow(ax, 11.8, 7.25, 10.7, 7.25)
    draw_arrow(ax, 12.9, 7.25, 13.1, 7.25)

    draw_layer_box(ax, 10.7, 6.2, 2.2, 0.5,
                  'ReLU + Dropout',
                  COLORS['gating'], fontsize=8)
    draw_layer_box(ax, 13.1, 6.2, 2.2, 0.5,
                  'Linear(6)',
                  COLORS['gating'], fontsize=8)
    draw_arrow(ax, 11.8, 6.45, 10.7, 6.45)
    draw_arrow(ax, 12.9, 6.45, 13.1, 6.45)

    draw_layer_box(ax, 10.7, 5.6, 4.6, 0.5,
                  'Weighted CrossEntropy Loss',
                  COLORS['gating'], fontsize=8, fontweight='bold')

    add_info_box(ax, 10.7, 5.0, 'Parameters: ~2,500 | Trained with class weights')

    # ===== Legend =====
    ax.text(0.5, 4.2, 'Layer Types:', fontsize=10, fontweight='bold')

    types = [
        (COLORS['baseline'], 'Baseline Layer'),
        (COLORS['expert'], 'Expert Layer'),
        (COLORS['gating'], 'Gating Layer'),
        (COLORS['data'], 'Data / I/O'),
    ]

    for i, (color, label) in enumerate(types):
        rect = Rectangle((1.5, 4.3 - i*0.3), 0.25, 0.2,
                        facecolor=color, edgecolor='black', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(1.85, 4.4 - i*0.3, label, ha='left', va='center', fontsize=8)

    # ===== Architecture Notes =====
    ax.text(6.5, 4.2, 'Architecture Details:', fontsize=10, fontweight='bold')
    ax.text(6.5, 3.9, u'\u2022 Baseline & Expert share ResNet1d backbone',
            fontsize=8, va='top')
    ax.text(6.5, 3.6, u'\u2022 Expert output dimension reduced (2 classes)',
            fontsize=8, va='top')
    ax.text(6.5, 3.3, u'\u2022 Expert output zero-padded to match baseline (6 dims)',
            fontsize=8, va='top')
    ax.text(6.5, 3.0, u'\u2022 Gating network learns optimal fusion strategy',
            fontsize=8, va='top')

    plt.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
    return fig

# =============================================================================
# Figure 3-3: Complete Framework
# =============================================================================

def create_figure_3_3():
    """Figure 3-3: Complete framework with training details"""
    fig, ax = setup_figure(figsize=(16, 11))

    # Title
    ax.text(8, 10.7, 'Figure 3-3: Complete GEE Framework - Architecture and Training Strategy',
            ha='center', va='top', fontsize=14, fontweight='bold')

    # ===== Top: Architecture =====

    # Input
    draw_layer_box(ax, 0.5, 8.5, 1.2, 1.0,
                  'Input',
                  COLORS['data'], fontsize=9)

    # Baseline
    draw_module_box(ax, 2.2, 9.0, 2.8, 1.8,
                   'Baseline', COLORS['module_border'])
    draw_layer_box(ax, 2.4, 10.2, 2.4, 0.5,
                  'ResNet1d',
                  COLORS['baseline'], fontsize=9)
    draw_layer_box(ax, 2.4, 9.5, 2.4, 0.4,
                  'FC -> Softmax',
                  COLORS['baseline'], fontsize=8)
    draw_vector_circles(ax, 2.5, 9.1, 6, 'p_base', COLORS['data'])

    # Expert
    draw_module_box(ax, 2.2, 6.5, 2.8, 1.8,
                   'Expert', COLORS['module_border'])
    draw_layer_box(ax, 2.4, 7.7, 2.4, 0.5,
                  'ResNet1d',
                  COLORS['expert'], fontsize=9)
    draw_layer_box(ax, 2.4, 7.0, 2.4, 0.4,
                  'FC -> Softmax',
                  COLORS['expert'], fontsize=8)
    draw_vector_circles(ax, 2.5, 6.6, 2, 'p_exp', COLORS['data'])

    # Gating
    draw_module_box(ax, 5.5, 7.5, 2.5, 3.0,
                   'Gating', COLORS['module_border'])
    draw_layer_box(ax, 5.7, 10.0, 2.1, 0.45,
                  'Concat',
                  COLORS['gating'], fontsize=8)
    draw_layer_box(ax, 5.7, 9.4, 2.1, 0.45,
                  'Linear(128)',
                  COLORS['gating'], fontsize=8)
    draw_layer_box(ax, 5.7, 8.8, 2.1, 0.45,
                  'Linear(64)',
                  COLORS['gating'], fontsize=8)
    draw_layer_box(ax, 5.7, 8.2, 2.1, 0.45,
                  'Linear(6)',
                  COLORS['gating'], fontsize=8)

    # Output
    draw_layer_box(ax, 8.5, 8.7, 1.5, 1.0,
                  'Output',
                  COLORS['data'], fontsize=9)
    draw_vector_circles(ax, 8.6, 8.3, 6, 'y', COLORS['data'])

    # Arrows
    draw_arrow(ax, 1.7, 9.0, 2.4, 10.0)
    draw_arrow(ax, 1.7, 9.0, 2.4, 7.5)
    draw_arrow(ax, 5.0, 9.8, 5.7, 10.0)
    draw_arrow(ax, 5.0, 7.3, 5.7, 8.5)
    draw_arrow(ax, 8.0, 9.0, 8.5, 9.2)

    # ===== Middle: Training Strategy =====

    ax.text(0.5, 5.5, 'Training Strategy:', fontsize=11, fontweight='bold',
            color=COLORS['module_border'])

    # Training data boxes
    draw_layer_box(ax, 0.5, 4.8, 3.0, 0.6,
                  'Baseline: All 6 classes (N=16,287)',
                  COLORS['baseline'], fontsize=8)
    draw_layer_box(ax, 3.8, 4.8, 3.0, 0.6,
                  'Expert: Classes 5,7 (N=280)',
                  COLORS['expert'], fontsize=8)
    draw_layer_box(ax, 7.1, 4.8, 3.2, 0.6,
                  'Gating: All 6 classes + Weighted CE',
                  COLORS['gating'], fontsize=8)

    # ===== Bottom: Class Distribution & Weights =====

    ax.text(0.5, 4.0, 'Class Distribution & Loss Weights:',
            fontsize=11, fontweight='bold', color=COLORS['module_border'])

    # Bar chart
    classes = ['C5', 'C6', 'C7', 'C8', 'C9', 'C10']
    samples = [215, 1034, 65, 4408, 1089, 9476]
    weights = [12.63, 2.62, 41.75, 0.62, 2.49, 0.29]

    for i, (cls, n, w) in enumerate(zip(classes, samples, weights)):
        x_pos = 0.5 + i * 1.0
        bar_height = n / 150.0  # Scale for visibility

        is_minority = cls in ['C5', 'C7']
        color = COLORS['expert'] if is_minority else COLORS['baseline']

        # Bar
        rect = Rectangle((x_pos, 3.1 - bar_height), 0.6, bar_height,
                        facecolor=color, edgecolor='black', linewidth=0.8)
        ax.add_patch(rect)

        # Labels
        ax.text(x_pos + 0.3, 3.0, cls, ha='center', fontsize=8)
        ax.text(x_pos + 0.3, 3.0 - bar_height - 0.15, f'{n}',
                ha='center', fontsize=7)

        # Weight annotation for minority classes
        if is_minority:
            ax.text(x_pos + 0.3, 3.35, f'w={w:.2f}',
                    ha='center', fontsize=7,
                    color='#D32F2F', fontweight='bold')

    # ===== Decision Logic =====

    ax.text(11, 5.5, 'Gating Decision Logic:',
            fontsize=11, fontweight='bold', color=COLORS['module_border'])

    draw_layer_box(ax, 11, 4.9, 4.5, 0.4,
                  'Majority class sample -> Trust Baseline',
                  COLORS['baseline'], fontsize=8)
    draw_layer_box(ax, 11, 4.4, 4.5, 0.4,
                  'Minority class sample -> Trust Expert',
                  COLORS['expert'], fontsize=8)

    # ===== Performance =====

    ax.text(11, 3.8, 'Performance Improvement:',
            fontsize=11, fontweight='bold', color=COLORS['module_border'])

    metrics = [
        ('Baseline Macro-F1:', '0.63'),
        ('GEE Macro-F1:', '0.83'),
        ('Improvement:', '+0.20'),
    ]

    for i, (label, value) in enumerate(metrics):
        y_pos = 3.3 - i * 0.35
        ax.text(11, y_pos, label, fontsize=9, va='center')
        ax.text(14.8, y_pos, value, fontsize=9, va='center',
                fontweight='bold', color='#D32F2F' if i == 2 else 'black')

    # ===== Legend =====

    ax.text(11, 1.8, 'Legend:', fontsize=10, fontweight='bold')

    legend_items = [
        (COLORS['baseline'], 'Majority Class Component'),
        (COLORS['expert'], 'Minority Class Component'),
        (COLORS['gating'], 'Gating Network'),
        ('#D32F2F', 'Key Metric / Minority'),
    ]

    for i, (color, label) in enumerate(legend_items):
        y_pos = 1.5 - i * 0.3
        rect = Rectangle((11, y_pos - 0.1), 0.22, 0.2,
                        facecolor=color, edgecolor='black', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(11.3, y_pos, label, ha='left', va='center', fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02)
    return fig

# =============================================================================
# Main
# =============================================================================

def main():
    """Generate all figures"""
    import os

    output_dir = 'thesis/figures'
    os.makedirs(output_dir, exist_ok=True)

    print("Generating GEE Architecture Figures v2.0...")

    # Figure 3-1
    print("\n[1/3] Figure 3-1: Architecture Overview...")
    fig1 = create_figure_3_1()
    fig1.savefig(f'{output_dir}/figure_3_1_gee_architecture_overview.png',
                 bbox_inches='tight', dpi=300)
    fig1.savefig(f'{output_dir}/figure_3_1_gee_architecture_overview.pdf',
                 bbox_inches='tight', dpi=300)
    plt.close(fig1)
    print("      Saved: figure_3_1_gee_architecture_overview.png/pdf")

    # Figure 3-2
    print("\n[2/3] Figure 3-2: Detailed Architecture...")
    fig2 = create_figure_3_2()
    fig2.savefig(f'{output_dir}/figure_3_2_detailed_component_architecture.png',
                 bbox_inches='tight', dpi=300)
    fig2.savefig(f'{output_dir}/figure_3_2_detailed_component_architecture.pdf',
                 bbox_inches='tight', dpi=300)
    plt.close(fig2)
    print("      Saved: figure_3_2_detailed_component_architecture.png/pdf")

    # Figure 3-3
    print("\n[3/3] Figure 3-3: Complete Framework...")
    fig3 = create_figure_3_3()
    fig3.savefig(f'{output_dir}/figure_3_3_complete_gee_framework.png',
                 bbox_inches='tight', dpi=300)
    fig3.savefig(f'{output_dir}/figure_3_3_complete_gee_framework.pdf',
                 bbox_inches='tight', dpi=300)
    plt.close(fig3)
    print("      Saved: figure_3_3_complete_gee_framework.png/pdf")

    print("\n" + "="*60)
    print("All figures generated successfully!")
    print("="*60)

if __name__ == '__main__':
    main()
