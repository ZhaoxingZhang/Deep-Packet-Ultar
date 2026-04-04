#!/usr/bin/env python3
"""
Generate GEE Architecture Figures

This script creates three academic-level architecture diagrams for the GEE network:
1. Figure 3-1: GEE Architecture Overview
2. Figure 3-2: Detailed Component Architecture
3. Figure 3-3: Complete GEE Framework

Reference style: DeepSeek neural network architecture diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

# Set style for academic publication
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Color scheme (inspired by DeepSeek architecture diagram)
COLORS = {
    'baseline': '#E8DCCA',      # Tan/beige for baseline model
    'expert': '#C8E6C9',        # Light green for expert model
    'gating': '#BBDEFB',        # Light blue for gating network
    'module_border': '#1976D2', # Blue for module boundaries
    'data': '#F5F5F5',          # Light gray for data representations
    'arrow': '#000000',         # Black for arrows
    'text': '#000000',          # Black for text
    'highlight': '#FF9800',     # Orange for highlights
}

def setup_figure(figsize=(13, 11)):
    """Setup figure with white background"""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(0, figsize[0])
    ax.set_ylim(0, figsize[1])
    ax.axis('off')
    return fig, ax

def draw_module_box(ax, x, y, width, height, title, color, linestyle='--'):
    """Draw a module box with dashed border"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=10",
                         edgecolor=COLORS['module_border'],
                         facecolor='none',
                         linewidth=2,
                         linestyle=linestyle)
    ax.add_patch(box)

    # Add title
    ax.text(x + width/2, y + height - 0.3, title,
            ha='center', va='top', fontsize=12, fontweight='bold',
            color=COLORS['module_border'])
    return y + height - 0.5

def draw_component_box(ax, x, y, width, height, text, color, fontsize=9, fontweight='normal'):
    """Draw a component box with rounded corners"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=5",
                         edgecolor='black',
                         facecolor=color,
                         linewidth=1.5)
    ax.add_patch(box)

    # Add text
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=fontsize,
            color=COLORS['text'], fontweight=fontweight)
    return y + height

def draw_vector_sequence(ax, x, y, num_circles, label, color=COLORS['data']):
    """Draw a sequence of circles representing a vector"""
    radius = 0.15
    spacing = 0.4
    total_width = num_circles * spacing

    for i in range(num_circles):
        circle = Circle((x + i * spacing, y), radius,
                       edgecolor='black',
                       facecolor=color,
                       linewidth=1)
        ax.add_patch(circle)

    # Add label
    ax.text(x + total_width/2, y - 0.4, label,
            ha='center', va='top', fontsize=8, style='italic')

    return total_width

def draw_arrow(ax, x1, y1, x2, y2, text='', style='-|>', curvature=0):
    """Draw an arrow with optional text"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style,
                           connectionstyle=f"arc3,rad={curvature}",
                           color=COLORS['arrow'],
                           linewidth=2,
                           shrinkA=0, shrinkB=0)
    ax.add_patch(arrow)

    if text:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.3, text,
                ha='center', va='bottom', fontsize=8)

def add_dimension_label(ax, x, y, text):
    """Add dimension label"""
    ax.text(x, y, text, ha='left', va='center',
            fontsize=7, bbox=dict(boxstyle="round,pad=3",
                                   facecolor='white',
                                   edgecolor='gray',
                                   linewidth=0.5))

# =============================================================================
# Figure 3-1: GEE Architecture Overview
# =============================================================================

def create_figure_3_1():
    """Create Figure 3-1: GEE Architecture Overview"""
    fig, ax = setup_figure()

    # Title
    ax.text(6.5, 10.5, 'Figure 3-1: GEE Architecture Overview',
            ha='center', va='top', fontsize=14, fontweight='bold')

    # ===== Input Layer =====
    input_x = 0.5
    input_y = 4
    draw_component_box(ax, input_x, input_y, 1.5, 1.5,
                      'Input\nPacket\n(1500 bytes)',
                      COLORS['data'], fontsize=9)

    # ===== Baseline Model Module =====
    baseline_x = 2.5
    baseline_y = 7
    draw_module_box(ax, baseline_x, baseline_y, 2.5, 3.5,
                   'Baseline Model', COLORS['module_border'])
    draw_component_box(ax, baseline_x + 0.3, baseline_y - 0.5, 1.9, 1,
                      'ResNet1d\n(16 Blocks)',
                      COLORS['baseline'])
    draw_component_box(ax, baseline_x + 0.3, baseline_y - 1.7, 1.9, 0.8,
                      'FC Layers\n(200→100→50→6)',
                      COLORS['baseline'])

    # Baseline output vector
    draw_vector_sequence(ax, baseline_x + 0.5, baseline_y - 2.8, 6,
                        'p_baseline ∈ ℝ⁶', COLORS['data'])

    # Training annotation
    ax.text(baseline_x + 1.25, baseline_y - 3.3,
            'Trained on ALL 6 classes',
            ha='center', fontsize=7, style='italic',
            color='#666666')

    # ===== Expert Model Module =====
    expert_x = 2.5
    expert_y = 2.5
    draw_module_box(ax, expert_x, expert_y, 2.5, 3.5,
                   'Minority Expert Model', COLORS['module_border'])
    draw_component_box(ax, expert_x + 0.3, expert_y - 0.5, 1.9, 1,
                      'ResNet1d\n(16 Blocks)',
                      COLORS['expert'])
    draw_component_box(ax, expert_x + 0.3, expert_y - 1.7, 1.9, 0.8,
                      'FC Layers\n(200→100→50→2)',
                      COLORS['expert'])

    # Expert output vector
    draw_vector_sequence(ax, expert_x + 0.5, expert_y - 2.8, 2,
                        'p_expert ∈ ℝ²', COLORS['data'])

    # Training annotation
    ax.text(expert_x + 1.25, expert_y - 3.3,
            'Trained on MINORITY classes\n(Class 5, 7 only)',
            ha='center', fontsize=7, style='italic',
            color='#666666')

    # ===== Gating Network Module =====
    gating_x = 6
    gating_y = 4.5
    draw_module_box(ax, gating_x, gating_y, 3, 3,
                   'Gating Network', COLORS['module_border'])
    draw_component_box(ax, gating_x + 0.3, gating_y - 0.3, 2.4, 0.7,
                      'Concat\n(12 dims)',
                      COLORS['gating'], fontsize=8)
    draw_component_box(ax, gating_x + 0.3, gating_y - 1.2, 2.4, 0.7,
                      'Linear(128) → ReLU',
                      COLORS['gating'], fontsize=8)
    draw_component_box(ax, gating_x + 0.3, gating_y - 2.1, 2.4, 0.7,
                      'Linear(64) → ReLU',
                      COLORS['gating'], fontsize=8)

    # Training annotation
    ax.text(gating_x + 1.5, gating_y - 2.7,
            'Weighted CE Loss',
            ha='center', fontsize=8, fontweight='bold',
            color=COLORS['highlight'])

    # ===== Output Layer =====
    output_x = 10
    output_y = 5.5
    draw_component_box(ax, output_x, output_y, 2, 1.5,
                      'Final Output\n6-Class Prediction',
                      COLORS['data'], fontsize=9)
    draw_vector_sequence(ax, output_x + 0.2, output_y - 0.3, 6,
                        'y ∈ ℝ⁶', COLORS['data'])

    # ===== Arrows: Input to Models =====
    draw_arrow(ax, 2.0, input_y + 0.75, baseline_x + 0.3, baseline_y - 0.5)
    draw_arrow(ax, 2.0, input_y + 0.75, expert_x + 0.3, expert_y - 0.5)

    # ===== Arrows: Models to Gating =====
    # Baseline to gating
    draw_arrow(ax, baseline_x + 2.5, baseline_y - 2.5,
              gating_x + 0.3, gating_y - 1.5, curvature=0.2)

    # Expert to gating (with extension note)
    draw_arrow(ax, expert_x + 2.5, expert_y - 2.5,
              gating_x + 0.3, gating_y - 0.5, curvature=-0.2)
    ax.text(5.2, 3.8, 'Extended to ℝ⁶\n(zero-padding)',
            ha='center', fontsize=6, style='italic', color='#666666')

    # ===== Arrow: Gating to Output =====
    draw_arrow(ax, gating_x + 2.7, gating_y - 1.2, output_x, output_y + 0.75)

    # ===== Legend =====
    legend_y = 0.3
    ax.text(0.5, legend_y + 0.8, 'Legend:', fontsize=10, fontweight='bold')

    # Legend items
    legend_items = [
        (COLORS['baseline'], 'Baseline Component'),
        (COLORS['expert'], 'Expert Component'),
        (COLORS['gating'], 'Gating Component'),
    ]

    for i, (color, label) in enumerate(legend_items):
        rect = Rectangle((1.5 + i * 2.5, legend_y), 0.3, 0.3,
                        facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(1.9 + i * 2.5, legend_y + 0.15, label,
                ha='left', va='center', fontsize=8)

    # ===== Key Innovation =====
    ax.text(10, 1, 'Key Innovations:',
            fontsize=9, fontweight='bold')
    ax.text(10, 0.6, '✓ Learnable fusion strategy',
            fontsize=8, va='center')
    ax.text(10, 0.3, '✓ Weighted CE loss',
            fontsize=8, va='center')

    plt.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)

    return fig

# =============================================================================
# Figure 3-2: Detailed Component Architecture
# =============================================================================

def create_figure_3_2():
    """Create Figure 3-2: Detailed Component Architecture"""
    fig, ax = setup_figure(figsize=(16, 10))

    # Title
    ax.text(8, 9.8, 'Figure 3-2: Detailed Component Architecture',
            ha='center', va='top', fontsize=14, fontweight='bold')

    # ===== Baseline Model (Detailed) =====
    baseline_x = 0.5
    baseline_y = 7.5
    draw_module_box(ax, baseline_x, baseline_y, 4.5, 6.5,
                   'Baseline Model: ResNet1d Architecture',
                   COLORS['module_border'])

    # Input
    draw_component_box(ax, baseline_x + 0.3, baseline_y - 0.5, 1.5, 0.6,
                      'Input (1×1500)',
                      COLORS['data'], fontsize=8)

    # Conv1
    draw_component_box(ax, baseline_x + 2.3, baseline_y - 0.5, 1.5, 0.6,
                      'Conv1d(64)',
                      COLORS['baseline'], fontsize=8)
    draw_arrow(ax, baseline_x + 1.8, baseline_y - 0.2,
              baseline_x + 2.3, baseline_y - 0.2)

    # BN + ReLU
    draw_component_box(ax, baseline_x + 2.3, baseline_y - 1.2, 1.5, 0.5,
                      'BN + ReLU',
                      COLORS['baseline'], fontsize=8)
    draw_arrow(ax, baseline_x + 3.05, baseline_y - 0.5,
              baseline_x + 3.05, baseline_y - 1.2)

    # BasicBlock notation
    draw_component_box(ax, baseline_x + 1.3, baseline_y - 2.2, 2.5, 1.2,
                      'BasicBlock ×16',
                      COLORS['baseline'], fontsize=9)
    ax.text(baseline_x + 2.55, baseline_y - 1.9,
            '[Conv1d→BN→ReLU→Conv1d→BN→ReLU\n + Skip Connection]',
            ha='center', fontsize=6, style='italic')
    draw_arrow(ax, baseline_x + 3.05, baseline_y - 1.7,
              baseline_x + 2.55, baseline_y - 2.2)

    # Global Pooling
    draw_component_box(ax, baseline_x + 2.3, baseline_y - 3.8, 1.5, 0.5,
                      'Global AvgPool',
                      COLORS['baseline'], fontsize=8)
    draw_arrow(ax, baseline_x + 2.55, baseline_y - 3.4,
              baseline_x + 2.55, baseline_y - 3.8)

    # FC layers
    draw_component_box(ax, baseline_x + 1.3, baseline_y - 4.6, 1.2, 0.5,
                      'FC(200)',
                      COLORS['baseline'], fontsize=8)
    draw_component_box(ax, baseline_x + 2.8, baseline_y - 4.6, 1.2, 0.5,
                      'FC(100)',
                      COLORS['baseline'], fontsize=8)
    draw_component_box(ax, baseline_x + 1.3, baseline_y - 5.4, 1.2, 0.5,
                      'FC(50)',
                      COLORS['baseline'], fontsize=8)
    draw_component_box(ax, baseline_x + 2.8, baseline_y - 5.4, 1.2, 0.5,
                      'FC(6)',
                      COLORS['baseline'], fontsize=8)

    # Connect FC layers
    draw_arrow(ax, baseline_x + 2.55, baseline_y - 4.3,
              baseline_x + 2.55, baseline_y - 4.6)
    draw_arrow(ax, baseline_x + 3.05, baseline_y - 4.35,
              baseline_x + 3.4, baseline_y - 4.35, curvature=0.1)
    draw_arrow(ax, baseline_x + 2.55, baseline_y - 5.1,
              baseline_x + 2.55, baseline_y - 5.4)
    draw_arrow(ax, baseline_x + 1.9, baseline_y - 5.15,
              baseline_x + 2.25, baseline_y - 5.15, curvature=0.1)

    # Softmax output
    draw_component_box(ax, baseline_x + 1.3, baseline_y - 6.2, 2.7, 0.5,
                      'Softmax → p_baseline ∈ ℝ⁶',
                      COLORS['data'], fontsize=8)
    draw_arrow(ax, baseline_x + 2.55, baseline_y - 5.9,
              baseline_x + 2.55, baseline_y - 6.2)

    # ===== Expert Model (Detailed) =====
    expert_x = 5.8
    expert_y = 7.5
    draw_module_box(ax, expert_x, expert_y, 4.5, 6.5,
                   'Minority Expert: Same Architecture',
                   COLORS['module_border'])

    # Same structure as baseline, but output is 2D
    draw_component_box(ax, expert_x + 0.3, expert_y - 0.5, 1.5, 0.6,
                      'Input (1×1500)',
                      COLORS['data'], fontsize=8)
    draw_component_box(ax, expert_x + 2.3, expert_y - 0.5, 1.5, 0.6,
                      'Conv1d(64)',
                      COLORS['expert'], fontsize=8)
    draw_arrow(ax, expert_x + 1.8, expert_y - 0.2,
              expert_x + 2.3, expert_y - 0.2)

    draw_component_box(ax, expert_x + 2.3, expert_y - 1.2, 1.5, 0.5,
                      'BN + ReLU',
                      COLORS['expert'], fontsize=8)
    draw_arrow(ax, expert_x + 3.05, expert_y - 0.5,
              expert_x + 3.05, expert_y - 1.2)

    draw_component_box(ax, expert_x + 1.3, expert_y - 2.2, 2.5, 1.2,
                      'BasicBlock ×16',
                      COLORS['expert'], fontsize=9)
    draw_arrow(ax, expert_x + 3.05, expert_y - 1.7,
              expert_x + 2.55, expert_y - 2.2)

    draw_component_box(ax, expert_x + 2.3, expert_y - 3.8, 1.5, 0.5,
                      'Global AvgPool',
                      COLORS['expert'], fontsize=8)
    draw_arrow(ax, expert_x + 2.55, expert_y - 3.4,
              expert_x + 2.55, expert_y - 3.8)

    draw_component_box(ax, expert_x + 1.3, expert_y - 4.6, 1.2, 0.5,
                      'FC(200)',
                      COLORS['expert'], fontsize=8)
    draw_component_box(ax, expert_x + 2.8, expert_y - 4.6, 1.2, 0.5,
                      'FC(100)',
                      COLORS['expert'], fontsize=8)
    draw_component_box(ax, expert_x + 1.3, expert_y - 5.4, 1.2, 0.5,
                      'FC(50)',
                      COLORS['expert'], fontsize=8)
    draw_component_box(ax, expert_x + 2.8, expert_y - 5.4, 1.2, 0.5,
                      'FC(2)',
                      COLORS['expert'], fontsize=8)

    # Connect FC layers
    draw_arrow(ax, expert_x + 2.55, expert_y - 4.3,
              expert_x + 2.55, expert_y - 4.6)
    draw_arrow(ax, expert_x + 3.05, expert_y - 4.35,
              expert_x + 3.4, expert_y - 4.35, curvature=0.1)
    draw_arrow(ax, expert_x + 2.55, expert_y - 5.1,
              expert_x + 2.55, expert_y - 5.4)
    draw_arrow(ax, expert_x + 1.9, expert_y - 5.15,
              expert_x + 2.25, expert_y - 5.15, curvature=0.1)

    draw_component_box(ax, expert_x + 1.3, expert_y - 6.2, 2.7, 0.5,
                      'Softmax → p_expert ∈ ℝ²',
                      COLORS['data'], fontsize=8)
    draw_arrow(ax, expert_x + 2.55, expert_y - 5.9,
              expert_x + 2.55, expert_y - 6.2)

    # ===== Gating Network (Detailed) =====
    gating_x = 11.5
    gating_y = 7.5
    draw_module_box(ax, gating_x, gating_y, 4, 6.5,
                   'Gating Network: MLP Architecture',
                   COLORS['module_border'])

    # Input
    draw_component_box(ax, gating_x + 0.3, gating_y - 0.5, 3.4, 0.6,
                      'Concat(p_baseline, p_expert)\nInput: 12 dims',
                      COLORS['gating'], fontsize=8)

    # Hidden layers
    draw_component_box(ax, gating_x + 0.5, gating_y - 1.5, 3, 0.7,
                      'Linear(128) + ReLU + Dropout(0.5)',
                      COLORS['gating'], fontsize=8)
    draw_arrow(ax, gating_x + 2, gating_y - 1.1,
              gating_x + 2, gating_y - 1.5)

    draw_component_box(ax, gating_x + 0.5, gating_y - 2.6, 3, 0.7,
                      'Linear(64) + ReLU + Dropout(0.5)',
                      COLORS['gating'], fontsize=8)
    draw_arrow(ax, gating_x + 2, gating_y - 2.2,
              gating_x + 2, gating_y - 2.6)

    # Output layer
    draw_component_box(ax, gating_x + 0.5, gating_y - 3.7, 3, 0.7,
                      'Linear(6) → Logits',
                      COLORS['gating'], fontsize=8)
    draw_arrow(ax, gating_x + 2, gating_y - 3.3,
              gating_x + 2, gating_y - 3.7)

    # CE Loss
    draw_component_box(ax, gating_x + 0.5, gating_y - 4.8, 3, 0.7,
                      'Weighted CrossEntropy Loss',
                      COLORS['highlight'], fontsize=8, fontweight='bold')
    draw_arrow(ax, gating_x + 2, gating_y - 4.4,
              gating_x + 2, gating_y - 4.8)

    # Weight formula
    ax.text(gating_x + 2, gating_y - 5.8,
            'w_i = N_samples / (N_classes × N_samples_i)',
            ha='center', fontsize=7, style='italic')

    # Parameter count
    ax.text(gating_x + 2, gating_y - 6.3,
            '≈2,500 parameters',
            ha='center', fontsize=7, color='#666666')

    # ===== Legend at bottom =====
    legend_y = 0.3
    ax.text(0.5, legend_y + 0.8, 'Layer Types:',
            fontsize=10, fontweight='bold')

    layer_types = [
        (COLORS['baseline'], 'Conv / Pool / FC (Baseline)'),
        (COLORS['expert'], 'Conv / Pool / FC (Expert)'),
        (COLORS['gating'], 'Linear / Activation (Gating)'),
        (COLORS['data'], 'Data / Output'),
    ]

    for i, (color, label) in enumerate(layer_types):
        y_pos = legend_y + 0.5 - i * 0.3
        rect = Rectangle((4, y_pos - 0.1), 0.3, 0.25,
                        facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(4.4, y_pos, label, ha='left', va='center', fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)

    return fig

# =============================================================================
# Figure 3-3: Complete GEE Framework
# =============================================================================

def create_figure_3_3():
    """Create Figure 3-3: Complete GEE Framework with Training Details"""
    fig, ax = setup_figure(figsize=(16, 12))

    # Title
    ax.text(8, 11.8, 'Figure 3-3: Complete GEE Framework - Training and Inference',
            ha='center', va='top', fontsize=14, fontweight='bold')

    # ===== Top: Architecture Overview =====

    # Input
    draw_component_box(ax, 0.5, 9.5, 1.5, 1,
                      'Input\nTraffic\nPacket',
                      COLORS['data'], fontsize=9)

    # Baseline Module
    baseline_x = 2.5
    baseline_y = 9.8
    draw_module_box(ax, baseline_x, baseline_y, 3, 2.5,
                   'Baseline Model\nResNet1d', COLORS['module_border'])
    draw_component_box(ax, baseline_x + 0.3, baseline_y - 0.5, 2.4, 0.6,
                      'Conv Blocks (16)',
                      COLORS['baseline'])
    draw_component_box(ax, baseline_x + 0.3, baseline_y - 1.3, 2.4, 0.6,
                      'FC Layers',
                      COLORS['baseline'])
    draw_vector_sequence(ax, baseline_x + 0.5, baseline_y - 2.1, 6,
                        'p_baseline', COLORS['data'])
    ax.text(baseline_x + 1.5, baseline_y - 2.4,
            '6 classes', ha='center', fontsize=7)

    # Expert Module
    expert_x = 2.5
    expert_y = 6.8
    draw_module_box(ax, expert_x, expert_y, 3, 2.5,
                   'Expert Model\nResNet1d', COLORS['module_border'])
    draw_component_box(ax, expert_x + 0.3, expert_y - 0.5, 2.4, 0.6,
                      'Conv Blocks (16)',
                      COLORS['expert'])
    draw_component_box(ax, expert_x + 0.3, expert_y - 1.3, 2.4, 0.6,
                      'FC Layers',
                      COLORS['expert'])
    draw_vector_sequence(ax, expert_x + 0.5, expert_y - 2.1, 2,
                        'p_expert', COLORS['data'])
    ax.text(expert_x + 1.5, expert_y - 2.4,
            '2 classes', ha='center', fontsize=7)

    # Gating Module
    gating_x = 6.5
    gating_y = 8
    draw_module_box(ax, gating_x, gating_y, 3, 3.5,
                   'Gating Network\nMLP', COLORS['module_border'])
    draw_component_box(ax, gating_x + 0.3, gating_y - 0.4, 2.4, 0.5,
                      'Concat(12)',
                      COLORS['gating'], fontsize=8)
    draw_component_box(ax, gating_x + 0.3, gating_y - 1.1, 2.4, 0.5,
                      'Linear(128)',
                      COLORS['gating'], fontsize=8)
    draw_component_box(ax, gating_x + 0.3, gating_y - 1.8, 2.4, 0.5,
                      'Linear(64)',
                      COLORS['gating'], fontsize=8)
    draw_component_box(ax, gating_x + 0.3, gating_y - 2.5, 2.4, 0.5,
                      'Linear(6)',
                      COLORS['gating'], fontsize=8)

    # Output
    draw_component_box(ax, 10.5, 8.5, 2, 1.5,
                      'Final\nPrediction',
                      COLORS['data'], fontsize=9)

    # Arrows
    draw_arrow(ax, 2.0, 10, baseline_x + 0.3, baseline_y - 0.2)
    draw_arrow(ax, 2.0, 10, expert_x + 0.3, expert_y - 0.2, curvature=-0.3)
    draw_arrow(ax, baseline_x + 2.8, baseline_y - 1.8,
              gating_x + 0.3, gating_y - 0.9, curvature=0.2)
    draw_arrow(ax, expert_x + 2.8, expert_y - 1.8,
              gating_x + 0.3, gating_y - 2.3, curvature=-0.2)
    draw_arrow(ax, gating_x + 2.7, gating_y - 1.5, 10.5, 9.2)

    # ===== Bottom: Training Strategy Details =====

    # Training datasets section
    train_y = 4.5
    ax.text(0.5, train_y + 1.5, 'Training Strategy:',
            fontsize=11, fontweight='bold', color=COLORS['module_border'])

    # Baseline training data
    draw_component_box(ax, 0.5, train_y, 2.5, 1.2,
                      'Baseline Training\nAll 6 Classes\n(N=16,287)',
                      COLORS['baseline'], fontsize=8)

    # Expert training data
    draw_component_box(ax, 3.5, train_y, 2.5, 1.2,
                      'Expert Training\nClasses 5, 7 Only\n(N=280)',
                      COLORS['expert'], fontsize=8)

    # Gating training data
    draw_component_box(ax, 6.5, train_y, 3, 1.2,
                      'Gating Network Training\nAll 6 Classes\nWeighted CE Loss',
                      COLORS['gating'], fontsize=8)

    # Class distribution visualization
    dist_y = 2.2
    ax.text(0.5, dist_y + 1.2, 'Class Distribution & Weights:',
            fontsize=11, fontweight='bold', color=COLORS['module_border'])

    # Draw bar chart for class distribution
    classes = ['5', '6', '7', '8', '9', '10']
    samples = [215, 1034, 65, 4408, 1089, 9476]
    weights = [12.63, 2.62, 41.75, 0.62, 2.49, 0.29]
    colors_list = [COLORS['expert'], COLORS['baseline'],
                   COLORS['expert'], COLORS['baseline'],
                   COLORS['baseline'], COLORS['baseline']]

    x_positions = np.arange(len(classes))
    bar_width = 0.35

    # Sample count bars
    for i, (x, count, weight) in enumerate(zip(x_positions, samples, weights)):
        # Scale for visualization
        bar_height = count / 100  # Scale down

        # Color based on minority/majority
        color = COLORS['expert'] if classes[i] in ['5', '7'] else COLORS['baseline']

        rect = Rectangle((0.5 + i * 0.8, dist_y - bar_height),
                        0.5, bar_height,
                        facecolor=color, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)

        # Label
        ax.text(0.75 + i * 0.8, dist_y - bar_height - 0.15,
                f'C{classes[i]}', ha='center', fontsize=7)
        ax.text(0.75 + i * 0.8, dist_y - bar_height - 0.3,
                f'{count}', ha='center', fontsize=6)

    # Weight annotations
    for i, (x, weight) in enumerate(zip(x_positions, weights)):
        if classes[i] in ['5', '7']:  # Only show minority weights
            ax.text(0.75 + i * 0.8, dist_y + 0.15,
                    f'w={weight:.2f}',
                    ha='center', fontsize=6,
                    color=COLORS['highlight'], fontweight='bold')

    # Decision logic
    decision_y = 0.5
    ax.text(6.5, decision_y + 1.2, 'Gating Decision Logic:',
            fontsize=11, fontweight='bold', color=COLORS['module_border'])

    draw_component_box(ax, 6.5, decision_y + 0.6, 3, 0.5,
                      'Majority Class → Trust Baseline',
                      COLORS['baseline'], fontsize=8)
    draw_component_box(ax, 6.5, decision_y, 3, 0.5,
                      'Minority Class → Trust Expert',
                      COLORS['expert'], fontsize=8)

    # Performance metrics
    perf_x = 10.5
    ax.text(perf_x, train_y + 1.5, 'Performance:',
            fontsize=11, fontweight='bold', color=COLORS['module_border'])

    metrics = [
        ('Baseline Macro-F1', '0.63'),
        ('GEE Macro-F1', '0.83'),
        ('Improvement', '+0.20'),
    ]

    for i, (label, value) in enumerate(metrics):
        y_pos = train_y - i * 0.5
        ax.text(perf_x, y_pos, f'{label}:',
                fontsize=9, va='center')
        ax.text(perf_x + 2.2, y_pos, value,
                fontsize=9, va='center',
                fontweight='bold' if i == 2 else 'normal',
                color=COLORS['highlight'] if i == 2 else 'black')

    # Legend
    legend_y = 0.3
    ax.text(10.5, legend_y, 'Legend:',
            fontsize=9, fontweight='bold')

    legend_items = [
        (COLORS['baseline'], 'Majority Class'),
        (COLORS['expert'], 'Minority Class'),
        (COLORS['gating'], 'Gating Component'),
        (COLORS['highlight'], 'Key Metric'),
    ]

    for i, (color, label) in enumerate(legend_items):
        y_pos = legend_y - 0.25 - i * 0.25
        rect = Rectangle((10.5, y_pos - 0.08), 0.25, 0.18,
                        facecolor=color, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(10.85, y_pos, label, ha='left', va='center', fontsize=7)

    plt.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)

    return fig

# =============================================================================
# Main
# =============================================================================

def main():
    """Generate all three figures"""
    import os

    # Create output directory
    output_dir = 'thesis/figures'
    os.makedirs(output_dir, exist_ok=True)

    print("Generating GEE Architecture Figures...")

    # Figure 3-1
    print("\n1. Creating Figure 3-1: GEE Architecture Overview...")
    fig1 = create_figure_3_1()
    fig1.savefig(f'{output_dir}/figure_3_1_gee_architecture_overview.png',
                 bbox_inches='tight', dpi=300)
    fig1.savefig(f'{output_dir}/figure_3_1_gee_architecture_overview.pdf',
                 bbox_inches='tight', dpi=300)
    plt.close(fig1)
    print("   ✓ Saved: figure_3_1_gee_architecture_overview.png/pdf")

    # Figure 3-2
    print("\n2. Creating Figure 3-2: Detailed Component Architecture...")
    fig2 = create_figure_3_2()
    fig2.savefig(f'{output_dir}/figure_3_2_detailed_component_architecture.png',
                 bbox_inches='tight', dpi=300)
    fig2.savefig(f'{output_dir}/figure_3_2_detailed_component_architecture.pdf',
                 bbox_inches='tight', dpi=300)
    plt.close(fig2)
    print("   ✓ Saved: figure_3_2_detailed_component_architecture.png/pdf")

    # Figure 3-3
    print("\n3. Creating Figure 3-3: Complete GEE Framework...")
    fig3 = create_figure_3_3()
    fig3.savefig(f'{output_dir}/figure_3_3_complete_gee_framework.png',
                 bbox_inches='tight', dpi=300)
    fig3.savefig(f'{output_dir}/figure_3_3_complete_gee_framework.pdf',
                 bbox_inches='tight', dpi=300)
    plt.close(fig3)
    print("   ✓ Saved: figure_3_3_complete_gee_framework.png/pdf")

    print("\n" + "="*60)
    print("All figures generated successfully!")
    print("="*60)
    print(f"\nOutput directory: {output_dir}/")
    print("\nGenerated files:")
    print("  • figure_3_1_gee_architecture_overview.png/pdf")
    print("  • figure_3_2_detailed_component_architecture.png/pdf")
    print("  • figure_3_3_complete_gee_framework.png/pdf")

if __name__ == '__main__':
    main()
