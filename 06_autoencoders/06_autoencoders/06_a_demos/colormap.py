"""
Matplotlib Colormap Visualization
Displays all available matplotlib colormaps as gradient bars
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def plot_colormap_gradients():
    """
    Create a comprehensive plot showing all matplotlib colormaps
    organized by category with gradient bars
    """
    
    # Get all available colormaps organized by category
    colormap_categories = {
        'Perceptually Uniform Sequential': [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        ],
        'Sequential': [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
        ],
        'Sequential (2)': [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper'
        ],
        'Diverging': [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'
        ],
        'Cyclic': [
            'twilight', 'twilight_shifted', 'hsv'
        ],
        'Qualitative': [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c'
        ],
        'Miscellaneous': [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'
        ]
    }
    
    # Create gradient data
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    
    # Calculate total number of colormaps for figure sizing
    total_cmaps = sum(len(cmaps) for cmaps in colormap_categories.values())
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(colormap_categories), 1, 
                            figsize=(12, total_cmaps * 0.4))
    
    # If only one category, make axes iterable
    if len(colormap_categories) == 1:
        axes = [axes]
    
    # Plot each category
    for ax_idx, (category, cmaps) in enumerate(colormap_categories.items()):
        ax = axes[ax_idx]
        
        # Create subplot for this category
        for i, cmap_name in enumerate(cmaps):
            try:
                # Get the colormap
                cmap = plt.get_cmap(cmap_name)
                
                # Plot the gradient
                ax.imshow(gradient, aspect='auto', cmap=cmap, 
                         extent=[0, 1, i, i+1])
                
                # Add colormap name
                ax.text(1.02, i+0.5, cmap_name, va='center', ha='left',
                       fontsize=9, transform=ax.transData)
                
            except Exception as e:
                print(f"Could not display colormap '{cmap_name}': {e}")
        
        # Set up the axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(cmaps))
        ax.set_yticks([])
        ax.set_title(category, fontsize=12, fontweight='bold', pad=10)
        
        # Only show x-axis on bottom plot
        if ax_idx < len(colormap_categories) - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel('Colormap Range (0 to 1)', fontsize=10)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)  # Make room for colormap names
    
    # Add main title
    fig.suptitle('Matplotlib Colormaps by Category', 
                fontsize=16, fontweight='bold', y=0.98)
    
    return fig

def plot_selected_colormaps(colormap_names=None):
    """
    Plot a selection of specific colormaps
    
    Args:
        colormap_names (list): List of colormap names to display
    """
    if colormap_names is None:
        # Default selection of popular colormaps
        colormap_names = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'coolwarm', 'RdYlBu', 'Spectral', 'jet', 'hsv',
            'hot', 'cool', 'spring', 'autumn', 'winter'
        ]
    
    # Create gradient data
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(colormap_names) * 0.5))
    
    # Plot each colormap
    for i, cmap_name in enumerate(colormap_names):
        try:
            cmap = plt.get_cmap(cmap_name)
            ax.imshow(gradient, aspect='auto', cmap=cmap, 
                     extent=[0, 1, i, i+1])
            ax.text(1.02, i+0.5, cmap_name, va='center', ha='left',
                   fontsize=10, transform=ax.transData)
        except Exception as e:
            print(f"Could not display colormap '{cmap_name}': {e}")
    
    # Set up the axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(colormap_names))
    ax.set_yticks([])
    ax.set_xlabel('Colormap Range (0 to 1)', fontsize=12)
    ax.set_title('Selected Matplotlib Colormaps', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    
    return fig

def demonstrate_colormap_usage():
    """
    Demonstrate how different colormaps affect data visualization
    """
    # Create sample data
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y) * np.exp(-(X**2 + Y**2)/4)
    
    # Popular colormaps to compare
    cmaps = ['viridis', 'coolwarm', 'RdYlBu', 'hot']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, cmap_name in enumerate(cmaps):
        im = axes[i].contourf(X, Y, Z, levels=20, cmap=cmap_name)
        axes[i].set_title(f'Colormap: {cmap_name}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[i])
        cbar.set_label('Value')
    
    plt.suptitle('Colormap Comparison on Sample Data', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
    # Create all plots
    print("Creating comprehensive colormap visualization...")
    fig1 = plot_colormap_gradients()
    
    print("Creating selected colormaps plot...")
    fig2 = plot_selected_colormaps()
    
    print("Creating colormap usage demonstration...")
    fig3 = demonstrate_colormap_usage()
    
    # Show all plots
    plt.show()
    
    print("\nColormap visualization complete!")
    print("\nTip: You can also call individual functions:")
    print("- plot_colormap_gradients() for all colormaps by category")
    print("- plot_selected_colormaps(['viridis', 'plasma']) for specific colormaps")
    print("- demonstrate_colormap_usage() for colormap comparison on data")
