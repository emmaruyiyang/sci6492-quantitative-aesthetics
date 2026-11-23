import numpy as np
import matplotlib.pyplot as plt

# Color scheme for the visualization
BG_COLOR = (0.1, 0.2, 0.3)        # Dark blue-gray background
TEXT_COLOR = (1, 1, 1)            # White text
WAVE_COLOR = (0.5, 1.0, 0.9)      # Cyan for waveforms and highlights
AXES_COLOR = (0.2, 0.3, 0.4)      # Lighter blue-gray for axes
F1_COLOR = (1.0, 0.3, 0.5)     
F2_COLOR = (0.3, 0.5, 1.0)
G1_COLOR = (1.0, 0.8, 1.0)
G2_COLOR = (0.8, 1.0, 0.8)
H_COLOR = (1.0, 1.0, 1.0)


def create_square_impulse(x, center, amplitude, width=0.1):
    """Create a square impulse function"""
    return np.where(np.abs(x - center) <= width/2, amplitude, 0)

def create_gaussian(x, center, amplitude, sigma=0.4):
    """Create a Gaussian function"""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)

def plot_five_functions():
    """Plot the five functions as specified"""
    # Create x range
    x = np.linspace(-3, 3, 1000)
    
    # Define the functions
    f1 = create_square_impulse(x, center=-1, amplitude=0.5)  # Square impulse at x=-1, max 0.5
    f2 = create_square_impulse(x, center=0.5, amplitude=1.0)  # Square impulse at x=+0.5, max 1.0
    
    g1 = create_gaussian(x, center=-1, amplitude=0.5)  # Gaussian scaled like f1
    g2 = create_gaussian(x, center=0.5, amplitude=1.0)  # Gaussian scaled like f2
    
    h = g1 + g2  # Sum of the two Gaussians
    
    # Create the plot with color scheme
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    plt.plot(x, f1, label='f1: Square impulse at x=-1 (amp=0.5)', linewidth=2, color=F1_COLOR)
    plt.plot(x, f2, label='f2: Square impulse at x=+0.5 (amp=1.0)', linewidth=2, color=F2_COLOR)
    plt.plot(x, g1, label='g1: Gaussian at x=-1 (amp=0.5)', linewidth=2, color=G1_COLOR, linestyle='--')
    plt.plot(x, g2, label='g2: Gaussian at x=+0.5 (amp=1.0)', linewidth=2, color=G2_COLOR, linestyle='--')
    plt.plot(x, h, label='h: Sum of g1+g2', linewidth=3, color=H_COLOR)
    
    # Add vertical lines at the centers
    plt.axvline(-1, color=AXES_COLOR, linestyle=':', alpha=0.7, label='Centers')
    plt.axvline(0.5, color=AXES_COLOR, linestyle=':', alpha=0.7)
    
    # Customize the plot with color scheme
    plt.grid(True, alpha=0.3, color=AXES_COLOR)
    plt.xlabel('x', fontsize=12, color=TEXT_COLOR)
    plt.ylabel('Amplitude', fontsize=12, color=TEXT_COLOR)
    plt.title('Five Superimposed Functions:\nSquare Impulses (f1, f2), Gaussians (g1, g2), and their Sum (h)', 
              fontsize=14, color=TEXT_COLOR)
    
    # Style the axes
    ax.spines['bottom'].set_color(AXES_COLOR)
    ax.spines['top'].set_color(AXES_COLOR)
    ax.spines['left'].set_color(AXES_COLOR)
    ax.spines['right'].set_color(AXES_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    
    # Legend with styled colors
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                       facecolor=BG_COLOR, edgecolor=AXES_COLOR)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    plt.xlim(-3, 3)
    plt.ylim(-0.1, 1.2)
    
    # Tight layout to prevent legend cutoff
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_five_functions()