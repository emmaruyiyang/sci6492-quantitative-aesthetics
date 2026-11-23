import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Color scheme for the visualization
BG_COLOR = (0.1, 0.2, 0.3)        # Dark blue-gray background
TEXT_COLOR = (1, 1, 1)            # White text
WAVE_COLOR = (0.5, 1.0, 0.9)      # Cyan for waveforms and highlights
AXES_COLOR = (0.2, 0.3, 0.4)      # Lighter blue-gray for axes
CUTOFF_COLOR = (1.0, 0.6, 0.0)    # Orange for cutoff line
SPECTROGRAM_CMAP = 'twilight'       # Colormap for spectrogram

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGES = ['data/modi_a.jpg', 'data/modi_b.jpg']
DUMMY_IMAGES = ['8x8 1 pixel', '8x8 h line', '8x8 v line', '8x8 checkerboard', '8x8 diagonal line', '16x16 circle', '16x16 gradient']
FILTERS = ['identity', 'dx', 'dy', 'box_blur', 'gaussian_blur', 'emboss', 'edge_detect', 'sharpen']

def create_dummy_image(image_type):
    """Create various dummy images for testing convolution."""
    if image_type == '8x8 1 pixel':
        img = np.zeros((8, 8))
        img[4, 4] = 255
        return img
    elif image_type == '8x8 h line':
        img = np.zeros((8, 8))
        img[4, :] = 255
        return img
    elif image_type == '8x8 v line':
        img = np.zeros((8, 8))
        img[:, 4] = 255
        return img
    elif image_type == '8x8 checkerboard':
        img = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    img[i, j] = 255
        return img
    elif image_type == '8x8 diagonal line':
        img = np.zeros((8, 8))
        for i in range(8):
            img[i, i] = 255
        return img
    elif image_type == '16x16 circle':
        img = np.zeros((16, 16))
        center = (8, 8)
        radius = 4
        y, x = np.ogrid[:16, :16]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        img[mask] = 255
        return img
    elif image_type == '16x16 gradient':
        img = np.zeros((16, 16))
        for i in range(16):
            img[i, :] = i * 16
        return img
    else:
        return np.zeros((8, 8))

def get_kernel(filter_name):
    """Get convolution kernels for different filters."""
    if filter_name == 'identity':
        return np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]])
    elif filter_name == 'dx':
        return np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    elif filter_name == 'dy':
        return np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    elif filter_name == 'box_blur':
        return np.ones((3, 3)) / 9
    elif filter_name == 'gaussian_blur':
        return np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]]) / 16
    elif filter_name == 'emboss':
        return np.array([[-2, -1, 0],
                        [-1,  1, 1],
                        [ 0,  1, 2]])
    elif filter_name == 'edge_detect':
        return np.array([[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]])
    elif filter_name == 'sharpen':
        return np.array([[ 0, -1,  0],
                        [-1,  5, -1],
                        [ 0, -1,  0]])
    else:
        return np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]])

def apply_convolution(image, kernel):
    """Apply convolution filter to image."""
    if len(image.shape) == 3:
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply convolution
    result = cv2.filter2D(image, -1, kernel)
    return result

def visualize_kernel(ax, kernel, title):
    """Visualize convolution kernel as a pixelated table with values."""
    ax.clear()
    ax.set_facecolor(BG_COLOR)
    
    # Create colormap for the kernel values
    vmin, vmax = kernel.min(), kernel.max()

    cmap = SPECTROGRAM_CMAP  # Use the defined colormap
    
    abs_max = max(abs(vmin), abs(vmax))
    # Display the kernel as an image
    if vmin < 0:
        im = ax.imshow(kernel, cmap=cmap, aspect='equal', vmin=-abs_max, vmax=abs_max)
    else:
        im = ax.imshow(kernel, cmap='bone', aspect='equal', vmin=0, vmax=1)

    # Add text annotations for each cell
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            value = kernel[i, j]
            # Use white text for better visibility on dark background
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                   color=TEXT_COLOR, fontweight='bold', fontsize=10)
    
    # Set up the plot with color scheme
    ax.set_title(title, fontsize=12, fontweight='bold', color=TEXT_COLOR)
    ax.set_xticks(range(kernel.shape[1]))
    ax.set_yticks(range(kernel.shape[0]))
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, color=WAVE_COLOR, linewidth=1, alpha=0.3)
    
    # Add colorbar with color scheme
    #cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    #cbar.ax.tick_params(colors=TEXT_COLOR)

class ConvolutionVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 8), facecolor=BG_COLOR)
        self.fig.suptitle('Interactive Convolution Visualization', fontsize=16, fontweight='bold', color=TEXT_COLOR)
        
        # Create subplots with dark background
        self.ax_image = plt.subplot(2, 4, 1, facecolor=BG_COLOR)
        self.ax_kernel = plt.subplot(2, 4, 2, facecolor=BG_COLOR)
        self.ax_result = plt.subplot(2, 4, 3, facecolor=BG_COLOR)
        self.ax_enhanced = plt.subplot(2, 4, 4, facecolor=BG_COLOR)
        
        # Create radio button areas
        self.ax_image_radio = plt.subplot(2, 4, 5, facecolor=BG_COLOR)
        self.ax_kernel_radio = plt.subplot(2, 4, 6, facecolor=BG_COLOR)
        self.ax_info = plt.subplot(2, 4, 7, facecolor=BG_COLOR)
        
        # Initialize with default selections
        self.current_image_name = DUMMY_IMAGES[0]
        self.current_kernel_name = FILTERS[0]
        
        # Setup radio buttons
        self.setup_radio_buttons()
        
        # Initial visualization
        self.update_visualization()
        
        plt.tight_layout()
        plt.show()
    
    def setup_radio_buttons(self):
        """Setup radio buttons for image and kernel selection."""
        # Image selection
        all_images = DUMMY_IMAGES + [os.path.basename(img) for img in IMAGES]
        self.radio_images = RadioButtons(self.ax_image_radio, all_images)
        self.radio_images.on_clicked(self.on_image_change)
        self.ax_image_radio.set_title('Select Image', fontweight='bold', color=TEXT_COLOR)
        self.ax_image_radio.tick_params(colors=TEXT_COLOR)
        
        # Style radio button labels with color scheme
        for label in self.radio_images.labels:
            label.set_color(TEXT_COLOR)
        
        # Kernel selection
        self.radio_kernels = RadioButtons(self.ax_kernel_radio, FILTERS)
        self.radio_kernels.on_clicked(self.on_kernel_change)
        self.ax_kernel_radio.set_title('Select Kernel', fontweight='bold', color=TEXT_COLOR)
        self.ax_kernel_radio.tick_params(colors=TEXT_COLOR)
        
        # Style radio button labels with color scheme
        for label in self.radio_kernels.labels:
            label.set_color(TEXT_COLOR)
    
    def on_image_change(self, label):
        """Handle image selection change."""
        self.current_image_name = label
        self.update_visualization()
    
    def on_kernel_change(self, label):
        """Handle kernel selection change."""
        self.current_kernel_name = label
        self.update_visualization()
    
    def load_image(self, image_name):
        """Load image based on selection."""
        if image_name in DUMMY_IMAGES:
            return create_dummy_image(image_name)
        else:
            # Load real image
            for img_path in IMAGES:
                if os.path.basename(img_path) == image_name:
                    image = cv2.imread(os.path.join(THIS_FOLDER, img_path))
                    if image is not None:
                        # Convert BGR to RGB for matplotlib
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        # Resize if too large
                        if image.shape[0] > 256 or image.shape[1] > 256:
                            image = cv2.resize(image, (256, 256))
                        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return create_dummy_image('8x8 1 pixel')  # fallback
    
    def update_visualization(self):
        """Update all four panels of the visualization."""
        # Load current image and kernel
        image = self.load_image(self.current_image_name)
        kernel = get_kernel(self.current_kernel_name)
        
        # Apply convolution
        result = apply_convolution(image, kernel)
        
        # Create enhanced visualization (absolute value or edge magnitude)
        enhanced = np.abs(result)
        
        # Update image panel
        self.ax_image.clear()
        self.ax_image.set_facecolor(BG_COLOR)
        self.ax_image.imshow(image, cmap='gray', vmin=0, vmax=255)
        self.ax_image.set_title(f'Original Image\n{self.current_image_name}', fontweight='bold', color=TEXT_COLOR)
        self.ax_image.axis('off')
        
        # Update kernel panel
        visualize_kernel(self.ax_kernel, kernel, f'Convolution Kernel\n{self.current_kernel_name}')
        
        # Update result panel
        self.ax_result.clear()
        self.ax_result.set_facecolor(BG_COLOR)
        has_negative = np.any(result < 0)
        if has_negative:
            abs_max = np.max(np.abs(result))
            self.ax_result.imshow(result, cmap=SPECTROGRAM_CMAP, vmin=-abs_max, vmax=abs_max)
        else:
            self.ax_result.imshow(result, cmap='gray', vmin=0, vmax=255)
        self.ax_result.set_title('Convolution Result', fontweight='bold', color=TEXT_COLOR)
        self.ax_result.axis('off')
        
        # Update enhanced panel (4th panel)
        self.ax_enhanced.clear()
        self.ax_enhanced.set_facecolor(BG_COLOR)
        self.ax_enhanced.imshow(enhanced, cmap='bone')
        self.ax_enhanced.set_title('Enhanced View\n(Absolute Values)', fontweight='bold', color=TEXT_COLOR)
        self.ax_enhanced.axis('off')
        
        # Add statistics in the info panel (bottom right)
        self.ax_info.clear()
        self.ax_info.set_facecolor(BG_COLOR)
        self.ax_info.axis('off')
        
        stats_text = f'Image Statistics:\n'
        stats_text += f'Original: min={image.min():.1f}, max={image.max():.1f}\n'
        stats_text += f'Result: min={result.min():.1f}, max={result.max():.1f}\n'
        stats_text += f'Enhanced: min={enhanced.min():.1f}, max={enhanced.max():.1f}\n\n'
        stats_text += f'Kernel Statistics:\n'
        stats_text += f'Sum: {kernel.sum():.3f}\n'
        stats_text += f'Mean: {kernel.mean():.3f}\n'
        stats_text += f'Size: {kernel.shape[0]}x{kernel.shape[1]}'
        
        self.ax_info.text(0.05, 0.95, stats_text, transform=self.ax_info.transAxes,
                         verticalalignment='top', bbox=dict(boxstyle='round', 
                         facecolor=AXES_COLOR, edgecolor=WAVE_COLOR, alpha=0.9), 
                         fontsize=10, color=TEXT_COLOR)
        
        plt.draw()

def main():
    """Main function to run the convolution visualizer."""
    print("Starting Interactive Convolution Visualizer...")
    print("Use the radio buttons to select different images and kernels.")
    print("The visualization shows:")
    print("- Left: Original image")
    print("- Center: Convolution kernel (color-coded with values)")
    print("- Right: Result of applying the kernel to the image")
    
    visualizer = ConvolutionVisualizer()

if __name__ == "__main__":
    main()
