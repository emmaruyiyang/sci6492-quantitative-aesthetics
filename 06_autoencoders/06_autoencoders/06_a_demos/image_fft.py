import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle

IMAGE_FILE = 'data/modi_a.jpg'
SIZE = 256

THIS_PATH = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(THIS_PATH, IMAGE_FILE)

# Color scheme for the visualization
BG_COLOR = (0.1, 0.2, 0.3)        # Dark blue-gray background
TEXT_COLOR = (1, 1, 1)            # White text
WAVE_COLOR = (0.5, 1.0, 0.9)      # Cyan for waveforms and highlights
AXES_COLOR = (0.2, 0.3, 0.4)      # Lighter blue-gray for axes
CUTOFF_COLOR = (1.0, 0.6, 0.0)    # Orange for cutoff line
SPECTROGRAM_CMAP = 'plasma'       # Colormap for spectrogram

def load_and_process_image(image_path, size):
    """Load and preprocess the image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Could not load image from: {image_path}")
    
    # Resize image
    img = cv2.resize(img, (size, size))
    
    return img.astype(np.float32) / 255.0

def compute_fft(image):
    """Compute 2D FFT of the image"""
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    return fft_shifted

def create_frequency_mask(shape, low_freq, high_freq):
    """Create a mask to filter frequency bands"""
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    
    # Create coordinate arrays
    y, x = np.ogrid[:rows, :cols]
    
    # Calculate distance from center
    distance = np.sqrt((x - center_col)**2 + (y - center_row)**2)
    
    # Create mask: 1 for frequencies to keep, 0 for frequencies to remove
    mask = np.ones((rows, cols))
    
    # Remove frequencies in the specified band
    mask[(distance >= low_freq) & (distance <= high_freq)] = 0
    
    return mask

def apply_frequency_filter(fft_shifted, mask):
    """Apply frequency filter and reconstruct image"""
    # Apply mask
    filtered_fft = fft_shifted * mask
    
    # Inverse FFT
    fft_ishifted = np.fft.ifftshift(filtered_fft)
    reconstructed = np.fft.ifft2(fft_ishifted)
    reconstructed = np.real(reconstructed)
    
    return filtered_fft, reconstructed

class InteractiveFFTVisualization:
    def __init__(self, image_path, size=256):
        self.size = size
        self.image = load_and_process_image(image_path, size)
        self.fft_original = compute_fft(self.image)
        self.max_freq = min(size // 2, int(np.sqrt(2) * size // 2))
        
        # Apply color scheme to matplotlib
        plt.style.use('dark_background')
        
        # Create the figure and subplots
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.patch.set_facecolor(BG_COLOR)
        self.fig.suptitle('Interactive FFT Frequency Filtering', fontsize=16, color=TEXT_COLOR)
        
        # Initialize plots
        self.setup_plots()
        
        # Create sliders
        self.create_sliders()
        
        # Initial update
        self.update(None)
    
    def setup_plots(self):
        """Setup the three subplots"""
        for ax in self.axes:
            ax.set_facecolor(BG_COLOR)
        
        # Original image
        self.axes[0].set_title('Original Image', color=TEXT_COLOR, fontsize=12)
        self.im_original = self.axes[0].imshow(self.image, cmap='gray')
        self.axes[0].axis('off')
        
        # FFT magnitude (log scale) - using the defined spectrogram colormap
        self.axes[1].set_title('FFT Magnitude Spectrum', color=TEXT_COLOR, fontsize=12)
        magnitude = np.log(np.abs(self.fft_original) + 1)
        self.im_fft = self.axes[1].imshow(magnitude, cmap=SPECTROGRAM_CMAP)
        self.axes[1].axis('off')
        
        # Reconstructed image
        self.axes[2].set_title('Filtered Reconstruction', color=TEXT_COLOR, fontsize=12)
        self.im_reconstructed = self.axes[2].imshow(self.image, cmap='gray')
        self.axes[2].axis('off')
        
        plt.tight_layout()
    
    def create_sliders(self):
        """Create the frequency band sliders"""
        # Make space for sliders
        plt.subplots_adjust(bottom=0.25)
        
        # Low frequency slider
        ax_low = plt.axes([0.2, 0.1, 0.5, 0.03])
        ax_low.set_facecolor(AXES_COLOR)
        self.slider_low = Slider(
            ax_low, 'Low Freq', 0, self.max_freq, 
            valinit=0, valfmt='%d',
            color=WAVE_COLOR, alpha=0.8
        )
        ax_low.spines['bottom'].set_color(TEXT_COLOR)
        ax_low.spines['top'].set_color(TEXT_COLOR)
        ax_low.spines['left'].set_color(TEXT_COLOR)
        ax_low.spines['right'].set_color(TEXT_COLOR)
        ax_low.tick_params(colors=TEXT_COLOR)
        ax_low.xaxis.label.set_color(TEXT_COLOR)
        ax_low.yaxis.label.set_color(TEXT_COLOR)
        
        # High frequency slider
        ax_high = plt.axes([0.2, 0.05, 0.5, 0.03])
        ax_high.set_facecolor(AXES_COLOR)
        self.slider_high = Slider(
            ax_high, 'High Freq', 0, self.max_freq, 
            valinit=self.max_freq, valfmt='%d',
            color=CUTOFF_COLOR, alpha=0.8
        )
        ax_high.spines['bottom'].set_color(TEXT_COLOR)
        ax_high.spines['top'].set_color(TEXT_COLOR)
        ax_high.spines['left'].set_color(TEXT_COLOR)
        ax_high.spines['right'].set_color(TEXT_COLOR)
        ax_high.tick_params(colors=TEXT_COLOR)
        ax_high.xaxis.label.set_color(TEXT_COLOR)
        ax_high.yaxis.label.set_color(TEXT_COLOR)
        
        # Connect sliders to update function
        self.slider_low.on_changed(self.update)
        self.slider_high.on_changed(self.update)
    
    def update(self, val):
        """Update the visualization when sliders change"""
        low_freq = self.slider_low.val
        high_freq = self.slider_high.val
        
        # Ensure low_freq <= high_freq
        if low_freq > high_freq:
            return
        
        # Create frequency mask
        mask = create_frequency_mask(self.image.shape, low_freq, high_freq)
        
        # Apply filter and reconstruct
        filtered_fft, reconstructed = apply_frequency_filter(self.fft_original, mask)
        
        # Update FFT display
        magnitude = np.log(np.abs(filtered_fft) + 1)
        self.im_fft.set_array(magnitude)
        
        # Update reconstructed image
        # Clip values to [0, 1] range for proper display
        reconstructed_display = np.clip(reconstructed, 0, 1)
        self.im_reconstructed.set_array(reconstructed_display)
        
        # Add visual indicators on FFT plot
        self.axes[1].clear()
        self.axes[1].set_facecolor(BG_COLOR)
        self.axes[1].imshow(magnitude, cmap=SPECTROGRAM_CMAP)
        self.axes[1].set_title(f'FFT Spectrum (Removing {low_freq:.0f}-{high_freq:.0f} freq band)', 
                              color=TEXT_COLOR, fontsize=12)
        self.axes[1].axis('off')
        
        # Add circles to show the frequency bands being filtered
        if low_freq < high_freq:
            center = self.size // 2
            circle_low = plt.Circle((center, center), low_freq, fill=False, 
                                  color=WAVE_COLOR, linewidth=2, alpha=0.9)
            circle_high = plt.Circle((center, center), high_freq, fill=False, 
                                   color=CUTOFF_COLOR, linewidth=2, alpha=0.9)
            self.axes[1].add_patch(circle_low)
            self.axes[1].add_patch(circle_high)
        
        self.fig.canvas.draw()
    
    def show(self):
        """Display the interactive visualization"""
        plt.show()

def main():
    """Main function to run the interactive FFT visualization"""
    try:
        # Create and show the interactive visualization
        viz = InteractiveFFTVisualization(IMAGE_PATH, SIZE)
        viz.show()
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Make sure the image file exists at: {IMAGE_PATH}")

if __name__ == "__main__":
    main()


