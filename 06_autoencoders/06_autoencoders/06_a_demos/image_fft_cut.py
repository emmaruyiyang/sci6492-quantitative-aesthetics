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

def create_magnitude_mask(fft_data, threshold):
    """Create a mask based on magnitude threshold"""
    # Calculate magnitude of FFT
    magnitude = np.abs(fft_data)
    
    # Create mask: 1 for magnitudes above threshold, 0 for below
    mask = (magnitude >= threshold).astype(float)
    
    return mask

def apply_magnitude_filter(fft_shifted, threshold):
    """Apply magnitude filter and reconstruct image"""
    # Create mask based on magnitude threshold
    mask = create_magnitude_mask(fft_shifted, threshold)
    
    # Apply mask
    filtered_fft = fft_shifted * mask
    
    # Inverse FFT
    fft_ishifted = np.fft.ifftshift(filtered_fft)
    reconstructed = np.fft.ifft2(fft_ishifted)
    reconstructed = np.real(reconstructed)
    
    return filtered_fft, reconstructed, mask

class InteractiveFFTVisualization:
    def __init__(self, image_path, size=256):
        self.size = size
        self.image = load_and_process_image(image_path, size)
        self.fft_original = compute_fft(self.image)
        
        # Calculate magnitude statistics for slider range
        magnitude = np.abs(self.fft_original)
        self.max_magnitude = np.max(magnitude)
        self.min_magnitude = np.min(magnitude[magnitude > 0])  # Exclude zeros
        
        # Apply color scheme to matplotlib
        plt.style.use('dark_background')
        
        # Create the figure and subplots
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.patch.set_facecolor(BG_COLOR)
        self.fig.suptitle('Interactive FFT Magnitude Threshold Filtering', fontsize=16, color=TEXT_COLOR)
        
        # Initialize plots
        self.setup_plots()
        
        # Create slider
        self.create_slider()
        
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
    
    def create_slider(self):
        """Create the magnitude threshold slider"""
        # Make space for slider
        plt.subplots_adjust(bottom=0.25)
        
        # Magnitude threshold slider - use log scale for better control
        ax_threshold = plt.axes([0.2, 0.1, 0.5, 0.03])
        ax_threshold.set_facecolor(AXES_COLOR)
        
        # Use log scale for the slider range
        log_min = np.log10(self.min_magnitude) if self.min_magnitude > 0 else -6
        log_max = np.log10(self.max_magnitude)
        # Start with a threshold that keeps most frequencies
        initial_threshold = log_min + 0.3 * (log_max - log_min)
        
        self.slider_threshold = Slider(
            ax_threshold, 'Magnitude Threshold', log_min, log_max, 
            valinit=initial_threshold, valfmt='%.2f',
            color=WAVE_COLOR, alpha=0.8
        )
        ax_threshold.spines['bottom'].set_color(TEXT_COLOR)
        ax_threshold.spines['top'].set_color(TEXT_COLOR)
        ax_threshold.spines['left'].set_color(TEXT_COLOR)
        ax_threshold.spines['right'].set_color(TEXT_COLOR)
        ax_threshold.tick_params(colors=TEXT_COLOR)
        ax_threshold.xaxis.label.set_color(TEXT_COLOR)
        ax_threshold.yaxis.label.set_color(TEXT_COLOR)
        
        # Connect slider to update function
        self.slider_threshold.on_changed(self.update)
    
    def update(self, val):
        """Update the visualization when slider changes"""
        # Get threshold value from slider (in log scale)
        log_threshold = self.slider_threshold.val
        threshold = 10 ** log_threshold
        
        # Apply magnitude filter and reconstruct
        filtered_fft, reconstructed, mask = apply_magnitude_filter(self.fft_original, threshold)
        
        # Update FFT display
        magnitude = np.log(np.abs(filtered_fft) + 1)
        self.im_fft.set_array(magnitude)
        
        # Update reconstructed image
        # Clip values to [0, 1] range for proper display
        reconstructed_display = np.clip(reconstructed, 0, 1)
        self.im_reconstructed.set_array(reconstructed_display)
        
        # Update FFT plot with mask visualization
        self.axes[1].clear()
        self.axes[1].set_facecolor(BG_COLOR)
        self.axes[1].imshow(magnitude, cmap=SPECTROGRAM_CMAP)
        
        # Count kept vs removed frequencies
        total_pixels = mask.size
        kept_pixels = np.sum(mask)
        kept_percentage = (kept_pixels / total_pixels) * 100
        
        self.axes[1].set_title(f'FFT Spectrum (Threshold: {threshold:.2e})\nKept: {kept_percentage:.1f}% of frequencies', 
                              color=TEXT_COLOR, fontsize=12)
        self.axes[1].axis('off')
        
        # Overlay the mask to show which frequencies are kept/removed
        # Red overlay shows removed frequencies
        mask_overlay = np.ma.masked_where(mask == 1, np.ones_like(mask))
        self.axes[1].imshow(mask_overlay, cmap='Reds', alpha=0.4, vmin=0, vmax=1)
        
        self.fig.canvas.draw_idle()
    
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


