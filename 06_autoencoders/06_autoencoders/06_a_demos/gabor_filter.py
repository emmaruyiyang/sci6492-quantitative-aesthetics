import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

# Color scheme for the visualization
BG_COLOR = (0.1, 0.2, 0.3)        # Dark blue-gray background
TEXT_COLOR = (1, 1, 1)            # White text
WAVE_COLOR = (0.5, 1.0, 0.9)      # Cyan for waveforms and highlights
AXES_COLOR = (0.2, 0.3, 0.4)      # Lighter blue-gray for axes
CUTOFF_COLOR = (1.0, 0.6, 0.0)    # Orange for cutoff line

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGES = ['data/modi_a.jpg', 'data/modi_b.jpg']

KERNEL_SIZE = 101  # Size of Gabor filter images

def create_gabor_kernel(sigma=5, gamma=0.5, lambda_val=10, theta=0, psi=0, ksize=KERNEL_SIZE):
    """
    Create a Gabor filter kernel
    
    Parameters:
    - sigma: standard deviation of Gaussian envelope
    - gamma: aspect ratio (ellipticity of support)
    - lambda_val: wavelength of sinusoidal factor
    - theta: orientation of the normal to parallel stripes (in radians)
    - psi: phase offset (in radians)
    - ksize: kernel size
    """
    # Create coordinate grids
    y, x = np.meshgrid(np.arange(-ksize//2, ksize//2 + 1),
                       np.arange(-ksize//2, ksize//2 + 1))
    
    # Rotate coordinates
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    
    # Gaussian envelope
    gaussian = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2))
    
    # Sinusoidal grating
    sinusoid = np.cos(2 * np.pi * x_theta / lambda_val + psi)
    
    # Gabor filter
    gabor = gaussian * sinusoid
    
    return gabor

def apply_gabor_filter(image, gabor_kernel):
    """Apply Gabor filter to image using convolution"""
    if len(image.shape) == 3:
        # Convert to grayscale if color image
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image
    
    # Apply filter using OpenCV filter2D
    filtered = cv2.filter2D(image_gray, cv2.CV_32F, gabor_kernel)
    
    return filtered

def load_and_prepare_image(image_path, max_size=300):
    """Load and resize image for processing"""
    image = cv2.imread(image_path)
    if image is None:
        # Create a synthetic test image if file not found
        image = create_test_image(max_size)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image if too large
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
    
    return image

def create_test_image(size=300):
    """Create a test image with various patterns"""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Add some geometric patterns
    cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(image, (200, 200), 50, (128, 128, 128), -1)
    
    # Add some lines with different orientations
    for i in range(0, size, 20):
        cv2.line(image, (i, 0), (i, size//2), (64, 64, 64), 2)
        cv2.line(image, (0, size//2 + i//2), (size, size//2 + i//2), (192, 192, 192), 1)
    
    return image

class InteractiveGaborFilter:
    def __init__(self):
        # Load image
        image_path = os.path.join(THIS_FOLDER, IMAGES[0])
        self.original_image = load_and_prepare_image(image_path)
        
        # Initial parameters
        self.sigma = 5.0
        self.gamma = 0.5
        self.lambda_val = 10.0
        self.theta = 0.0
        self.psi = 0.0
        
        self.setup_plot()
        self.update_display()
        
    def setup_plot(self):
        """Setup the matplotlib figure and axes"""
        plt.style.use('dark_background')
        
        # Create figure with custom layout
        self.fig = plt.figure(figsize=(15, 8))
        self.fig.patch.set_facecolor(BG_COLOR)
        
        # Create grid layout - images on top, sliders below
        gs = gridspec.GridSpec(3, 3, figure=self.fig, 
                              height_ratios=[1, 1, 0.4], width_ratios=[1, 1, 1])
        
        # Main image displays
        self.ax_original = self.fig.add_subplot(gs[0:2, 0])
        self.ax_kernel = self.fig.add_subplot(gs[0:2, 1])
        self.ax_filtered = self.fig.add_subplot(gs[0:2, 2])
        
        # Setup axes
        for ax, title in zip([self.ax_original, self.ax_kernel, self.ax_filtered],
                           ['Original Image', 'Gabor Kernel', 'Filtered Image']):
            ax.set_title(title, color=TEXT_COLOR, fontsize=12, pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor(BG_COLOR)
        
        # Create sliders
        self.create_sliders()
        
        # Adjust layout manually for vertical slider stack at bottom
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.35, hspace=0.3, wspace=0.3)
        
    def create_sliders(self):
        """Create parameter sliders"""
        # Slider positions (left, bottom, width, height) - vertical stack under images
        slider_height = 0.025
        slider_width = 0.6  # Wider sliders since they're stacked vertically
        vertical_spacing = 0.035
        base_left = 0.2  # Center the sliders horizontally
        base_bottom = 0.22  # Start higher to accommodate 5 sliders
        
        # Sigma slider (Gaussian std dev)
        ax_sigma = plt.axes([base_left, base_bottom, slider_width, slider_height])
        self.slider_sigma = Slider(ax_sigma, 'σ (blur)', 1.0, 20.0, valinit=self.sigma, 
                                  facecolor=WAVE_COLOR, alpha=0.7)
        
        # Gamma slider (aspect ratio)
        ax_gamma = plt.axes([base_left, base_bottom - vertical_spacing, slider_width, slider_height])
        self.slider_gamma = Slider(ax_gamma, 'γ (aspect)', 0.1, 2.0, valinit=self.gamma,
                                  facecolor=WAVE_COLOR, alpha=0.7)
        
        # Lambda slider (wavelength)
        ax_lambda = plt.axes([base_left, base_bottom - 2*vertical_spacing, slider_width, slider_height])
        self.slider_lambda = Slider(ax_lambda, 'λ (wave)', 5.0, 50.0, valinit=self.lambda_val,
                                   facecolor=WAVE_COLOR, alpha=0.7)
        
        # Theta slider (orientation)
        ax_theta = plt.axes([base_left, base_bottom - 3*vertical_spacing, slider_width, slider_height])
        self.slider_theta = Slider(ax_theta, 'θ (angle)', 0.0, np.pi, valinit=self.theta,
                                  facecolor=WAVE_COLOR, alpha=0.7)
        
        # Psi slider (phase)
        ax_psi = plt.axes([base_left, base_bottom - 4*vertical_spacing, slider_width, slider_height])
        self.slider_psi = Slider(ax_psi, 'ψ (phase)', 0.0, 2*np.pi, valinit=self.psi,
                                facecolor=WAVE_COLOR, alpha=0.7)
        
        # Connect sliders to update function
        self.slider_sigma.on_changed(self.on_slider_change)
        self.slider_gamma.on_changed(self.on_slider_change)
        self.slider_lambda.on_changed(self.on_slider_change)
        self.slider_theta.on_changed(self.on_slider_change)
        self.slider_psi.on_changed(self.on_slider_change)
        
    def on_slider_change(self, val):
        """Handle slider changes"""
        self.sigma = self.slider_sigma.val
        self.gamma = self.slider_gamma.val
        self.lambda_val = self.slider_lambda.val
        self.theta = self.slider_theta.val
        self.psi = self.slider_psi.val
        
        self.update_display()
        
    def update_display(self):
        """Update all three displays"""
        # Create Gabor kernel
        gabor_kernel = create_gabor_kernel(
            sigma=self.sigma,
            gamma=self.gamma,
            lambda_val=self.lambda_val,
            theta=self.theta,
            psi=self.psi
        )
        
        # Apply filter to image
        filtered_image = apply_gabor_filter(self.original_image, gabor_kernel)
        
        # Clear axes
        self.ax_original.clear()
        self.ax_kernel.clear()
        self.ax_filtered.clear()
        
        # Display original image
        self.ax_original.imshow(self.original_image)
        self.ax_original.set_title('Original Image', color=TEXT_COLOR, fontsize=12)
        self.ax_original.set_xticks([])
        self.ax_original.set_yticks([])
        
        # Display Gabor kernel
        kernel_display = (gabor_kernel - gabor_kernel.min()) / (gabor_kernel.max() - gabor_kernel.min())
        self.ax_kernel.imshow(kernel_display, cmap='RdBu_r')
        self.ax_kernel.set_title(f'Gabor Kernel\nσ={self.sigma:.1f}, γ={self.gamma:.1f}', 
                                color=TEXT_COLOR, fontsize=12)
        self.ax_kernel.set_xticks([])
        self.ax_kernel.set_yticks([])
        
        # Display filtered image
        max_abs = np.max(np.abs(filtered_image))
        self.ax_filtered.imshow(filtered_image, cmap='twilight', vmin=-max_abs, vmax=max_abs)
        self.ax_filtered.set_title(f'Filtered Image\nλ={self.lambda_val:.1f}, θ={self.theta:.2f}', 
                                  color=TEXT_COLOR, fontsize=12)
        self.ax_filtered.set_xticks([])
        self.ax_filtered.set_yticks([])
        
        # Refresh display
        self.fig.canvas.draw_idle()

def main():
    """Main function to run the interactive Gabor filter"""
    print("Interactive Gabor Filter Visualization")
    print("Use sliders to adjust filter parameters:")
    print("  σ (sigma): Gaussian standard deviation (blur)")
    print("  γ (gamma): Aspect ratio (ellipticity)")
    print("  λ (lambda): Wavelength of sinusoidal component")
    print("  θ (theta): Orientation angle")
    print("  ψ (psi): Phase offset")
    print("\nClose the window to exit.")
    
    # Create interactive plot
    app = InteractiveGaborFilter()
    plt.show()

if __name__ == "__main__":
    main()

