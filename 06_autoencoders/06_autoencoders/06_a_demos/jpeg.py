import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from io import BytesIO
from matplotlib.widgets import Slider

BG_COLOR = (0.1, 0.2, 0.3)        # Dark blue-gray background
TEXT_COLOR = (1, 1, 1)            # White text
WAVE_COLOR = (0.5, 1.0, 0.9)      # Cyan for waveforms and highlights
AXES_COLOR = (0.2, 0.3, 0.4)      # Lighter blue-gray for axes
CUTOFF_COLOR = (1.0, 0.6, 0.0)    # Orange for cutoff line

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGES = ['data/modi_a.jpg', 'data/modi_b.jpg']

def compress_image_jpeg(image, quality=50, subsampling=2):
    """
    Compress an image using JPEG compression with specified quality and subsampling.
    
    Args:
        image: PIL Image object
        quality: JPEG quality (1-100, higher = better quality)
        subsampling: Chroma subsampling (0=4:4:4, 1=4:2:2, 2=4:2:0)
    
    Returns:
        PIL Image object of compressed image
    """
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Create a BytesIO object to simulate file saving/loading
    buffer = BytesIO()
    
    # Save with JPEG compression
    image.save(buffer, 
               format='JPEG', 
               quality=quality, 
               subsampling=subsampling)
    
    # Load the compressed image back
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    
    return compressed_image

def create_interactive_jpeg_demo():
    """Create an interactive demo showing original vs JPEG compressed images."""
    
    # Load the first image as default
    image_path = os.path.join(THIS_FOLDER, IMAGES[0])
    original_image = Image.open(image_path)
    
    # Create figure and subplots with color scheme
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle('JPEG Compression Demo', fontsize=16, color=TEXT_COLOR)
    
    # Initial compression
    compressed_image = compress_image_jpeg(original_image, quality=50, subsampling=2)
    
    # Display images
    im1 = ax1.imshow(np.array(original_image))
    ax1.set_title('Original Image', color=TEXT_COLOR)
    ax1.set_facecolor(AXES_COLOR)
    ax1.axis('off')
    
    im2 = ax2.imshow(np.array(compressed_image))
    ax2.set_title('JPEG Compressed (Quality: 50, Subsampling: 4:2:0)', color=TEXT_COLOR)
    ax2.set_facecolor(AXES_COLOR)
    ax2.axis('off')
    
    # Create sliders with color scheme
    plt.subplots_adjust(bottom=0.25)
    
    # Quality slider
    ax_quality = plt.axes([0.2, 0.1, 0.6, 0.03])
    ax_quality.set_facecolor(AXES_COLOR)
    quality_slider = Slider(ax_quality, 'Quality', 1, 100, valinit=50, valfmt='%d',
                           color=WAVE_COLOR, alpha=0.8)
    quality_slider.label.set_color(TEXT_COLOR)
    quality_slider.valtext.set_color(TEXT_COLOR)
    
    # Subsampling slider
    ax_subsampling = plt.axes([0.2, 0.05, 0.6, 0.03])
    ax_subsampling.set_facecolor(AXES_COLOR)
    subsampling_slider = Slider(ax_subsampling, 'Subsampling', 0, 2, 
                               valinit=2, valfmt='%d', 
                               valstep=1, color=WAVE_COLOR, alpha=0.8)
    subsampling_slider.label.set_color(TEXT_COLOR)
    subsampling_slider.valtext.set_color(TEXT_COLOR)
    
    def update_compression(val):
        """Update the compressed image when sliders change."""
        quality = int(quality_slider.val)
        subsampling = int(subsampling_slider.val)
        
        # Get subsampling label
        subsampling_labels = {0: '4:4:4', 1: '4:2:2', 2: '4:2:0'}
        subsampling_label = subsampling_labels.get(subsampling, '4:2:0')
        
        # Compress image with new parameters
        compressed = compress_image_jpeg(original_image, quality, subsampling)
        
        # Update the display
        im2.set_array(np.array(compressed))
        ax2.set_title(f'JPEG Compressed (Quality: {quality}, Subsampling: {subsampling_label})', 
                     color=TEXT_COLOR)
        
        # Calculate file size difference (approximate)
        original_size = len(original_image.tobytes())
        buffer = BytesIO()
        compressed.save(buffer, format='JPEG', quality=quality, subsampling=subsampling)
        compressed_size = len(buffer.getvalue())
        compression_ratio = original_size / compressed_size
        
        # Update title with compression info
        ax2.set_title(f'JPEG Compressed (Quality: {quality}, Subsampling: {subsampling_label})\n'
                     f'Compression Ratio: {compression_ratio:.1f}:1', color=TEXT_COLOR)
        
        fig.canvas.draw()
    
    # Connect sliders to update function
    quality_slider.on_changed(update_compression)
    subsampling_slider.on_changed(update_compression)
    
    # Add image selector buttons
    def switch_image(image_index):
        """Switch to a different image."""
        nonlocal original_image
        image_path = os.path.join(THIS_FOLDER, IMAGES[image_index])
        original_image = Image.open(image_path)
        
        # Update original image display
        im1.set_array(np.array(original_image))
        ax1.set_title(f'Original Image ({os.path.basename(IMAGES[image_index])})', 
                     color=TEXT_COLOR)
        
        # Update compressed image
        update_compression(None)
    
    # Add buttons for image selection with color scheme
    from matplotlib.widgets import Button
    
    ax_button1 = plt.axes([0.05, 0.85, 0.1, 0.04])
    ax_button1.set_facecolor(AXES_COLOR)
    ax_button2 = plt.axes([0.05, 0.8, 0.1, 0.04])
    ax_button2.set_facecolor(AXES_COLOR)
    
    button1 = Button(ax_button1, 'Modi A', color=AXES_COLOR, hovercolor=WAVE_COLOR)
    button1.label.set_color(TEXT_COLOR)
    button2 = Button(ax_button2, 'Modi B', color=AXES_COLOR, hovercolor=WAVE_COLOR)
    button2.label.set_color(TEXT_COLOR)
    
    button1.on_clicked(lambda x: switch_image(0))
    button2.on_clicked(lambda x: switch_image(1))
    
    plt.show()

if __name__ == "__main__":
    create_interactive_jpeg_demo()
