import cv2
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import open_clip
from PIL import Image
import torch.nn.functional as F

TEXTS = ["cat", "dog", "bird", "car", "building", "human", "robot"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Color scheme for the visualization
BG_COLOR = (0.1, 0.2, 0.3)        # Dark blue-gray background
TEXT_COLOR = (1, 1, 1)            # White text
WAVE_COLOR = (0.5, 1.0, 0.9)      # Cyan for waveforms and highlights
AXES_COLOR = (0.2, 0.3, 0.4)      # Lighter blue-gray for axes
CUTOFF_COLOR = (1.0, 0.6, 0.0)    # Orange for cutoff line
SPECTROGRAM_CMAP = 'plasma'       # Colormap for spectrogram

def crop_center_square(frame):
    """Crop the frame to a center square"""
    h, w = frame.shape[:2]
    size = min(h, w)
    start_y = (h - size) // 2
    start_x = (w - size) // 2
    return frame[start_y:start_y + size, start_x:start_x + size]

def encode_image_with_clip(image, model, preprocess):
    """Encode an image using CLIP"""
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    # Preprocess and encode
    image_input = preprocess(pil_image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = F.normalize(image_features, p=2, dim=1)
    
    return image_features.cpu().numpy().flatten()

def encode_texts_with_clip(texts, model, tokenizer):
    """Encode text prompts using CLIP"""
    text_tokens = tokenizer(texts).to(DEVICE)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, p=2, dim=1)
    
    return text_features.cpu().numpy()

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def plot_barcode(vector, ax, title="", cmap='viridis'):
    """Plot a vector as a barcode visualization"""
    # Reshape vector to be a 2D array for imshow
    barcode = vector.reshape(1, -1)
    
    ax.imshow(barcode, cmap=cmap, aspect='auto')
    ax.set_title(title, color=TEXT_COLOR)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(BG_COLOR)

def create_visualization(frame, image_vector, text_vectors, text_labels, fig=None, axes=None):
    """Create or update the complete visualization"""
    # Calculate cosine similarities
    similarities = []
    for text_vec in text_vectors:
        sim = cosine_similarity(image_vector, text_vec)
        similarities.append(sim)

    #normalize similarities for better visualization
    similarities = np.array(similarities)
    similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-8)

    
    # Create figure if it doesn't exist
    if fig is None:
        fig = plt.figure(figsize=(12, 12))
        fig.patch.set_facecolor(BG_COLOR)
        
        # Create subplots
        ax1 = plt.subplot(5, 1, 1)
        ax2 = plt.subplot(5, 1, 2)
        ax3 = plt.subplot(5, 1, 3)
        ax4 = plt.subplot(5, 1, 4)
        ax5 = plt.subplot(5, 1, 5)
        axes = [ax1, ax2, ax3, ax4, ax5]
        
        # Set up static elements that don't change
        axes[1].set_title('Image CLIP Encoding', color=TEXT_COLOR, fontsize=12)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].set_facecolor(BG_COLOR)
        
        axes[2].set_title('Text CLIP Encodings', color=TEXT_COLOR, fontsize=12)
        axes[2].set_yticks(range(len(text_labels)))
        axes[2].set_yticklabels(text_labels, color=TEXT_COLOR)
        axes[2].set_xticks([])
        
        axes[3].set_title('Component-wise Products (Image × Text)', color=TEXT_COLOR, fontsize=12)
        axes[3].set_yticks(range(len(text_labels)))
        axes[3].set_yticklabels(text_labels, color=TEXT_COLOR)
        axes[3].set_xticks([])
        
        axes[4].set_title('Cosine Similarities with Image', color=TEXT_COLOR, fontsize=12)
        axes[4].set_ylabel('Similarity', color=TEXT_COLOR)
        axes[4].tick_params(colors=TEXT_COLOR)
        axes[4].set_facecolor(BG_COLOR)
        
        plt.tight_layout()
    
    # Clear axes content for updating
    for ax in axes:
        ax.clear()
    
    # Update frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    axes[0].imshow(frame_rgb)
    axes[0].set_title('Webcam Frame (Center Square)', color=TEXT_COLOR, fontsize=14)
    axes[0].axis('off')
    
    # Update image CLIP vector as barcode
    fixed_range = (-0.1, 0.1)
    barcode = image_vector.reshape(1, -1)
    axes[1].imshow(barcode, cmap='PiYG', aspect='auto', vmin=fixed_range[0], vmax=fixed_range[1])
    axes[1].set_title('Image CLIP Encoding', color=TEXT_COLOR, fontsize=12)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_facecolor(BG_COLOR)
    
    # Update text vectors
    text_matrix = np.stack(text_vectors)
    im = axes[2].imshow(text_matrix, cmap='PiYG', aspect='auto', vmin=fixed_range[0], vmax=fixed_range[1])
    axes[2].set_title('Text CLIP Encodings', color=TEXT_COLOR, fontsize=12)
    axes[2].set_yticks(range(len(text_labels)))
    axes[2].set_yticklabels(text_labels, color=TEXT_COLOR)
    axes[2].set_xticks([])
    
    # Update component-wise products
    product_matrix = []
    for text_vec in text_vectors:
        product = image_vector * text_vec  # Element-wise multiplication
        product_matrix.append(product)
    product_matrix = np.stack(product_matrix)

    im_product = axes[3].imshow(product_matrix, cmap='bone', aspect='auto', vmin=fixed_range[0]*0.2, vmax=fixed_range[1]*0.2)
    axes[3].set_title('Component-wise Products (Image × Text)', color=TEXT_COLOR, fontsize=12)
    axes[3].set_yticks(range(len(text_labels)))
    axes[3].set_yticklabels(text_labels, color=TEXT_COLOR)
    axes[3].set_xticks([])
    
    # Update similarity scores
    bars = axes[4].bar(text_labels, similarities, color=WAVE_COLOR)
    axes[4].set_title('Cosine Similarities with Image', color=TEXT_COLOR, fontsize=12)
    axes[4].set_ylabel('Similarity', color=TEXT_COLOR)
    axes[4].tick_params(colors=TEXT_COLOR)
    axes[4].set_facecolor(BG_COLOR)
    
    # Add value labels on bars
    for bar, sim in zip(bars, similarities):
        height = bar.get_height()
        axes[4].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{sim:.3f}', ha='center', va='bottom', color=TEXT_COLOR)
    
    return fig, axes

def main():
    """Main function to run the webcam CLIP visualization"""
    print("Loading CLIP model...")
    # Same OpenAI ViT-B/32 weights as the original clip package, loaded via open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=DEVICE)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    print(f"Model loaded on {DEVICE}")
    
    # Encode text prompts
    print("Encoding text prompts...")
    text_vectors = encode_texts_with_clip(TEXTS, model, tokenizer)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set webcam properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    print("Starting webcam feed. Press 'q' to quit, 's' to save current visualization.")
    print("Close the matplotlib window to stop the program.")
    
    plt.ion()  # Turn on interactive mode
    
    # Initialize figure and axes - will be created on first frame
    fig = None
    axes = None
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process every few frames to reduce computational load
            frame_count += 1
            if frame_count % 5 != 0:  # Process every 5th frame
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue
            
            # Show original webcam feed
            #cv2.imshow('Webcam Feed (Press Q to quit)', frame)
            
            # Crop to center square
            cropped_frame = crop_center_square(frame)
            
            # Encode with CLIP
            image_vector = encode_image_with_clip(cropped_frame, model, preprocess)
            
            # Create or update visualization
            fig, axes = create_visualization(cropped_frame, image_vector, text_vectors, TEXTS, fig, axes)
            
            # Update the plot
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)  # Small pause to allow updates
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = cv2.getTickCount()
                filename = f'clip_visualization_{timestamp}.png'
                fig.savefig(filename, dpi=300, bbox_inches='tight', 
                           facecolor=BG_COLOR, edgecolor='none')
                print(f"Visualization saved as {filename}")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if fig is not None:
            plt.close(fig)
        plt.ioff()  # Turn off interactive mode
        print("Webcam closed")

if __name__ == "__main__":
    main()