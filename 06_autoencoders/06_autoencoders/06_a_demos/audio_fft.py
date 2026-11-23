import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.widgets import Button, Slider
import sounddevice as sd
import threading
import os
import time

# Color scheme for the visualization
BG_COLOR = (0.1, 0.2, 0.3)        # Dark blue-gray background
TEXT_COLOR = (1, 1, 1)            # White text
WAVE_COLOR = (0.5, 1.0, 0.9)      # Cyan for waveforms and highlights
AXES_COLOR = (0.2, 0.3, 0.4)      # Lighter blue-gray for axes
CUTOFF_COLOR = (1.0, 0.6, 0.0)    # Orange for cutoff line
SPECTROGRAM_CMAP = 'plasma'       # Colormap for spectrogram

class AudioFFTVisualizer:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        
        # Load audio file
        print(f"Loading audio file: {audio_path}")
        self.y, self.sr = librosa.load(audio_path, sr=None)
        self.duration = len(self.y) / self.sr
        
        # Initialize parameters
        self.cutoff_freq = self.sr // 4  # Default to quarter of sample rate
        self.is_playing = False
        self.play_thread = None
        self.current_audio_type = 'filtered'
        
        # Create time axis for synchronization
        self.time_axis = np.linspace(0, self.duration, len(self.y))
        
        # Compute spectrogram
        self.stft = librosa.stft(self.y, n_fft=2048, hop_length=512)
        self.magnitude = np.abs(self.stft)
        self.phase = np.angle(self.stft)
        
        # Setup the figure and axes
        self.setup_figure()
        
        print(f"Audio loaded: {self.duration:.2f} seconds, {self.sr} Hz sample rate")
        
    def setup_synchronized_navigation(self):
        """Set up synchronized navigation across all time-based plots"""
        # Set initial time limits for all axes
        time_limits = (0, self.duration)
        
        self.ax_wave_orig.set_xlim(time_limits)
        self.ax_wave_filt.set_xlim(time_limits)
        self.ax_spec.set_xlim(time_limits)
        
        # Enable navigation toolbar for better zoom/pan controls
        # The sharex parameter already handles the synchronization
        
        # Add vertical cursor lines for time reference
        self.cursor_lines = []
        
    def setup_figure(self):
        """Initialize the matplotlib figure with waveform, spectrogram, and controls"""
        self.fig = plt.figure(figsize=(14, 12), facecolor=BG_COLOR)
        
        # Create subplots with better spacing
        gs = self.fig.add_gridspec(4, 1, height_ratios=[1, 1, 1.5, 0.8], 
                                  hspace=0.4, bottom=0.15, top=0.95)
        
        # Original waveform (master axis for synchronization)
        self.ax_wave_orig = self.fig.add_subplot(gs[0])
        self.plot_waveform(self.y, self.ax_wave_orig, "Original Waveform")
        
        # Filtered waveform (synchronized with original)
        self.ax_wave_filt = self.fig.add_subplot(gs[1], sharex=self.ax_wave_orig)
        self.plot_waveform(self.y, self.ax_wave_filt, "Filtered Waveform")
        
        # Spectrogram (synchronized with waveforms)
        self.ax_spec = self.fig.add_subplot(gs[2], sharex=self.ax_wave_orig)
        self.plot_spectrogram()
        
        # Controls area
        self.ax_controls = self.fig.add_subplot(gs[3])
        self.setup_controls()
        
        # Set up synchronized navigation
        self.setup_synchronized_navigation()
        
        # Mark initial setup as complete to preserve zoom levels on updates
        self._initial_setup_done = True
        
    def plot_waveform(self, audio_data, ax, title):
        """Plot waveform on given axis with consistent time axis"""
        ax.clear()
        
        # Use consistent time axis for all waveforms
        if len(audio_data) == len(self.y):
            # Full-length audio (original)
            time_vals = self.time_axis
        else:
            # Filtered audio might have slightly different length due to STFT processing
            time_vals = np.linspace(0, len(audio_data) / self.sr, len(audio_data))
        
        ax.plot(time_vals, audio_data, color=WAVE_COLOR, alpha=0.7, linewidth=0.8)
        ax.set_title(title, color=TEXT_COLOR)
        ax.set_ylabel('Amplitude', color=TEXT_COLOR)
        ax.grid(True, alpha=0.3, color=AXES_COLOR)
        
        # Set axis colors
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.spines['bottom'].set_color(AXES_COLOR)
        ax.spines['top'].set_color(AXES_COLOR)
        ax.spines['right'].set_color(AXES_COLOR)
        ax.spines['left'].set_color(AXES_COLOR)
        
        # Set consistent vertical scale for both waveforms
        ax.set_ylim(-1.0, 1.0)
        
        # Only show x-axis label on the bottom waveform to reduce clutter
        if hasattr(self, 'ax_wave_filt') and ax == self.ax_wave_filt:
            ax.set_xlabel('Time (s)')
        else:
            ax.set_xlabel('')
        
    def plot_spectrogram(self):
        """Plot spectrogram with cutoff frequency line"""
        self.ax_spec.clear()
        
        # Convert to dB scale
        magnitude_db = librosa.amplitude_to_db(self.magnitude, ref=np.max)
        
        # Display spectrogram with explicit time axis
        img = librosa.display.specshow(
            magnitude_db, 
            sr=self.sr, 
            hop_length=512,
            x_axis='time', 
            y_axis='hz',
            ax=self.ax_spec,
            cmap=SPECTROGRAM_CMAP
        )
        
        # Only set initial time limits if not already set (preserve zoom)
        if not hasattr(self, '_initial_setup_done'):
            self.ax_spec.set_xlim(0, self.duration)
        
        # Add cutoff frequency line
        self.ax_spec.axhline(y=self.cutoff_freq, color=CUTOFF_COLOR, linestyle='--', 
                           linewidth=2, label=f'Cutoff: {self.cutoff_freq:.0f} Hz')
        
        self.ax_spec.set_title(f'Spectrogram (Cutoff: {self.cutoff_freq:.0f} Hz)', color=TEXT_COLOR)
        self.ax_spec.set_xlabel('Time (s)', color=TEXT_COLOR)
        self.ax_spec.set_ylabel('Frequency (Hz)', color=TEXT_COLOR)
        
        # Set spectrogram axis colors
        self.ax_spec.set_facecolor(BG_COLOR)
        self.ax_spec.tick_params(colors=TEXT_COLOR)
        self.ax_spec.spines['bottom'].set_color(AXES_COLOR)
        self.ax_spec.spines['top'].set_color(AXES_COLOR)
        self.ax_spec.spines['right'].set_color(AXES_COLOR)
        self.ax_spec.spines['left'].set_color(AXES_COLOR)

        #set the legend to the bottom
        #self.ax_spec.legend(loc='lower right')

        
        # Add colorbar only once
        # if not hasattr(self, 'colorbar'):
        #     self.colorbar = plt.colorbar(img, ax=self.ax_spec)
        #     self.colorbar.set_label('Magnitude (dB)')
        
        # Set frequency limits to show meaningful range
        self.ax_spec.set_ylim(0, min(8000, self.sr // 2))  # Show up to 8kHz or Nyquist
    
    def setup_controls(self):
        """Setup interactive controls (button and slider)"""
        self.ax_controls.clear()
        self.ax_controls.set_xlim(0, 10)
        self.ax_controls.set_ylim(0, 3)
        self.ax_controls.axis('off')
        
        # Play button
        self.ax_button = plt.axes([0.35, 0.05, 0.15, 0.04])
        self.play_button = Button(self.ax_button, 'Play Filtered')
        self.play_button.on_clicked(self.play_filtered_audio)
        
        # Play original button
        self.ax_button_orig = plt.axes([0.52, 0.05, 0.15, 0.04])
        self.play_orig_button = Button(self.ax_button_orig, 'Play Original')
        self.play_orig_button.on_clicked(self.play_original_audio)
        
        # Frequency cutoff slider
        self.ax_slider = plt.axes([0.2, 0.12, 0.6, 0.03])
        max_freq = self.sr // 2  # Nyquist frequency
        self.freq_slider = Slider(
            self.ax_slider, 
            'Cutoff Freq (Hz)', 
            100, 
            max_freq,
            valinit=self.cutoff_freq,
            valfmt='%0.0f Hz'
        )
        self.freq_slider.on_changed(self.update_cutoff_frequency)
        
        # Status text and instructions
        self.ax_controls.text(2.5, 2.5, f'Sample Rate: {self.sr} Hz', 
                            ha='center', va='center', fontsize=9, color=TEXT_COLOR)
        self.ax_controls.text(7.5, 2.5, f'Duration: {self.duration:.2f} seconds', 
                            ha='center', va='center', fontsize=9, color=TEXT_COLOR)
        self.ax_controls.text(5, 2, 'Use toolbar to zoom/pan timeline (all plots synchronized)', 
                            ha='center', va='center', fontsize=9, style='italic', color=WAVE_COLOR)
        self.status_text = self.ax_controls.text(5, 1.2, 'Ready to play', 
                                               ha='center', va='center', fontsize=10, 
                                               color=WAVE_COLOR)
        
        # Set controls area background
        self.ax_controls.set_facecolor(BG_COLOR)
        
    def update_cutoff_frequency(self, val):
        """Update cutoff frequency and replot spectrogram"""
        # Store current zoom level before updating
        current_xlim = self.ax_wave_orig.get_xlim()
        current_ylim_spec = self.ax_spec.get_ylim()
        
        self.cutoff_freq = val
        self.plot_spectrogram()
        
        # Update filtered waveform
        filtered_audio = self.apply_frequency_filter()
        self.plot_waveform(filtered_audio, self.ax_wave_filt, 
                          f"Filtered Waveform (cutoff: {self.cutoff_freq:.0f} Hz)")
        
        # Restore zoom level after updating
        self.ax_wave_orig.set_xlim(current_xlim)
        self.ax_wave_filt.set_xlim(current_xlim)
        self.ax_spec.set_xlim(current_xlim)
        self.ax_spec.set_ylim(current_ylim_spec)
        
        self.fig.canvas.draw()
        
    def apply_frequency_filter(self):
        """Apply frequency filter using FFT"""
        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)
        
        # Create copy of STFT
        filtered_stft = self.stft.copy()
        
        # Find frequency bins above cutoff
        cutoff_bin = np.where(freqs > self.cutoff_freq)[0]
        
        # Zero out frequencies above cutoff
        if len(cutoff_bin) > 0:
            filtered_stft[cutoff_bin[0]:, :] = 0
        
        # Convert back to time domain
        filtered_audio = librosa.istft(filtered_stft, hop_length=512)
        
        return filtered_audio
        
    def play_filtered_audio(self, event):
        """Play the filtered audio"""
        if self.is_playing:
            self.stop_playback()
            return
            
        # Apply frequency filter
        filtered_audio = self.apply_frequency_filter()
        
        # Update status
        self.status_text.set_text('Playing filtered audio...')
        self.status_text.set_color(TEXT_COLOR)
        self.play_button.label.set_text('Stop')
        self.fig.canvas.draw()
        
        # Play audio in separate thread
        self.is_playing = True
        self.play_thread = threading.Thread(
            target=self._play_audio_thread, 
            args=(filtered_audio, 'filtered')
        )
        self.play_thread.start()
        
    def play_original_audio(self, event):
        """Play the original audio"""
        if self.is_playing:
            self.stop_playback()
            return
            
        # Update status
        self.status_text.set_text('Playing original audio...')
        self.status_text.set_color(TEXT_COLOR)
        self.play_orig_button.label.set_text('Stop')
        self.fig.canvas.draw()
        
        # Play audio in separate thread
        self.is_playing = True
        self.play_thread = threading.Thread(
            target=self._play_audio_thread, 
            args=(self.y, 'original')
        )
        self.play_thread.start()
        
    def _play_audio_thread(self, audio_data, audio_type='filtered'):
        """Thread function to play audio"""
        try:
            # Normalize audio to prevent clipping
            audio_normalized = audio_data / (np.max(np.abs(audio_data)) + 1e-6)
            
            # Play audio
            sd.play(audio_normalized, samplerate=self.sr)
            sd.wait()  # Wait for playback to complete
            
        except Exception as e:
            print(f"Error playing audio: {e}")
        finally:
            self.is_playing = False
            self.current_audio_type = audio_type
            # Update UI in main thread
            self.fig.canvas.mpl_connect('draw_event', self._reset_play_button)
            self.fig.canvas.draw()
            
    def _reset_play_button(self, event):
        """Reset play button after playback"""
        if not self.is_playing:
            self.status_text.set_text('Ready to play')
            self.status_text.set_color(WAVE_COLOR)
            self.play_button.label.set_text('Play Filtered')
            self.play_orig_button.label.set_text('Play Original')
            
    def stop_playback(self):
        """Stop audio playback"""
        if self.is_playing:
            sd.stop()
            self.is_playing = False
            
    def show(self):
        """Display the visualization"""
        plt.show()

# Main execution
if __name__ == "__main__":
    this_folder = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(this_folder, 'data/no_country_bardem.mp3')
    
    try:
        visualizer = AudioFFTVisualizer(audio_path)
        visualizer.show()
    except Exception as e:
        print(f"Error: {e}")


