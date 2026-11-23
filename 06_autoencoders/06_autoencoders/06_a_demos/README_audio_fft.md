# Audio FFT Visualizer

An interactive audio visualization tool that demonstrates frequency domain filtering using FFT (Fast Fourier Transform).

## Features

1. **Waveform Display**: Shows both original and filtered audio waveforms
2. **Spectrogram**: Visual representation of the frequency content over time
3. **Interactive Controls**:
   - Frequency cutoff slider (100 Hz to Nyquist frequency)
   - Play Filtered button: Plays audio with frequencies above cutoff removed
   - Play Original button: Plays the unfiltered audio

## How It Works

1. **Load Audio**: The script loads an audio file and computes its Short-Time Fourier Transform (STFT)
2. **Filter**: When you adjust the slider, frequencies above the cutoff are removed from the spectrum
3. **Reconstruct**: The filtered spectrum is converted back to a waveform using Inverse FFT
4. **Play**: The reconstructed audio is played back, demonstrating the effect of frequency filtering

## Controls

- **Cutoff Frequency Slider**: Drag to set the maximum frequency to keep in the audio
- **Play Filtered**: Play the frequency-filtered version of the audio
- **Play Original**: Play the original unfiltered audio for comparison
- **Synchronized Timeline**: All three plots (original waveform, filtered waveform, and spectrogram) share the same time axis
- **Navigation**: Use the matplotlib toolbar to zoom and pan - all plots will zoom together along the time axis

## Requirements

```bash
pip install librosa sounddevice matplotlib numpy
```

## Usage

```bash
python audio_fft.py
```

The visualization will open in a matplotlib window with interactive controls. Experiment with different cutoff frequencies to hear how removing high frequencies affects the audio quality.

## Educational Value

This tool demonstrates:
- How audio signals can be decomposed into frequency components
- The effect of low-pass filtering on audio perception
- The relationship between time-domain waveforms and frequency-domain representations
- Real-time audio processing using FFT

## Technical Details

- Uses librosa for audio loading and STFT computation
- Real-time audio playback via sounddevice
- Interactive matplotlib widgets for controls
- Threaded audio playback to maintain GUI responsiveness
- Synchronized timeline navigation using matplotlib's `sharex` feature
- Consistent time axis across all visualizations for easy correlation