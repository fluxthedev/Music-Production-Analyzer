import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
# IMPORTANT: Replace this with the full path to your audio file!
# Example on Mac/Linux: "/Users/yourname/Music/my_track.wav"
# Example on Windows: "C:/Users/yourname/Music/my_track.wav"
AUDIO_FILE_PATH = "path/to/your/track.wav" 
# ---------------------


def analyze_audio_file(filepath):
    """
    Analyzes an audio file to extract loudness, peak, frequency, and stereo data.
    """
    if not os.path.exists(filepath):
        print(f"--- ERROR: File not found at '{filepath}' ---")
        print("Please update the AUDIO_FILE_PATH variable in the script and try again.")
        return

    # --- 1. Load Audio & Basic Info ---
    print(f"Analyzing: {os.path.basename(filepath)}\n")
    data, rate = sf.read(filepath)
    data = data.astype(np.float32)
    
    if data.ndim > 1:
        num_channels, num_samples = data.shape[1], data.shape[0]
        is_stereo = num_channels == 2
    else:
        num_channels, num_samples = 1, data.shape[0]
        is_stereo = False

    print("--- Basic Info ---")
    print(f"Sample Rate: {rate} Hz")
    print(f"Duration: {num_samples / rate:.2f} seconds")
    print(f"Channels: {num_channels}")
    print("-" * 20)

    # --- 2. LOUDNESS & PEAK ANALYSIS ---
    print("\n--- Loudness & Peak Analysis ---")
    meter = pyln.Meter(rate) 
    integrated_loudness = meter.integrated_loudness(data)
    loudness_range = meter.loudness_range(data)
    peak_level_db = 20 * np.log10(np.max(np.abs(data)))

    print(f"1. Integrated LUFS: {integrated_loudness:.2f} LUFS")
    print(f"3. Loudness Range (LRA): {loudness_range:.2f} LU")
    print(f"4. Peak Level (Sample Peak): {peak_level_db:.2f} dBFS")
    print("-" * 20)

    # --- 3. FREQUENCY ANALYSIS ---
    print("\n--- Frequency Analysis ---")
    mono_data = data.mean(axis=1) if is_stereo else data
    N = len(mono_data)
    yf = fft(mono_data)
    xf = fftfreq(N, 1 / rate)[:N//2]
    fft_magnitude_db = 20 * np.log10(2.0/N * np.abs(yf[0:N//2]))

    target_freqs = {'Sub Bass': 40, 'Low-Mid/Mud': 250, 'Presence': 2000, 'Air': 12000}
    print("5-8. Levels at Key Frequencies:")
    for name, freq in target_freqs.items():
        idx = (np.abs(xf - freq)).argmin()
        print(f"     - {name} (~{xf[idx]:.0f}Hz): {fft_magnitude_db[idx]:.2f} dB")

    plt.figure(figsize=(12, 6))
    plt.plot(xf, fft_magnitude_db)
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.title('9. Overall Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.xlim(20, 20000)
    plt.ylim(np.min(fft_magnitude_db[xf>20]) - 5, np.max(fft_magnitude_db) + 5)
    spectrum_filename = "frequency_spectrum.png"
    plt.savefig(spectrum_filename)
    print(f"\n9. Overall Shape: A plot has been saved as '{spectrum_filename}'.")
    print("-" * 20)

    # --- 4. STEREO & MONO ANALYSIS ---
    print("\n--- Stereo & Mono Analysis ---")
    if not is_stereo:
        print("Track is mono. Skipping stereo analysis.")
    else:
        mono_filepath = "mono_version_for_listening.wav"
        sf.write(mono_filepath, mono_data, rate)
        print(f"10. Mono Compatibility: A mono version of your track was saved as '{mono_filepath}'.")
        print("    Listen to this file and compare it to the original.")

        left, right = data[:, 0], data[:, 1]
        mid, side = (left + right) / 2, (left - right) / 2
        mid_fft, side_fft = fft(mid), fft(side)
        low_end_mask = xf < 150
        mid_low_end_energy = np.sum(np.abs(mid_fft[low_end_mask])**2)
        side_low_end_energy = np.sum(np.abs(side_fft[low_end_mask])**2)
        side_to_mid_ratio = (side_low_end_energy / mid_low_end_energy) * 100 if mid_low_end_energy > 0 else float('inf')

        print(f"\n11. Low-End Stereo Check (below 150Hz):")
        print(f"    Side channel energy is ~{side_to_mid_ratio:.2f}% of the Mid channel energy.")
    
    print("-" * 20)
    print("\nAnalysis complete.")


if __name__ == "__main__":
    analyze_audio_file(AUDIO_FILE_PATH)
