import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import os
import argparse

# --- CONFIGURATION ---
# This script now accepts a file path via the command line.
# Example: python analyze.py "/path/to/your/track.wav" --genre pop
# ---------------------


def provide_loudness_feedback(lufs, peak_db):
    """
    Provides interpretive feedback on loudness and peak levels.
    """
    print("\n--- Loudness Feedback ---")

    # LUFS Feedback
    if -16 <= lufs <= -13:
        print(f"  - LUFS ({lufs:.2f}): Excellent. This level is optimal for most streaming services like Spotify and Apple Music.")
    elif lufs > -13:
        print(f"  - LUFS ({lufs:.2f}): Loud. While this will be turned down by most platforms, it may be appropriate for club tracks or certain genres. Be mindful of potential distortion.")
    elif -20 <= lufs < -16:
        print(f"  - LUFS ({lufs:.2f}): Quiet. This is a conservative level. You have headroom to increase the volume for more impact, unless a wide dynamic range is intended.")
    else:
        print(f"  - LUFS ({lufs:.2f}): Very Quiet. Consider increasing the overall level unless you are mastering for a very specific, dynamic medium like film.")

    # Peak Level Feedback
    if peak_db > -1.0:
        print(f"  - Peak ({peak_db:.2f} dBFS): WARNING! Your peak level is very high. This can cause clipping or distortion on many playback systems and streaming platforms. Aim for -1.0 dBFS or lower.")
    elif -2.0 <= peak_db <= -1.0:
        print(f"  - Peak ({peak_db:.2f} dBFS): Good. Your peak is within a safe range for modern streaming services, preventing unwanted distortion from lossy encoding.")
    else:
        print(f"  - Peak ({peak_db:.2f} dBFS): Safe. Your peak level is very conservative. You have room to increase it slightly if you want more punch without risking clipping.")


def provide_tonal_balance_feedback(xf, fft_db, genre):
    """
    Provides feedback on tonal balance based on genre profiles.
    """
    print("\n--- Tonal Balance Feedback ---")

    # Define genre profiles: {genre: {'band_name': [min_freq, max_freq], ...}}
    GENRE_PROFILES = {
        "general": {"Low End": [20, 200], "Mid Range": [200, 2000], "High End": [2000, 20000]},
        "pop": {"Low End": [30, 250], "Mids": [250, 4000], "Highs": [4000, 18000]},
        "rock": {"Bass": [40, 200], "Guitars/Vocals": [200, 5000], "Cymbals/Air": [5000, 16000]},
        "electronic": {"Sub Bass": [20, 100], "Bass/Mid": [100, 1500], "Synth Highs": [1500, 19000]},
        "hiphop": {"Deep Bass": [20, 100], "Vocals/Snares": [100, 3000], "Hi-Hats/Clarity": [3000, 15000]},
        "jazz": {"Upright Bass": [40, 200], "Piano/Horns": [200, 6000], "Brushes/Air": [6000, 20000]}
    }

    profile = GENRE_PROFILES.get(genre, GENRE_PROFILES["general"])

    # Calculate average energy in each band
    band_energies = {}
    for name, (f_min, f_max) in profile.items():
        mask = (xf >= f_min) & (xf < f_max)
        if np.any(mask):
            band_energies[name] = np.mean(fft_db[mask])
        else:
            band_energies[name] = -np.inf # No energy in this band

    # Simple feedback based on relative energies
    # A more sophisticated model could be trained on reference tracks.
    # This is a heuristic approach.
    low_key, mid_key, high_key = list(profile.keys())
    low_energy = band_energies[low_key]
    mid_energy = band_energies[mid_key]
    high_energy = band_energies[high_key]

    print(f"  - Genre Profile: '{genre.capitalize()}'")
    if low_energy > mid_energy + 3 and low_energy > high_energy + 3:
        print("  - Feedback: The low-end is very prominent. This might be intentional for genres like Hip-Hop or Electronic, but could sound muddy in others.")
    elif high_energy > mid_energy + 4:
        print("  - Feedback: The high-end is very bright. This can add 'air' and clarity, but be cautious of it sounding harsh, especially at high volumes.")
    elif mid_energy > low_energy + 2 and mid_energy > high_energy + 2:
        print("  - Feedback: The mid-range is forward. This is great for emphasizing vocals and instruments like guitars, but ensure it doesn't make the mix sound 'boxy'.")
    else:
        print("  - Feedback: The tonal balance appears relatively even. This is a good starting point for a clean mix.")


def detect_clipping(audio_data, threshold=0.999, consecutive_samples=3):
    """
    Detects potential clipping/distortion in the audio data.
    """
    print("\n--- Signal Integrity Analysis ---")

    clipped_frames = np.sum(np.abs(audio_data) >= threshold)

    if clipped_frames > 0:
        # Check for consecutive samples to be more certain of clipping
        is_clipping = False
        for channel in range(audio_data.shape[1] if audio_data.ndim > 1 else 1):
            data_channel = audio_data[:, channel] if audio_data.ndim > 1 else audio_data
            in_clip = False
            count = 0
            for sample in data_channel:
                if abs(sample) >= threshold:
                    if not in_clip:
                        in_clip = True
                    count += 1
                    if count >= consecutive_samples:
                        is_clipping = True
                        break
                else:
                    in_clip = False
                    count = 0
            if is_clipping:
                break

        if is_clipping:
            print(f"  - Clipping: WARNING! Found {clipped_frames} samples at or near 0 dBFS.")
            print(f"    Detected at least one instance of {consecutive_samples} or more consecutive clipped samples.")
            print("    This is a strong indicator of digital distortion. Consider lowering the input gain or using a limiter.")
        else:
            print(f"  - Clipping: OK. Found {clipped_frames} samples near 0 dBFS, but no consecutive clipping was detected. The signal is likely clean.")
    else:
        print("  - Clipping: Excellent. No samples are near the maximum level. The signal is free of digital clipping.")


def analyze_audio_file(filepath, genre="general"):
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

    # --- 2. SIGNAL INTEGRITY ANALYSIS ---
    detect_clipping(data)
    print("-" * 20)

    # --- 3. LOUDNESS & PEAK ANALYSIS ---
    print("\n--- Loudness & Peak Analysis ---")
    meter = pyln.Meter(rate)
    integrated_loudness = meter.integrated_loudness(data)
    loudness_range = meter.loudness_range(data)
    peak_level_db = 20 * np.log10(np.max(np.abs(data)))

    print(f"1. Integrated LUFS: {integrated_loudness:.2f} LUFS")
    print(f"3. Loudness Range (LRA): {loudness_range:.2f} LU")
    print(f"4. Peak Level (Sample Peak): {peak_level_db:.2f} dBFS")

    # --- 3a. Interpretive Loudness Feedback ---
    provide_loudness_feedback(integrated_loudness, peak_level_db)

    print("-" * 20)

    # --- 4. FREQUENCY ANALYSIS ---
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

    # --- 3a. Tonal Balance Feedback ---
    provide_tonal_balance_feedback(xf, fft_magnitude_db, genre)

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
    parser = argparse.ArgumentParser(
        description="Analyze an audio file for music production feedback."
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="The full path to the audio file to analyze."
    )
    parser.add_argument(
        "--genre",
        type=str,
        default="general",
        choices=["general", "pop", "rock", "electronic", "hiphop", "jazz"],
        help="The genre of the music to tailor the analysis (optional)."
    )
    args = parser.parse_args()

    analyze_audio_file(args.filepath, args.genre)
