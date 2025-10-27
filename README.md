# Music Production Analyzer

A command-line tool designed to provide music producers and audio engineers with detailed, interpretive feedback on their tracks. Moving beyond simple data readouts, this analyzer offers actionable insights to help you make informed decisions during the mixing and mastering process.

## Key Features

The analyzer provides a comprehensive suite of metrics and feedback, including:

*   **Signal Integrity Analysis:**
    *   Detects potential digital **clipping and distortion**, warning you if your track is running too hot.
*   **Loudness & Peak Analysis:**
    *   Measures **Integrated LUFS**, **Loudness Range (LRA)**, and **Peak Levels**.
    *   Provides **interpretive feedback** on loudness, comparing your track to targets for major streaming services (Spotify, Apple Music, etc.).
*   **Dynamic Range Analysis:**
    *   Calculates **Peak-to-Loudness Ratio (PLR)** to gauge the overall dynamic impact.
    *   Measures the **Crest Factor** to give insight into the "punchiness" of your transients.
*   **Frequency Analysis:**
    *   Generates a **Frequency Spectrum Plot** (`frequency_spectrum.png`) for a visual representation of your track's tonal balance.
    *   Provides **genre-aware tonal balance feedback**, comparing your track's frequency profile to common standards for various genres (Pop, Rock, Electronic, etc.).
*   **Musical Context Analysis:**
    *   Estimates the track's **Tempo (BPM)**.
    *   Estimates the track's **Musical Key**.
*   **Stereo & Mono Analysis:**
    *   Checks for **mono compatibility** with a **Stereo Correlation Meter**, helping you avoid phase issues.
    *   Analyzes the **stereo width of the low-end**, a critical factor for clean, powerful mixes.
    *   Saves a mono version of your track (`mono_version_for_listening.wav`) for direct comparison.

## Installation

To get started, you'll need Python 3. Follow these steps to set up the tool:

1.  **Clone the repository or download the source code.**

2.  **Navigate to the project directory:**
    ```bash
    cd /path/to/music-production-analyzer
    ```

3.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the analysis from your terminal using the following command structure:

```bash
python analyze.py "path/to/your/audio_file.wav" --genre your_genre
```

### Arguments

*   `filepath` (required): The full path to the audio file you want to analyze. Be sure to wrap the path in quotes if it contains spaces.
*   `--genre` (optional): The genre of your track. This helps tailor the tonal balance feedback.
    *   **Choices:** `general`, `pop`, `rock`, `electronic`, `hiphop`, `jazz`
    *   If omitted, the `general` profile will be used.

### Example

```bash
python analyze.py "/Users/me/Music/My Mixes/track_final_mix.wav" --genre electronic
```

## Understanding the Analysis

*   **Integrated LUFS:** Measures the perceived loudness of the track over its entire duration. Streaming services often normalize audio to a target LUFS level (e.g., -14 LUFS for Spotify).
*   **Peak-to-Loudness Ratio (PLR):** The difference between the peak level and the integrated loudness. A higher PLR generally means a more dynamic and punchy track, while a lower PLR indicates a more compressed, dense sound.
*   **Crest Factor:** The ratio of the peak amplitude to the RMS (average) level. It's a great indicator of how "spiky" a waveform is. High crest factors are common in tracks with sharp transients (like snare drums), while lower values suggest more sustained sounds.
*   **Stereo Correlation:** Measures the phase relationship between the left and right channels.
    *   A value close to **+1** indicates a signal that is very similar in both channels (essentially mono or centered).
    *   A value close to **0** indicates a wide stereo image.
    *   A value close to **-1** indicates significant out-of-phase content, which can cause elements of your mix to disappear when played in mono.
