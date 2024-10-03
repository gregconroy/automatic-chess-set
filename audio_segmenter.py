from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os
import numpy as np
import matplotlib.pyplot as plt

# Function to segment audio and save each part as MP3
def segment_audio(file_path, save_dir="segments", min_silence_len=200, silence_thresh=-40, start_padding_ms=20, stop_padding_ms=200, show_plot=False):
    # Extract name, category, and take from the file name
    file_name = os.path.basename(file_path)
    name, category, take = parse_file_name(file_name)
    
    # Load the WAV audio file
    audio = AudioSegment.from_wav(file_path)

    # Detect non-silent parts [(start_ms, end_ms), ...]
    non_silent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    # Add padding to each non-silent segment
    padded_ranges = []
    for start_ms, end_ms in non_silent_ranges:
        start_ms = max(0, start_ms - start_padding_ms)  # Ensure it doesn't go below 0
        end_ms = min(len(audio), end_ms + stop_padding_ms)  # Ensure it doesn't exceed audio length
        padded_ranges.append((start_ms, end_ms))

    # Create directory for saving the segments if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define categories
    pieces = ["king", "queen", "bishop", "knight", "rook", "pawn"]
    numbers = [str(i) for i in range(1, 9)]
    letters = [chr(i) for i in range(ord('a'), ord('h') + 1)]

    # Select the category to work with
    if category == "pieces":
        words = pieces
    elif category == "numbers":
        words = numbers
    elif category == "letters":
        words = letters
    else:
        print(f"Error: Unknown category '{category}'. Choose from 'pieces', 'numbers', or 'letters'.")
        return

    # Ensure we don't have more segments than words
    if len(padded_ranges) > len(words):
        print(f"Warning: More segments detected than expected words in category '{category}'!")
        return

    # Iterate over each non-silent part and export it as a separate MP3 file
    for i, (start_ms, end_ms) in enumerate(padded_ranges):
        # Cut the audio segment
        segment = audio[start_ms:end_ms]
        
        # Save with the format "name_piece_take.mp3"
        word = words[i] if i < len(words) else f"segment_{i+1}"
        output_filename = f"{name}_{word}_{take}.mp3"
        segment.export(os.path.join(save_dir, output_filename), format="mp3")

        print(f"Segment saved: {output_filename}")
    
    # Visualize the audio waveform and non-silent ranges if show_plot is True
    if show_plot:
        visualize_audio(audio, non_silent_ranges, padded_ranges)


# Function to parse the file name and extract name, category, and take
def parse_file_name(file_name):
    # Strip the file extension and split by underscores
    base_name = os.path.splitext(file_name)[0]
    name, category, take_str = base_name.split("_")
    take = int(take_str)  # Convert take to an integer
    return name, category, take


# Function to visualize audio waveform and detected non-silent parts
def visualize_audio(audio, non_silent_ranges, padded_ranges):
    # Convert audio to numpy array for plotting
    samples = np.array(audio.get_array_of_samples())
    time_axis = np.linspace(0, len(audio) / 1000.0, num=len(samples))  # Time axis in seconds

    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, samples, label="Waveform")

    # Highlight the non-silent parts
    for start_ms, end_ms in non_silent_ranges:
        plt.axvspan(start_ms / 1000.0, end_ms / 1000.0, color='yellow', alpha=0.5, label="Non-silent region")

    # Highlight the padded areas
    for start_ms, end_ms in padded_ranges:
        plt.axvspan(start_ms / 1000.0, end_ms / 1000.0, color='lightblue', alpha=0.3, label="Padded region")

    plt.title("Audio Waveform and Detected Non-silent Regions with Padding")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.show()

person="rox"

segment_audio(f"./Audio/{person}_pieces_{1}.wav", show_plot=True)