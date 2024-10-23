from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os
import numpy as np
import matplotlib.pyplot as plt

# Function to segment audio into takes, then words, and save in separate folders
def segment_audio(file_path, name, category, save_dir="Segments", take_duration_ms=16000, min_silence_len=200, silence_thresh=-40, start_padding_ms=20, stop_padding_ms=200, show_plot=False, format='wav'):
    # Extract name and category from the file name
    file_name = os.path.basename(file_path)
    
    # Load the WAV audio file
    audio = AudioSegment.from_wav(file_path)

    # Split the recording into takes based on a fixed duration (take_duration_ms)
    num_takes = len(audio) // take_duration_ms + 1
    takes = [audio[i*take_duration_ms:(i+1)*take_duration_ms] for i in range(num_takes)]
    
    # Create a directory for saving the segments if it doesn't exist
    category_dir = os.path.join(save_dir, category)
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)

    invalid_takes = []

    # Iterate over each take and apply word segmentation
    for take_num, take_audio in enumerate(takes, start=1):  
        take_audio = take_audio[100:-100]
        # Detect non-silent parts [(start_ms, end_ms), ...]
        non_silent_ranges = detect_nonsilent(take_audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
        
        # Check if the take is completely silent and discard it if true
        if not non_silent_ranges:
            return  # Stop processing further takes

        print(f"Processing take {take_num}")

        # Add padding to each non-silent segment
        padded_ranges = []
        for start_ms, end_ms in non_silent_ranges:
            start_ms = max(0, start_ms - start_padding_ms)  # Ensure it doesn't go below 0
            end_ms = min(len(take_audio), end_ms + stop_padding_ms)  # Ensure it doesn't exceed audio length
            padded_ranges.append((start_ms, end_ms))
        
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
        
        invalid = False
        
        # Ensure we don't have more segments than words
        if len(padded_ranges) > len(words):
            print(f"Warning: More segments detected than expected words in category '{category}' for take {take_num}!")
            invalid = True
            invalid_takes.append(take_num)
            visualize_audio(take_audio, non_silent_ranges, padded_ranges)
            # return
        
        if len(padded_ranges) < len(words):
            print(f"Error: Fewer segments detected than expected for category '{category}' in take {take_num}!")
            invalid = True
            invalid_takes.append(take_num)
            visualize_audio(take_audio, non_silent_ranges, padded_ranges)
            # return

        if not invalid:
            # Iterate over each non-silent part and export it as a separate audio file
            for i, (start_ms, end_ms) in enumerate(padded_ranges):
                # Cut the audio segment
                segment = take_audio[start_ms:end_ms]
                
                # Save with the format "name_piece_take.format"
                word = words[i] if i < len(words) else f"segment_{i+1}"
                output_filename = f"{name}_{word}_{take_num}.{format}"
                segment.export(os.path.join(category_dir, output_filename), format=format)

                print(f"Segment saved: {output_filename}")
            
            # Visualize the audio waveform and non-silent ranges if show_plot is True
            if show_plot:
                visualize_audio(take_audio, non_silent_ranges, padded_ranges)

    if invalid:
        print(f"Invalid takes:\n{invalid_takes}")


# Function to parse the file name and extract name and category
def parse_file_name(file_name):
    # Strip the file extension and split by underscores
    base_name = os.path.splitext(file_name)[0]
    name, category = base_name.split("_")
    return name, category


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


# Example usage
person = "greg"
# segment_audio(f"./Audio/{person}_pieces.wav", show_plot=False)
# segment_audio(f"./Audio/{person}_letters.wav", show_plot=False, start_padding_ms=30)
# segment_audio(f"./Audio/{person}_pieces_filtered.wav", person, "pieces", show_plot=False)
# segment_audio(f"./Audio/{person}_letters_filtered.wav", person, "letters", show_plot=False)
segment_audio(f"./Audio/{person}_numbers_filtered.wav", person, "numbers", silence_thresh=-35, show_plot=False)


# person = "roxanne"
# segment_audio(f"./Audio/{person}_pieces.wav", show_plot=False)

# person = "mike"
# segment_audio(f"./Audio/{person}_pieces.wav", show_plot=False)

# person = "dewald"
# segment_audio(f"./Audio/{person}_pieces.wav", show_plot=False)
