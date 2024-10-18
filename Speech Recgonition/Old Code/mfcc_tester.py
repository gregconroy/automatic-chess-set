import os
import numpy as np
import matplotlib.pyplot as plt
from mfcc_class import MFCCExtractor  # Import your MFCCExtractor class

# Function to extract MFCCs from specific audio files for chess pieces
def extract_specific_mfccs(directory, pieces, person, sample_num):
    mfcc_list = []
    file_names = []

    for piece in pieces:
        file_name = f'{person}_{piece}_{sample_num}.wav'  # Use external variables for name and sample number
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):  # Check if the file exists
            mfcc_features = MFCCExtractor.extract(file_path)
            mfcc_list.append(mfcc_features)
            file_names.append(piece.capitalize())  # Store the piece name

    return mfcc_list, file_names  # Return MFCCs and piece names

# Function to visualize MFCCs for all pieces together and save the plot
def visualize_all_mfccs(mfcc_list, file_names, person, save_path):
    """Visualize MFCCs for all pieces in a 3x2 grid layout with a title, and save the plot."""
    plt.figure(figsize=(15, 10))
    
    # Create a 3x2 grid for the subplots
    rows = 3
    cols = 2

    # Add a main title
    plt.suptitle(f'MFCC Visualizations for Chess Pieces\nby {person.capitalize()}', fontsize=16)

    for i, mfcc in enumerate(mfcc_list):
        plt.subplot(rows, cols, i + 1)  # Adjust for a 3x2 grid
        plt.imshow(mfcc, aspect='auto', origin='lower', cmap='viridis')
        plt.title(f'MFCC for {file_names[i]}')
        plt.xlabel('Frame Index')
        plt.ylabel('MFCC Coefficient Index')
        plt.colorbar(label='Amplitude')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust spacing to prevent overlap and make room for the title
    
    # Save the plot to the specified path
    plt.savefig(os.path.join(save_path, f'MFCC_Visualizations_{person}.png'), bbox_inches='tight')
    plt.close()  # Close the plot to free up memory

# Main execution
directory_path = './Segments/pieces/'  # Directory containing piece audio files
pieces = ['king', 'queen', 'bishop', 'knight', 'rook', 'pawn']  # List of chess pieces
person_name = "greg"  # External variable for the person's name
sample_number = 1  # External variable for the sample number
save_directory = './Plots/'  # Directory to save the plot

# Create the save directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Extract MFCCs for specific pieces using external variables
mfcc_list, file_names = extract_specific_mfccs(directory_path, pieces, person_name, sample_number)

# Visualize all MFCCs and save the plot
visualize_all_mfccs(mfcc_list, file_names, person_name, save_directory)  # Visualize all MFCCs
