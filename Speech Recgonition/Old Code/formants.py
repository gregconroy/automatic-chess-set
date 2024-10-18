import librosa
import numpy as np
import matplotlib.pyplot as plt

def lpc(signal, order):
    """ Calculate Linear Predictive Coding coefficients. """
    n = len(signal)
    
    # Check if the signal length is sufficient for the LPC order
    if n < order + 1:
        raise ValueError(f"Signal length {n} is shorter than order {order + 1} required for LPC.")

    # Autocorrelation
    r = np.correlate(signal, signal, mode='full')[n-1:]  # Auto-correlation
    r = r[:order + 1]  # Take only the first 'order' values

    # Initialize the Toeplitz matrix
    R = np.zeros((order + 1, order + 1))
    
    # Populate the Toeplitz matrix
    for i in range(order + 1):
        R[i, :order + 1 - i] = r[i:order + 1]

    # Print shapes for debugging
    print(f"Shape of R: {R.shape}")
    print(f"Shape of r: {r.shape}")
    print(f"Contents of r: {r}")

    # Solve the Toeplitz system for the LPC coefficients
    try:
        a = np.linalg.solve(R, r)  # Use r directly instead of r[1:order + 1]
    except np.linalg.LinAlgError as e:
        print(f"Error in LPC calculation: {e}")
        return None

    return a

def extract_formants(y, sr):
    # Use LPC to estimate the formants
    order = 12
    # Calculate LPC coefficients
    a = lpc(y, order)
    
    if a is None:
        return []

    # Find roots of the polynomial
    r = np.roots(a)

    # Get the formants (roots) that are within the unit circle
    formants = r[np.abs(r) < 1]

    # Calculate the angles (in Hz) and sort them
    formant_freqs = np.angle(formants) * (sr / (2 * np.pi))
    formant_freqs = np.sort(np.abs(formant_freqs))

    return formant_freqs[:5]  # Return the first three formants

def visualize_formants(formants, title="Formant Frequencies"):
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(formants) + 1), formants, color='skyblue')
    plt.xticks(range(1, len(formants) + 1), [f'F{i}' for i in range(1, len(formants) + 1)])
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.grid()
    plt.show()

def process_audio_file(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path)
    
    # Extract formants
    formants = extract_formants(y, sr)
    
    # Visualize formants
    if formants.size > 0:  # Check if formants array is not empty
        visualize_formants(formants, title=f"Formants for {file_path}")
    else:
        print("No formants extracted.")

# Example usage:
mp3_file_path = 'Segments/letters/greg_a_1.mp3'  # Replace with your MP3 file path
process_audio_file(mp3_file_path)
