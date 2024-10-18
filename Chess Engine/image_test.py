import tkinter as tk
from PIL import Image, ImageTk

# Function to scale the image
def load_and_scale_image(image_path, new_size):
    # Open the image using PIL
    image = Image.open(image_path)
    
    # Resize the image
    scaled_image = image.resize(new_size, Image.LANCZOS)  # Use LANCZOS for high-quality downsampling
    
    return ImageTk.PhotoImage(scaled_image)

# Initialize tkinter window
root = tk.Tk()
root.title("Display and Scale PNG Example")

# Load the image file path
image_path = "./Chess Engine/Images/Chess Pieces/white_pawn.png"

# Define new size for the image (width, height)
new_size = (100, 100)  # Change this to your desired size

# Load and scale the image
photo = load_and_scale_image(image_path, new_size)

# Create a canvas widget
canvas = tk.Canvas(root, width=new_size[0], height=new_size[1])
canvas.pack()

# Display the scaled image on the canvas
canvas.create_image(0, 0, anchor=tk.NW, image=photo)

# Run the tkinter main loop
root.mainloop()
