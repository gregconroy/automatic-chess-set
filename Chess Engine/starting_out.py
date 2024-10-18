import tkinter as tk
from PIL import Image, ImageTk
import time

class ChessUI(tk.Tk):
    def __init__(self, bitboard, refresh_rate=240):
        super().__init__()
        self.title("Bitboard Chess")
        
        # Set the initial size
        self.initial_width = 512
        self.initial_height = 512
        self.geometry(f"{self.initial_width}x{self.initial_height}")

        # Center the window on the screen
        self.center_window()

        self.bitboard = bitboard
        self.refresh_rate = refresh_rate  # Set the desired refresh rate
        self.redraw_delay = 100  # Delay in milliseconds for redraw after resizing

        # Create a frame to hold the canvas
        self.frame = tk.Frame(self, bg="black", bd=0, highlightthickness=0)  # Remove borders and highlight thickness
        self.frame.pack(expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(self.frame, bg="black", highlightthickness=0)  # Set canvas background to black and remove highlight thickness
        self.canvas.pack(expand=True)  # Use expand=True to allow resizing

        self.square_size = 64  # Initial square size
        self.piece_images = {}  # Dictionary to store image references
        self.load_images()  # Load images once
        self.bind("<Configure>", self.on_resize)  # Bind resize event

        # Variables for dragging
        self.dragged_piece = None
        self.dragged_piece_position = None
        self.start_x = None
        self.start_y = None

        self.canvas.bind("<Button-1>", self.on_piece_click)  # Left mouse button click
        self.canvas.bind("<B1-Motion>", self.on_piece_drag)  # Mouse drag
        self.canvas.bind("<ButtonRelease-1>", self.on_piece_release)  # Mouse button release

        self.draw_board()
        self.draw_pieces()

        self.last_update_time = time.time()  # Track the last update time using time module
        self.redraw_after_resize = None  # Variable to hold the redraw after resize task

    def center_window(self):
        """Center the window on the screen."""
        self.update_idletasks()  # Update "requested size" from geometry manager
        width = self.winfo_width()
        height = self.winfo_height()

        # Get screen width and height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Calculate x and y coordinates to center the window
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)

        # Set the position of the window
        self.geometry(f"{width}x{height}+{x}+{y}")

    def load_images(self):
        """Load images from file paths."""
        piece_paths = {
            'P': "./Chess Engine/Images/Chess Pieces/white_pawn.png",
            'R': "./Chess Engine/Images/Chess Pieces/white_rook.png",
            'N': "./Chess Engine/Images/Chess Pieces/white_knight.png",
            'B': "./Chess Engine/Images/Chess Pieces/white_bishop.png",
            'Q': "./Chess Engine/Images/Chess Pieces/white_queen.png",
            'K': "./Chess Engine/Images/Chess Pieces/white_king.png",
            'p': "./Chess Engine/Images/Chess Pieces/black_pawn.png",
            'r': "./Chess Engine/Images/Chess Pieces/black_rook.png",
            'n': "./Chess Engine/Images/Chess Pieces/black_knight.png",
            'b': "./Chess Engine/Images/Chess Pieces/black_bishop.png",
            'q': "./Chess Engine/Images/Chess Pieces/black_queen.png",
            'k': "./Chess Engine/Images/Chess Pieces/black_king.png"
        }

        # Load and scale the images for the initial size
        for symbol, path in piece_paths.items():
            image = Image.open(path)
            scaled_image = image.resize((self.square_size, self.square_size), Image.LANCZOS)
            self.piece_images[symbol] = ImageTk.PhotoImage(scaled_image)

    def on_resize(self, event):
        """Handle resizing of the window."""
        if self.redraw_after_resize:
            self.after_cancel(self.redraw_after_resize)  # Cancel the previous redraw task
        self.redraw_after_resize = self.after(self.redraw_delay, self.update_board)  # Schedule a redraw after the delay

    def update_board(self):
        """Update the board and pieces after resizing."""
        new_width = self.frame.winfo_width()  # Use frame width
        new_height = self.frame.winfo_height()  # Use frame height
        
        # Calculate new square size based on the smaller dimension
        new_square_size = min(new_width, new_height) // 8

        # Update the canvas size to fit the new dimensions
        self.canvas.config(width=new_square_size * 8, height=new_square_size * 8)

        # Only redraw if the square size has changed significantly
        if new_square_size != self.square_size:
            self.square_size = new_square_size
            self.canvas.delete("all")  # Clear the canvas
            self.load_images()  # Reload and scale images for the new size
            self.draw_board()  # Redraw the board
            self.draw_pieces()  # Redraw the pieces

    def draw_board(self):
        """Draw the chessboard squares."""
        colors = ["#f0d9b5", "#b58863"]  # Light and dark square colors
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                x1 = col * self.square_size
                y1 = row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                
                # Ensure the rectangle fills the canvas completely
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    def draw_pieces(self):
        """Draw pieces on the board using the bitboards."""
        for row in range(8):
            for col in range(8):
                square_index = (7 - row) * 8 + col
                piece = self.get_piece_at(square_index)
                if piece:
                    x = col * self.square_size
                    y = row * self.square_size
                    self.canvas.create_image(x, y, anchor=tk.NW, image=self.piece_images[piece], tags=piece)

    def get_piece_at(self, square):
        """Get the piece symbol at a specific square based on the bitboards."""
        # Check white pieces
        if (self.bitboard.white_pawns >> square) & 1:
            return 'P'
        if (self.bitboard.white_rooks >> square) & 1:
            return 'R'
        if (self.bitboard.white_knights >> square) & 1:
            return 'N'
        if (self.bitboard.white_bishops >> square) & 1:
            return 'B'
        if (self.bitboard.white_queen >> square) & 1:
            return 'Q'
        if (self.bitboard.white_king >> square) & 1:
            return 'K'
        
        # Check black pieces
        if (self.bitboard.black_pawns >> square) & 1:
            return 'p'
        if (self.bitboard.black_rooks >> square) & 1:
            return 'r'
        if (self.bitboard.black_knights >> square) & 1:
            return 'n'
        if (self.bitboard.black_bishops >> square) & 1:
            return 'b'
        if (self.bitboard.black_queen >> square) & 1:
            return 'q'
        if (self.bitboard.black_king >> square) & 1:
            return 'k'
        
        return None

    def on_piece_click(self, event):
        """Handle piece click event."""
        x, y = event.x, event.y
        col = x // self.square_size
        row = y // self.square_size
        square_index = (7 - row) * 8 + col
        self.dragged_piece = self.get_piece_at(square_index)
        self.dragged_piece_position = square_index

        # Store the initial mouse position
        self.start_x = x
        self.start_y = y

    def on_piece_drag(self, event):
        """Handle piece drag event."""
        if self.dragged_piece:
            x, y = event.x, event.y
            # Move the dragged piece
            self.canvas.delete("dragged_piece")  # Clear previous dragged piece
            self.canvas.create_image(x, y, anchor=tk.CENTER, image=self.piece_images[self.dragged_piece], tags="dragged_piece")

    def on_piece_release(self, event):
        """Handle piece release event."""
        if self.dragged_piece:
            x, y = event.x, event.y
            col = x // self.square_size
            row = y // self.square_size
            new_square_index = (7 - row) * 8 + col
            
            # Update the bitboard here to reflect the movement
            self.bitboard.update_bitboard(self.dragged_piece_position, new_square_index)

            # Clear the old position of the dragged piece
            old_x = (self.dragged_piece_position % 8) * self.square_size
            old_y = (7 - (self.dragged_piece_position // 8)) * self.square_size
            self.canvas.create_rectangle(old_x, old_y, old_x + self.square_size, old_y + self.square_size, fill="", outline="")
            
            # Redraw the board and pieces
            self.canvas.delete("all")  # Clear the canvas
            self.draw_board()  # Redraw the board
            self.draw_pieces()  # Redraw the pieces

            # Reset dragged piece variables
            self.dragged_piece = None
            self.dragged_piece_position = None



class Bitboard:
    def __init__(self):
        # White pieces
        self.white_pawns   = 0x00FF000000000000  # Rank 2 pawns
        self.white_rooks   = 0x8100000000000000  # a1, h1 rooks
        self.white_knights = 0x4200000000000000  # b1, g1 knights
        self.white_bishops = 0x2400000000000000  # c1, f1 bishops
        self.white_queen   = 0x1000000000000000  # d1 queen
        self.white_king    = 0x0800000000000000  # e1 king

        # Black pieces
        self.black_pawns   = 0x000000000000FF00  # Rank 7 pawns
        self.black_rooks   = 0x0000000000000081  # a8, h8 rooks
        self.black_knights = 0x0000000000000042  # b8, g8 knights
        self.black_bishops = 0x0000000000000024  # c8, f8 bishops
        self.black_queen   = 0x0000000000000010  # d8 queen
        self.black_king    = 0x0000000000000008  # e8 king

    def update_bitboard(self, old_square, new_square):
        """Update the bitboard to reflect the movement of a piece."""
        piece = self.get_piece_at(old_square)

        # Determine the bit position
        bit_position_old = 63 - old_square
        bit_position_new = 63 - new_square

        if piece.isupper():  # White piece
            if piece == 'P':
                self.white_pawns &= ~(1 << bit_position_old)  # Clear old position
                self.white_pawns |= (1 << bit_position_new)   # Set new position
            elif piece == 'R':
                self.white_rooks &= ~(1 << bit_position_old)
                self.white_rooks |= (1 << bit_position_new)
            elif piece == 'N':
                self.white_knights &= ~(1 << bit_position_old)
                self.white_knights |= (1 << bit_position_new)
            elif piece == 'B':
                self.white_bishops &= ~(1 << bit_position_old)
                self.white_bishops |= (1 << bit_position_new)
            elif piece == 'Q':
                self.white_queen &= ~(1 << bit_position_old)
                self.white_queen |= (1 << bit_position_new)
            elif piece == 'K':
                self.white_king &= ~(1 << bit_position_old)
                self.white_king |= (1 << bit_position_new)
        
        else:  # Black piece
            if piece == 'p':
                self.black_pawns &= ~(1 << bit_position_old)
                self.black_pawns |= (1 << bit_position_new)
            elif piece == 'r':
                self.black_rooks &= ~(1 << bit_position_old)
                self.black_rooks |= (1 << bit_position_new)
            elif piece == 'n':
                self.black_knights &= ~(1 << bit_position_old)
                self.black_knights |= (1 << bit_position_new)
            elif piece == 'b':
                self.black_bishops &= ~(1 << bit_position_old)
                self.black_bishops |= (1 << bit_position_new)
            elif piece == 'q':
                self.black_queen &= ~(1 << bit_position_old)
                self.black_queen |= (1 << bit_position_new)
            elif piece == 'k':
                self.black_king &= ~(1 << bit_position_old)
                self.black_king |= (1 << bit_position_new)

    def get_piece_at(self, square):
        """Get the piece symbol at a specific square based on the bitboards."""
        # Check white pieces
        if (self.white_pawns >> square) & 1:
            return 'P'
        if (self.white_rooks >> square) & 1:
            return 'R'
        if (self.white_knights >> square) & 1:
            return 'N'
        if (self.white_bishops >> square) & 1:
            return 'B'
        if (self.white_queen >> square) & 1:
            return 'Q'
        if (self.white_king >> square) & 1:
            return 'K'
        
        # Check black pieces
        if (self.black_pawns >> square) & 1:
            return 'p'
        if (self.black_rooks >> square) & 1:
            return 'r'
        if (self.black_knights >> square) & 1:
            return 'n'
        if (self.black_bishops >> square) & 1:
            return 'b'
        if (self.black_queen >> square) & 1:
            return 'q'
        if (self.black_king >> square) & 1:
            return 'k'
        
        return None

if __name__ == "__main__":
    bitboard = Bitboard()  # Initialize the bitboard
    app = ChessUI(bitboard)
    app.mainloop()
