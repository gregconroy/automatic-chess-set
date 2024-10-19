import tkinter as tk
from PIL import Image, ImageTk
import time
# from chess_engine import ChessEngine

class ChessUI(tk.Tk):
    def __init__(self, chess_engine):
        super().__init__()
        self.title("Bitboard Chess")

        self.chess_engine = chess_engine

        self.board_colours = ["#576175", "#A7AFBE"]
        self.legal_colours = ["#8F3D3D", "#D19494"] 
        self.bitboard_colours = {
            "0": ["#576175", "#A7AFBE"],
            "1": ["#993300", "#FF9966"]
        }
        self.background_colour = "#333333"
        self.config(bg=self.background_colour)

        self.selected_bitboards = []

        # Set the initial size
        self.initial_width = 700
        self.initial_height = 512
        self.geometry(f"{self.initial_width}x{self.initial_height}")

        self.redraw_delay = 100 # Delay in milliseconds for redraw after resizing

        # Use grid layout
        self.grid_rowconfigure(0, weight=1) 
        self.grid_columnconfigure(0, weight=1)  
        self.grid_columnconfigure(1, weight=1)  

        # Create a frame to hold the canvas (for the chessboard)
        self.frame = tk.Frame(self, bg=self.background_colour, bd=0, highlightthickness=0)
        self.frame.grid(row=0, column=0, sticky="nsew")  # Fill the left side (sticky="nsew" makes it expand)

        self.canvas = tk.Canvas(self.frame, bg=self.background_colour, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)  # Use expand=True to allow resizing

        self.square_size = 512 // 8  # Initial square size
        self.piece_images = {}  # Dictionary to store image references
        self.center_window()
        self.load_images()  # Load images once
        self.bind("<Configure>", self.on_resize)  # Bind resize event

        # Variables for dragging
        self.dragged_piece = None
        self.dragged_piece_position = None
        self.start_x = None
        self.start_y = None

        self.legal_move_bitboard = None

        self.canvas.bind("<Button-1>", self.on_piece_click)  # Left mouse button click
        self.canvas.bind("<B1-Motion>", self.on_piece_drag)  # Mouse drag
        self.canvas.bind("<ButtonRelease-1>", self.on_piece_release)  # Mouse button release

        self.last_update_time = time.time()  # Track the last update time using time module
        self.redraw_after_resize = None  # Variable to hold the redraw after resize task

        # Checkbox frame for selecting bitboards
        self.checkbox_frame = tk.Frame(self, bg=self.background_colour)  # Match background to main window for a cleaner look
        self.checkbox_frame.grid(row=0, column=1, sticky="nsew", pady=60)  # Add some padding for better spacing

        self.checkbox_vars = {}  # Store checkbox variables for each bitboard

        self.perft_button = tk.Button(
            self.checkbox_frame,
            text="Run Perft",
            command=self.button_click,
            bg="#666666",  # Button background color
            fg="white",    # Button text color
            font=("Helvetica", 10),  # Font settings
            relief=tk.FLAT  # Flat button style
        )
        self.perft_button.pack(pady=10)  # Add some vertical padding

        # Descriptive bitboard labels
        bitboard_descriptions = {
            "f": "Full Board",
            "W": "White Pieces",
            "B": "Black Pieces",
            "p": "Pawns",
            "r": "Rooks",
            "n": "Knights",
            "b": "Bishops",
            "q": "Queen",
            "k": "King",
            "e": "En Passant Squares",
            # "wc": "White Castling",
            # "bc": "Black Castling",
            "wa": "White Attacks",
            "ba": "Black Attacks"
        }

        for bitboard, description in bitboard_descriptions.items():
            var = tk.BooleanVar()
            checkbox = tk.Checkbutton(
                self.checkbox_frame, 
                text=description,  # Use descriptive text
                variable=var, 
                command=self.update_selected_bitboards,
                bg=self.background_colour,  # Match background for a clean look
                fg="white",  # White text for contrast
                font=("Helvetica", 10),  # Simple, clean font
                selectcolor="#666666",  # Use a minimalist selection color
                relief=tk.FLAT  # Flat style for minimalist design
            )
            checkbox.pack(anchor='w', pady=2, expand=True)  # Minimal vertical padding between checkboxes
            self.checkbox_vars[bitboard] = var

        self.chess_engine.set_draw_ref(self.draw)

        self.can_proceed = False
        self.bind('<Return>', self.on_enter)

    def on_enter(self, event):
        self.can_proceed = True

    def button_click(self):
        print(self.chess_engine.perft(5))
        chess_engine.examine_leaves()

    def draw(self):
        self.draw_board()
        self.draw_pieces()
        self.update_idletasks()
        self.update()
        # self.after(10)
        
        while not self.can_proceed:
            self.update_idletasks()
            self.update()
            self.after(100)

        self.can_proceed = False

    def update_selected_bitboards(self):
        """Update the list of selected bitboards based on checkbox state."""
        self.selected_bitboards = [bitboard for bitboard, var in self.checkbox_vars.items() if var.get()]
        self.draw_board()  # Redraw the board with selected bitboards
        self.draw_pieces()

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
            # 'wp': "./Chess Engine/Images/Chess Pieces/white_pawn.png",
            # 'wr': "./Chess Engine/Images/Chess Pieces/white_rook.png",
            # 'wn': "./Chess Engine/Images/Chess Pieces/white_knight.png",
            # 'wb': "./Chess Engine/Images/Chess Pieces/white_bishop.png",
            # 'wq': "./Chess Engine/Images/Chess Pieces/white_queen.png",
            # 'wk': "./Chess Engine/Images/Chess Pieces/white_king.png",
            # 'bp': "./Chess Engine/Images/Chess Pieces/black_pawn.png",
            # 'br': "./Chess Engine/Images/Chess Pieces/black_rook.png",
            # 'bn': "./Chess Engine/Images/Chess Pieces/black_knight.png",
            # 'bb': "./Chess Engine/Images/Chess Pieces/black_bishop.png",
            # 'bq': "./Chess Engine/Images/Chess Pieces/black_queen.png",
            # 'bk': "./Chess Engine/Images/Chess Pieces/black_king.png"
            'wp': "./Images/Chess Pieces/white_pawn.png",
            'wr': "./Images/Chess Pieces/white_rook.png",
            'wn': "./Images/Chess Pieces/white_knight.png",
            'wb': "./Images/Chess Pieces/white_bishop.png",
            'wq': "./Images/Chess Pieces/white_queen.png",
            'wk': "./Images/Chess Pieces/white_king.png",
            'bp': "./Images/Chess Pieces/black_pawn.png",
            'br': "./Images/Chess Pieces/black_rook.png",
            'bn': "./Images/Chess Pieces/black_knight.png",
            'bb': "./Images/Chess Pieces/black_bishop.png",
            'bq': "./Images/Chess Pieces/black_queen.png",
            'bk': "./Images/Chess Pieces/black_king.png"
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
            self.draw_board()
            self.draw_pieces()  # Redraw the pieces

    def draw_board(self):
        """Draw the chessboard squares."""

        if len(self.selected_bitboards) > 0:
            self.draw_bitboard()
            return

        for row in range(8):
            for col in range(8):
                color = self.board_colours[(row + col) % 2]
                square_index = (7 - row) * 8 + col
                x1 = col * self.square_size
                y1 = row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                
                # Ensure the rectangle fills the canvas completely
                square_tag = f"square_{square_index}"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="", tags=square_tag)


    def draw_pieces(self):
        """Draw pieces on the board using the bitboards."""
        self.canvas.delete("piece")  # Clear any previously drawn pieces (optional)
        for row in range(8):
            for col in range(8):
                square_index = (7 - row) * 8 + col
                piece = self.chess_engine.get_piece_at(square_index)
                if piece:
                    x = col * self.square_size
                    y = row * self.square_size
                    
                    # Create a unique tag for each piece
                    piece_tag = f"{piece}_{square_index}"  # e.g., "wp_12"
                    self.canvas.delete(piece_tag)
                    self.canvas.create_image(x, y, anchor=tk.NW, image=self.piece_images[piece], tags=piece_tag)

    def draw_bitboard(self):
        """Display the selected bitboard on the canvas."""
        self.canvas.delete("bitboard_square")

        if len(self.selected_bitboards) == 0:
            return

        self.bitboards = {
            "p": chess_engine.piece_bitboards["p"],
            "r": chess_engine.piece_bitboards["r"],
            "n": chess_engine.piece_bitboards["n"],
            "b": chess_engine.piece_bitboards["b"],
            "q": chess_engine.piece_bitboards["q"],
            "k": chess_engine.piece_bitboards["k"],
            "W": chess_engine.colour_bitboards["w"],
            "B": chess_engine.colour_bitboards["b"],
            "f": chess_engine.full_bitboard,
            "e": chess_engine.en_passant_bitboard,
            "wc": chess_engine.castling_bitboards['w'],
            "bc": chess_engine.castling_bitboards['b'],
            "wa": chess_engine.attack_bitboards["w"],
            "ba": chess_engine.attack_bitboards["b"]
        }


        bitboard = (1 << 64) - 1

        for bitboard_key in self.selected_bitboards:
            bitboard &= self.bitboards[bitboard_key]

        # Iterate through the 64 squares of the chessboard
        for square_index in range(64):
            col = square_index % 8
            row = 7 - (square_index // 8)  # Convert the square index to row (flipped vertically)
            x1 = col * self.square_size
            y1 = row * self.square_size
            x2 = x1 + self.square_size
            y2 = y1 + self.square_size
            if (bitboard >> square_index) & 1:  # Check if the square is occupied (bit is set to 1)
                colour = self.bitboard_colours["1"][(row + col) % 2]
            else:
                colour = self.bitboard_colours["0"][(row + col) % 2]

            self.canvas.create_rectangle(x1, y1, x2, y2, fill=colour, outline="", tags="bitboard_square")

        self.draw_pieces()

    def draw_piece_in_original_position(self, position):
        """Redraw the piece at its original position."""
        col = position % 8
        row = 7 - (position // 8)  # Convert back to row
        x = col * self.square_size
        y = row * self.square_size
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.piece_images[self.dragged_piece], tags=f"{self.dragged_piece}_{position}")
    
    def show_legal_moves(self):
        """Highlight legal moves on the board."""
        # Clear previous legal move indicators
        self.canvas.delete("legal_move")

        if self.legal_move_bitboard is None:
            return  # No legal moves to show

        for square_index in range(64):  # There are 64 squares on the chessboard
            if (self.legal_move_bitboard >> square_index) & 1:  # Check if the square is a legal move
                col = square_index % 8
                row = 7 - (square_index // 8)  # Convert back to row
                colour = self.legal_colours[(row + col) % 2]
                x1 = col * self.square_size
                y1 = row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                
                # Draw a rectangle for the legal move square
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=colour, outline="", tags="legal_move")

        self.draw_pieces()


    def in_board_bounds(self, x, y):
        return 0 <= x <= self.frame.winfo_width() and 0 <= y <= self.frame.winfo_height() 

    def on_piece_click(self, event):
        """Handle piece click event."""
        x, y = event.x, event.y

        self.legal_move_bitboard = None

        if not self.in_board_bounds(x, y):
            print("Mouse click not in bounds")
            return

        col = x // self.square_size
        row = y // self.square_size
        square_index = (7 - row) * 8 + col

        piece = self.chess_engine.get_piece_at(square_index)
        if piece:
            print("Showing legal moves")
            self.legal_move_bitboard = self.chess_engine.get_legal_moves(square_index)
            self.dragged_piece = piece
            self.dragged_piece_position = square_index

        self.show_legal_moves()
        

    def on_piece_drag(self, event):
        """Handle piece drag event."""
        if self.dragged_piece:
            x, y = event.x, event.y

            # Move the dragged piece
            # Clear the previous dragged piece
            self.canvas.delete("dragged_piece")
            
            # Create a unique tag for the dragged piece
            piece_tag = f"{self.dragged_piece}_{self.dragged_piece_position}"  # e.g., "wp_12"
            
            # Now, delete the actual piece image from the canvas
            self.canvas.delete(piece_tag)  # Clear the specific piece from the canvas

            # Draw the dragged piece at the current mouse position
            self.canvas.create_image(x, y, anchor=tk.CENTER, image=self.piece_images[self.dragged_piece], tags="dragged_piece")

    def on_piece_release(self, event):
        """Handle piece release event."""
        if self.dragged_piece:
            x, y = event.x, event.y

            if not self.in_board_bounds(x, y):
                print("Mouse release not in bounds")
                self.draw_piece_in_original_position(self.dragged_piece_position)
                return

            col = x // self.square_size
            row = y // self.square_size
            new_square_index = (7 - row) * 8 + col

            if self.dragged_piece_position == new_square_index:
                self.draw_piece_in_original_position(self.dragged_piece_position)
                self.canvas.delete("dragged_piece")
                self.dragged_piece = None
                self.dragged_piece_position = None
                return

            # Validate the move before updating the bitboard
            if self.chess_engine.request_move(self.dragged_piece_position, new_square_index):
                # Clear the old position of the dragged piece
                old_x = (self.dragged_piece_position % 8) * self.square_size
                old_y = (7 - (self.dragged_piece_position // 8)) * self.square_size
                self.canvas.create_rectangle(old_x, old_y, old_x + self.square_size, old_y + self.square_size, fill="", outline="")
                
                # Redraw the board and pieces
                self.canvas.delete("all")  # Clear the canvas
                self.draw_board()  # Redraw the board
                self.draw_pieces()  # Redraw the pieces
            else:
                print("Invalid move")  # Notify user of invalid move
                self.draw_piece_in_original_position(self.dragged_piece_position)
                self.canvas.delete("dragged_piece")

            # Reset dragged piece variables
            self.dragged_piece = None
            self.dragged_piece_position = None

    def format_piece(self, piece):
        return ''.join(piece[::-1])


if __name__ == "__main__":
    from chess_engine import ChessEngine
    chess_engine = ChessEngine()
    chess_ui = ChessUI(chess_engine)
    chess_ui.mainloop()
