from collections import deque

class ChessEngine:
    def __init__(self):
        self.piece_bitboards = {
            "p": 0x00FF00000000FF00,  # Pawns
            "r": 0x8100000000000081,  # Rooks
            "n": 0x4200000000000042,  # Knights
            "b": 0x2400000000000024,  # Bishops
            "q": 0x1000000000000010,  # Queens
            "k": 0x0800000000000008   # Kings
        }
        
        # Colour bitboards (w: white, b: black)
        self.colour_bitboards = {
            "w": 0x000000000000FFFF,  # All white pieces
            "b": 0xFFFF000000000000   # All black pieces
        }

        self.full_bitboard = 0xFFFF00000000FFFF # All pieces

        self.en_passant_bitboards ={
            'w': 0x0,
            'b': 0x0
        } # For En Passant capturing

        self.castling_bitboards = {
            'w': 0x0,
            'b': 0x0
        }
        self.castling_mask = 0x7600000000000076
        self.castling_attack_mask = 0x3600000000000036
        self.qs_castling_mask = 0xF8000000000000F8
        self.ks_castling_mask = 0x0F0000000000000F
        self.first_row_mask = 0xFF
        self.last_row_mask = 0xFF00000000000000
        self.eligible_castling_pieces = 0x8900000000000089

        self.generate_moves = {
            "p": self.__generate_pawn_moves,
            "r": self.__generate_rook_moves,
            "n": self.__generate_knight_moves,
            "b": self.__generate_bishop_moves,
            "q": self.__generate_queen_moves,
            "k": self.__generate_king_moves,
        }

        self.attack_bitboards = {
            'w': 0x0,
            'b': 0x0
        }

        self.colour_to_move = 'w'

        self.slider_attacks = {
            'r': [0x0]*64,
            'b': [0x0]*64,
            'q': [0x0]*64
        }

        self.saved_board_state = None

        self.king_attacks = [0x0]*64
        self.pawn_attacks = {'w': [0x0] * 64, 'b': [0x0] * 64}  # Initialize attack arrays for both colors
        self.knight_attacks = [0x0]*64
        self.path_masks = [[0x0] * 64 for _ in range(64)]
        self.pin_masks = [0xFFFFFFFFFFFFFFFF] * 64
        self.check_masks = [0xFFFFFFFFFFFFFFFF] * 64 # lower array size later
        self.pin_mask_bitboard = 0x0 # no pieces are pinned
        self.check_mask_bitboard = 0x0 # no pieces are checking

        self.__generate_path_masks()
        self.__generate_slider_attacks()
        self.__generate_king_attacks()
        self.__generate_knight_attacks()
        self.__generate_pawn_attacks()
        self.__update_attack_bitboards()

        self.leaves = []

    def examine_leaves(self):
        for leaf in self.leaves:
            self.piece_bitboards = leaf[0]
            self.colour_bitboards = leaf[1]

            self.draw()

    def set_draw_ref(self, draw):
        self.draw = draw

    def perft(self, depth, board_state=None):
        # print(f"Depth: {depth}")
        if depth == 0:
            # self.leaves.append([self.piece_bitboards.copy(), self.colour_bitboards.copy()])
            return 1

        if board_state:
            self.__load_board_state(board_state)

        count = 0

        colour_bitboard = self.colour_bitboards[self.colour_to_move]

        # print(f"Colour to move: {self.colour_to_move}")

        # self.print_bitboard(colour_bitboard, f"{self.colour_to_move} piece bitboard")

        if colour_bitboard:
            # print(f"Depth: {depth}")

            source_bitboards = self.__get_individual_bitboards(colour_bitboard)

            for source_bitboard in source_bitboards:
                # self.print_bitboard(source_bitboard,  "Source bitboard")

                all_legal_move_bitboard = self.__get_legal_moves(source_bitboard)
                # self.print_bitboard(all_legal_move_bitboard,  "Legal moves")

                if all_legal_move_bitboard:
                    legal_move_bitboards = self.__get_individual_bitboards(all_legal_move_bitboard)

                    for target_bitboard in legal_move_bitboards:
                        # print(f"Colour to move: {self.colour_to_move}")
                        # self.print_bitboard(target_bitboard,  "Target bitboard")
                        # print('Got board state')
                        prev_board_state = self.__get_current_board_state()
                        self.__update_bitboard(source_bitboard, target_bitboard)
                        current_board_state = self.__get_current_board_state()
                        # self.draw()
                        count += self.perft(depth - 1, current_board_state)
                        self.__load_board_state(prev_board_state)
                        # print('Reloaded board state')

        # print("bottom of perft")
        # self.draw()
        return count
    
    def __save_board_state(self, current_board_state):
        """Save the current state of the board and relevant attributes."""
        self.saved_board_state = current_board_state

    def __get_current_board_state(self):
        """Get the current state of the board and relevant attributes."""
        return {
            "piece bitboards": self.piece_bitboards.copy(),
            "colour bitboards": self.colour_bitboards.copy(),
            "full bitboard": self.full_bitboard,
            "en passant bitboards": self.en_passant_bitboards.copy(),
            "castling bitboard": self.castling_bitboards.copy(),
            "eligible castling pieces": self.eligible_castling_pieces,
            "colour to move": self.colour_to_move,
            "attack bitboards": self.attack_bitboards.copy(),
            "pin masks": self.pin_masks,
            "check masks": self.check_masks,
            "pin mask bitboard": self.pin_mask_bitboard,
            "check mask bitboard": self.check_mask_bitboard,
        }

    def __load_board_state(self, board_state):
        board_state = board_state
        """Load the board state from a given dictionary."""
        self.piece_bitboards = board_state["piece bitboards"]
        self.colour_bitboards = board_state["colour bitboards"]
        self.full_bitboard = board_state["full bitboard"]
        self.en_passant_bitboards = board_state["en passant bitboards"]
        self.castling_bitboards = board_state["castling bitboard"]
        self.eligible_castling_pieces = board_state["eligible castling pieces"]
        self.colour_to_move = board_state["colour to move"]
        self.attack_bitboards = board_state["attack bitboards"]
        self.pin_masks = board_state["pin masks"]
        self.check_masks = board_state["check masks"]
        self.pin_mask_bitboard = board_state["pin mask bitboard"]
        self.check_mask_bitboard = board_state["check mask bitboard"]

    def get_piece_at(self, target_square):
        piece = self.__get_piece_at(1 << target_square)
        if piece:
            return ''.join(piece[::-1])
        
        return None
    
    def request_move(self, source_square, destination_square):
        # Create bitboard representations for the source and destination squares
        source_bitboard = 1 << source_square  # From square
        destination_bitboard = 1 << destination_square  # To square

        if not source_bitboard & self.colour_bitboards[self.colour_to_move]:
            print("Not your turn")
            return False
        
        valid_move = self.__is_valid_move(source_bitboard, destination_bitboard)

        if valid_move:
            self.__update_bitboard(source_bitboard, destination_bitboard)
            return True
        
        return False
    
    def get_legal_moves(self, source_square):
        source_bitboard = 1 << source_square  # From square
        
        return self.__get_legal_moves(source_bitboard)
    
    def print_bitboard(self, bitboard, title=None):
        # Initialize an 8x8 grid
        board = [[0 for _ in range(8)] for _ in range(8)]

        # Fill the board with 1s where the piece is present
        for square in range(64):
            if (bitboard >> square) & 1:
                row = 7 - (square // 8)  # Convert to row (0-7)
                col = square % 8          # Column (0-7)
                board[row][col] = 1

        # Print the separator before the board
        print(title if title else "")
        
        # Print the board
        for row in board:
            print(' '.join(str(cell) for cell in row))
        
        # Print the separator after the board
        print()

    def print_piece_board(self, piece):
        """Prints the board representation of a single piece's bitboard as 1s and 0s."""
        if piece is None:
            print("No piece bitboard to print")
            return
        
        piece_type, piece_colour = piece

        piece_bitboard = self.piece_bitboards[piece_type] & self.colour_bitboards[piece_colour]
        
        self.print_bitboard(piece_bitboard)

    def check_for_promotion(self, destination_bitboard, piece_type, piece_colour):
        """Check if a pawn has reached the promotion rank and replace it with a queen."""
        if piece_type == 'p':
            row, col = divmod(self.__get_square_from_bitboard(destination_bitboard), 8)

            # For white pawns, check if it has reached the 8th rank (row 7)
            if piece_colour == 'w' and row == 7:
                self.__promote_pawn(destination_bitboard, 'q', piece_colour)  # Promote to queen
            
            # For black pawns, check if it has reached the 1st rank (row 0)
            elif piece_colour == 'b' and row == 0:
                self.__promote_pawn(destination_bitboard, 'q', piece_colour)  # Promote to queen

    def __promote_pawn(self, destination_bitboard, new_piece_type, piece_colour):
        """Replace the pawn with a new piece (queen in this case) at the destination square."""

        # Clear the pawn from bitboards
        self.piece_bitboards['p'] &= ~destination_bitboard
        self.colour_bitboards[piece_colour] &= ~destination_bitboard
        self.full_bitboard &= ~destination_bitboard

        # Set the new piece (queen) in the appropriate bitboards
        self.piece_bitboards[new_piece_type] |= destination_bitboard
        self.colour_bitboards[piece_colour] |= destination_bitboard
        self.full_bitboard |= destination_bitboard

    def __update_bitboard(self, source_bitboard, destination_bitboard):
        """Update the bitboards for moving a piece."""
        self.colour_to_move = 'b' if self.colour_to_move == 'w' else 'w'

        piece = self.__get_piece_at(source_bitboard)
        captured_piece = self.__get_piece_at(destination_bitboard)
        
        if piece is None:
            return  # No piece at the from_square to move
        
        piece_type, piece_colour = piece

        # Clear from bitboards
        self.piece_bitboards[piece_type] &= ~(source_bitboard)
        self.colour_bitboards[piece_colour] &= ~(source_bitboard)
        self.full_bitboard &= ~(source_bitboard)
        
        # If there is a piece on the destination square, it's a capture
        if captured_piece:
            captured_type, captured_colour = captured_piece
            # Clear the captured piece from both the type and colour bitboards
            self.piece_bitboards[captured_type] &= ~(destination_bitboard)
            self.colour_bitboards[captured_colour] &= ~(destination_bitboard)

        # Set bitboards
        self.piece_bitboards[piece_type] |= (destination_bitboard)    
        self.colour_bitboards[piece_colour] |= (destination_bitboard)
        self.full_bitboard |= (destination_bitboard)

        self.check_for_promotion(destination_bitboard, piece_type, piece_colour)
        self.__update_attack_bitboards()

        self.__check_for_en_passant(source_bitboard, destination_bitboard, piece_type, piece_colour)  
        self.__check_for_castle(destination_bitboard, piece_type, piece_colour)

        self.__update_castling_bitboards()

        self.pin_masks = [0xFFFFFFFFFFFFFFFF] * 64
        self.check_masks = [0xFFFFFFFFFFFFFFFF] * 64 # lower array size later
        self.pin_mask_bitboard = 0x0
        self.check_mask_bitboard = 0x0

        self.__update_knight_check_pin_masks(destination_bitboard, piece_type, piece_colour)
        self.__update_pawn_check_pin_masks(destination_bitboard, piece_type, piece_colour)


        for i in range(64):
            bb = 1<<i
            bb &= self.piece_bitboards['r'] | self.piece_bitboards['b'] | self.piece_bitboards['q']
            if bb:
                self.__update_check_pin_masks(bb)

    def __check_for_en_passant(self, source_bitboard, destination_bitboard, piece_type, piece_colour):
        # Check for en passant
        if piece_type == 'p':  # If the piece is a pawn
            current_square = self.__get_square_from_bitboard(source_bitboard)
            destination_square = self.__get_square_from_bitboard(destination_bitboard)

            # Calculate the move direction
            move_direction = 16 if piece_colour == "w" else -16  # 16 for white, -16 for black

            if destination_square == current_square + move_direction:  # Moved two squares forward
                # Set the en passant target for the opponent
                target_bitboard = 1 << (destination_square - (8 if piece_colour == "w" else -8))
                self.en_passant_bitboards["b" if piece_colour == "w" else "w"] = target_bitboard
            elif destination_bitboard & self.en_passant_bitboards[piece_colour]:  # Check for en passant capture
                en_passant_square = (destination_square - (8 if piece_colour == "w" else -8))
                # Remove the opponent's pawn
                self.colour_bitboards["b" if piece_colour == "w" else "w"] &= ~(1 << en_passant_square)
                self.piece_bitboards['p'] &= ~(1 << en_passant_square)  # Handle both colours
                self.full_bitboard &= ~(1 << en_passant_square)
                
                # Clear the en passant bitboard for both sides after capture
                self.en_passant_bitboards['w'] = 0
                self.en_passant_bitboards['b'] = 0
            else:
                # No move to set en passant target
                self.en_passant_bitboards['w'] = 0
                self.en_passant_bitboards['b'] = 0
        else:
            # Clear both en passant bitboards if it's not a pawn move
            self.en_passant_bitboards['w'] = 0
            self.en_passant_bitboards['b'] = 0

            

    def __check_for_castle(self, destination_bitboard, piece_type, piece_colour):
        if piece_type == 'k':
            if destination_bitboard & self.castling_bitboards[piece_colour]:
                if destination_bitboard & (1 << 1):  
                    rook_source_bitboard = 0x1     
                    rook_destination_bitboard = 0x4  
                elif destination_bitboard & (1 << 5):  
                    rook_source_bitboard = 0x80      
                    rook_destination_bitboard = 0x10  
                elif destination_bitboard & (1 << 57): 
                    rook_source_bitboard = (1 << 56)    
                    rook_destination_bitboard = (1 << 58) 
                else:
                    rook_source_bitboard = (1 << 63)    
                    rook_destination_bitboard = (1 << 60) 

                self.piece_bitboards['r'] &= ~(rook_source_bitboard)    
                self.colour_bitboards[piece_colour] &= ~(rook_source_bitboard)
                self.full_bitboard &= ~(rook_source_bitboard) 

                self.piece_bitboards['r'] |= (rook_destination_bitboard)    
                self.colour_bitboards[piece_colour] |= (rook_destination_bitboard)
                self.full_bitboard |= (rook_destination_bitboard)   

    def __update_castling_bitboards(self):
        # Get the white king and rooks bitboards
        white_king = self.piece_bitboards['k'] & self.colour_bitboards['w']
        white_rooks = self.piece_bitboards['r'] & self.colour_bitboards['w']

        black_king = self.piece_bitboards['k'] & self.colour_bitboards['b']
        black_rooks = self.piece_bitboards['r'] & self.colour_bitboards['b'] 

        self.eligible_castling_pieces &= white_king | white_rooks | black_king | black_rooks

        # Calculate the castling pieces bitboard
        castling_pieces = self.eligible_castling_pieces
        castling_pieces &= (white_king & ~self.attack_bitboards['b'] | white_rooks) | (black_king & ~self.attack_bitboards['w'] | black_rooks)


        # Determine the castling bitboard
        self.castling_bitboards['w'] = (~(self.full_bitboard | self.attack_bitboards['b'] & self.castling_attack_mask) & self.castling_mask) | (castling_pieces) & self.first_row_mask
        self.castling_bitboards['b'] = (~(self.full_bitboard | self.attack_bitboards['w'] & self.castling_attack_mask) & self.castling_mask) | (castling_pieces) & self.last_row_mask

        # Update valid_castles based on the castling masks
        valid_castles = 0  # Clear both king-side and queen-side flags

        # Check king-side castling
        if (self.castling_bitboards['w'] & self.ks_castling_mask) == (self.ks_castling_mask & self.first_row_mask):
            valid_castles |=  0x2
        # Check queen-side castling
        if (self.castling_bitboards['w'] & self.qs_castling_mask) == (self.qs_castling_mask & self.first_row_mask):
            valid_castles |=  0x20

        self.castling_bitboards['w'] = valid_castles

        # Update valid_castles based on the castling masks
        valid_castles = 0  # Clear both king-side and queen-side flags

        # Check king-side castling
        if (self.castling_bitboards['b'] & self.ks_castling_mask) == (self.ks_castling_mask & self.last_row_mask):
            valid_castles |=  0x0200000000000000
        # Check queen-side castling
        if (self.castling_bitboards['b'] & self.qs_castling_mask) == (self.qs_castling_mask & self.last_row_mask):
            valid_castles |=  0x2000000000000000

        self.castling_bitboards['b'] = valid_castles

    def __update_pawn_check_pin_masks(self, piece_bitboard, piece_type, piece_colour):
        if piece_type != 'p':  # Check if the piece is a pawn
            return
        
        # Determine the color of the opposing king
        king_colour = 'b' if piece_colour == 'w' else 'w'
        king_bitboard = self.piece_bitboards['k'] & self.colour_bitboards[king_colour]  # Get the king's bitboard
        piece_square = self.__get_square_from_bitboard(piece_bitboard)  # Get the square index of the pawn
        attack_mask = self.pawn_attacks[piece_colour][piece_square]  # Get the attack mask for the pawn based on its color

        # Check if the attack mask intersects with the king's position
        if attack_mask & king_bitboard:
            # print("King in check")
            self.check_masks[piece_square] = piece_bitboard  # Update check masks
            self.check_mask_bitboard |= 1 << piece_square  # Update the check mask bitboard


    def __update_knight_check_pin_masks(self, piece_bitboard, piece_type, piece_colour):
        if piece_type != 'n':
            return
        
        king_colour = "b" if piece_colour == "w" else "w"
        king_bitboard = self.piece_bitboards['k'] & self.colour_bitboards[king_colour]
        piece_square = self.__get_square_from_bitboard(piece_bitboard)
        attack_mask = self.knight_attacks[piece_square]
        
        if attack_mask & king_bitboard:
            # print("King in check")
            self.check_masks[piece_square] = piece_bitboard
            self.check_mask_bitboard |= 1 << piece_square

    def __update_check_pin_masks(self, source_bitboard):
        pinning_piece = self.__get_piece_at(source_bitboard)
        pinning_type, pinning_colour = pinning_piece
        king_colour = "b" if pinning_colour == "w" else "w"
        king_bitboard = self.piece_bitboards['k'] & self.colour_bitboards[king_colour]
        pinning_square = self.__get_square_from_bitboard(source_bitboard)
        king_square = self.__get_square_from_bitboard(king_bitboard)
        slider_mask = self.slider_attacks[pinning_type][pinning_square]
        path_mask = self.path_masks[pinning_square][king_square] & (slider_mask | source_bitboard)

        if path_mask:
            if (path_mask & ~source_bitboard) & self.colour_bitboards[pinning_colour]:
                # print('Enemy piece(s) alleviating pin')
                pass
            elif path_mask & self.colour_bitboards[king_colour]:
                if (path_mask & self.colour_bitboards[king_colour]).bit_count() == 1:
                    pinned_piece_square = self.__get_square_from_bitboard(path_mask & self.colour_bitboards[king_colour])
                    self.pin_masks[pinned_piece_square] = path_mask
                    # self.print_bitboard(path_mask | source_bitboard)
                    self.pin_mask_bitboard |= 1 << pinned_piece_square
                else:
                    # print('Multiple pieces alleviating pin')
                    pass
            else:
                if slider_mask & king_bitboard:
                    # print("King in check")
                    # self.print_bitboard(source_bitboard | king_bitboard, title="King and Source")
                    # self.print_bitboard(path_mask, title="Path Mask")
                    # self.print_bitboard(slider_mask, title="Slider Mask")
                    self.check_masks[pinning_square] = path_mask
                    self.check_mask_bitboard |= 1 << pinning_square

    def __update_attack_bitboards(self):
        # Clear the attack bitboards for both colours
        self.attack_bitboards['w'] = 0
        self.attack_bitboards['b'] = 0

        # Iterate through all piece types and their corresponding bitboards
        for piece_type in self.piece_bitboards:
            for colour in ['w', 'b']:
                # Get the bitboard for the current piece and colour
                piece_bitboard = self.piece_bitboards[piece_type] & self.colour_bitboards[colour]

                if piece_type == 'p':
                    if colour == 'w':
                        # White pawn attacks
                        left_attack = piece_bitboard << 7  # North-east attack
                        right_attack = piece_bitboard << 9  # North-west attack
                        
                        # Prevent wrapping from the left to right side
                        right_attack &= ~0x0101010101010101  # Remove attacks from the left-most column
                        left_attack &= ~0x8080808080808080  # Remove attacks from the right-most column

                    elif colour == 'b':
                        # Black pawn attacks
                        left_attack = piece_bitboard >> 9  # South-east attack
                        right_attack = piece_bitboard >> 7  # South-west attack
                        
                        # Prevent wrapping from the left to right side
                        left_attack &= ~0x8080808080808080  # Remove attacks from the right-most column
                        right_attack &= ~0x0101010101010101  # Remove attacks from the left-most column

                    # Combine the attacks
                    self.attack_bitboards[colour] |= left_attack | right_attack
                elif piece_type == 'k':
                    king_attacks = self.king_attacks[self.__get_square_from_bitboard(piece_bitboard)]
                    self.attack_bitboards[colour] |= king_attacks
                else:
                    # For each piece that is present, generate its attack moves
                    while piece_bitboard:
                        square = self.__get_square_from_bitboard(piece_bitboard)
                        legal_moves = self.generate_moves[piece_type](1 << square, colour)  # Generate legal moves for the piece

                        # Update the attack bitboard for the current colour
                        self.attack_bitboards[colour] |= legal_moves
                        
                        # Remove the processed piece from the bitboard
                        piece_bitboard &= ~(1 << square)

    def __generate_king_attacks(self):
        """Generate attack bitboards for the king from every square."""
        for square in range(64):
            row, col = divmod(square, 8)
            attack_mask = 0

            # Define all potential king move offsets (relative to the current square)
            king_moves = [
                (-1, -1), (-1, 0), (-1, 1),  # Top row
                (0, -1),         (0, 1),     # Same row (left and right)
                (1, -1), (1, 0), (1, 1)      # Bottom row
            ]

            # Iterate through all possible king moves
            for move in king_moves:
                r, c = row + move[0], col + move[1]
                if 0 <= r < 8 and 0 <= c < 8:  # Ensure the move is within the board
                    attack_mask |= (1 << (r * 8 + c))  # Set bit for the attack square

            self.king_attacks[square] = attack_mask  # Store attack bitboard for this square

    def __generate_pawn_attacks(self):
        """Generate pawn attack bitboards for black and white pawns."""
        # Calculate attacks for white pawns
        for square in range(64):
            row, col = divmod(square, 8)  # Get row and column from square index
            
            if row < 7:  # Ensure we don't go out of bounds for white pawns
                # Left attack (diagonal left)
                if col > 0:
                    self.pawn_attacks['w'][square] |= (1 << ((row + 1) * 8 + (col - 1)))
                
                # Right attack (diagonal right)
                if col < 7:
                    self.pawn_attacks['w'][square] |= (1 << ((row + 1) * 8 + (col + 1)))

        # Calculate attacks for black pawns
        for square in range(64):
            row, col = divmod(square, 8)  # Get row and column from square index
            
            if row > 0:  # Ensure we don't go out of bounds for black pawns
                # Left attack (diagonal left)
                if col > 0:
                    self.pawn_attacks['b'][square] |= (1 << ((row - 1) * 8 + (col - 1)))
                
                # Right attack (diagonal right)
                if col < 7:
                    self.pawn_attacks['b'][square] |= (1 << ((row - 1) * 8 + (col + 1)))

    def __generate_knight_attacks(self):
        """Generate knight attack bitboards for all 64 squares."""        
        for square in range(64):
            row, col = divmod(square, 8)  # Convert square index to row and column
            attacks = 0  # Initialize knight attack bitboard
            
            # Define all possible knight moves (row offset, col offset)
            knight_moves = [
                (-2, -1), (-2, 1),   # 2 up, 1 left or 1 right
                (-1, -2), (-1, 2),   # 1 up, 2 left or 2 right
                (1, -2), (1, 2),     # 1 down, 2 left or 2 right
                (2, -1), (2, 1)      # 2 down, 1 left or 1 right
            ]
            
            # Iterate through all possible knight moves
            for move in knight_moves:
                new_row = row + move[0]
                new_col = col + move[1]
                
                # Ensure the new position is on the board (within bounds 0-7 for both row and col)
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    new_square = new_row * 8 + new_col  # Convert row and column back to square index
                    attacks |= (1 << new_square)  # Set the bit for the valid knight move

            self.knight_attacks[square] = attacks  # Store the attack bitboard for the current square

    def __generate_slider_attacks(self):
        # Loop through each square on the board
        for row in range(8):
            for col in range(8):
                # Initialize the mask for the current square
                rook_mask = 0
                bishop_mask = 0
                
                # Generate the rook's attack mask
                # Vertical and horizontal moves for the rook
                for r in range(8):
                    if r != row:  # Avoid the rook's current position
                        rook_mask |= (1 << (r * 8 + col))  # Vertical moves
                for c in range(8):
                    if c != col:  # Avoid the rook's current position
                        rook_mask |= (1 << (row * 8 + c))  # Horizontal moves

                # Generate the bishop's attack mask
                # Diagonal moves for the bishop
                # Up-right
                r, c = row, col
                while r < 7 and c < 7:  # Check bounds for up-right
                    r += 1
                    c += 1
                    bishop_mask |= (1 << (r * 8 + c))  # Set the bit for the square

                # Up-left
                r, c = row, col
                while r < 7 and c > 0:  # Check bounds for up-left
                    r += 1
                    c -= 1
                    bishop_mask |= (1 << (r * 8 + c))  # Set the bit for the square

                # Down-right
                r, c = row, col
                while r > 0 and c < 7:  # Check bounds for down-right
                    r -= 1
                    c += 1
                    bishop_mask |= (1 << (r * 8 + c))  # Set the bit for the square

                # Down-left
                r, c = row, col
                while r > 0 and c > 0:  # Check bounds for down-left
                    r -= 1
                    c -= 1
                    bishop_mask |= (1 << (r * 8 + c))  # Set the bit for the square

                # Store the masks in the dictionary
                self.slider_attacks['r'][row * 8 + col] = rook_mask
                self.slider_attacks['b'][row * 8 + col] = bishop_mask
                self.slider_attacks['q'][row * 8 + col] = rook_mask | bishop_mask

    def __generate_path_masks(self):
        # Iterate through all possible positions on the chessboard
        for square1 in range(64):  # First square position
            for square2 in range(64):  # Second square position
                if square1 == square2:
                    continue  # Skip if both pieces are on the same square

                row1, col1 = divmod(square1, 8)
                row2, col2 = divmod(square2, 8)
                path_mask = 0  # Initialize the bitboard for the path

                # Check if the squares are aligned
                if row1 == row2:  # Same row
                    path_mask |= (1 << square1)  # Include the first square in the path
                    start_col = min(col1, col2) + 1
                    end_col = max(col1, col2)
                    for col in range(start_col, end_col):
                        path_mask |= (1 << (row1 * 8 + col))  # Set bits in the same row

                elif col1 == col2:  # Same column
                    path_mask |= (1 << square1)  # Include the first square in the path
                    start_row = min(row1, row2) + 1
                    end_row = max(row1, row2)
                    for row in range(start_row, end_row):
                        path_mask |= (1 << (row * 8 + col1))  # Set bits in the same column

                elif abs(row1 - row2) == abs(col1 - col2):  # Same diagonal
                    path_mask |= (1 << square1)  # Include the first square in the path
                    row_step = 1 if row2 > row1 else -1
                    col_step = 1 if col2 > col1 else -1
                    r, c = row1 + row_step, col1 + col_step
                    while (r != row2) and (c != col2):
                        path_mask |= (1 << (r * 8 + c))  # Set bits in the diagonal path
                        r += row_step
                        c += col_step

                # Store the path mask in the 2D array
                self.path_masks[square1][square2] = path_mask



    def __is_valid_move(self, source_bitboard, destiniation_bitboard):
        legal_move_bitboard = self.__get_legal_moves(source_bitboard)
        return legal_move_bitboard and destiniation_bitboard & legal_move_bitboard
    
    def __get_legal_moves(self, source_bitboard):
        piece = self.__get_piece_at(source_bitboard)
        
        if piece is None:
            return None
        
        piece_type, piece_colour = piece

        if self.colour_to_move != piece_colour:
            return None
        
        source_square = self.__get_square_from_bitboard(source_bitboard)
        pinned_mask = self.pin_masks[source_square]
        legal_moves = self.generate_moves[piece_type](source_bitboard, piece_colour)
        legal_moves &= ~self.colour_bitboards[piece_colour]
        legal_moves &= pinned_mask

        # self.print_bitboard(source_bitboard, "Source bitboard")


        if self.check_mask_bitboard:
            checking_bitboards = self.__get_individual_bitboards(self.check_mask_bitboard)
            for checking_bitboard in checking_bitboards:
                # self.print_bitboard(checking_bitboard, "Checking bitboard")

                if piece_type == 'k':
                    attacking_piece = self.__get_piece_at(checking_bitboard)
                    ap_type, _ = attacking_piece
                    if ap_type != 'n' and ap_type != 'p':
                        attacking_square = self.__get_square_from_bitboard(checking_bitboard)
                        if self.__is_same_diag(source_square, attacking_square):
                            slider_mask = self.slider_attacks['b'][self.__get_square_from_bitboard(checking_bitboard)]
                        else:
                            slider_mask = self.slider_attacks['r'][self.__get_square_from_bitboard(checking_bitboard)]
                            
                        legal_moves &= ~slider_mask
                else:
                    if self.check_mask_bitboard.bit_count() > 1:
                        return 0x0

                    check_mask = self.check_masks[self.__get_square_from_bitboard(checking_bitboard)]
                    # self.print_bitboard(check_mask, "Check mask")
                    legal_moves &= check_mask
                    # self.print_bitboard(legal_moves, "Legal moves")

        return legal_moves

    def __generate_pawn_moves(self, source_bitboard, colour):
        """Generates legal pawn moves including forward moves, captures, promotion, and en passant."""
        
        # Find the current square of the pawn
        current_square = self.__get_square_from_bitboard(source_bitboard)
        legal_moves_bitboard = 0
        
        # Determine move direction and starting/promotion ranks based on colour
        if colour == "w":  # White pawns move upwards (to higher ranks)
            move_direction = 8  # Move forward 1 square
            start_rank = 1      # Starting rank for white pawns (rank 2, index 1)
            opponent_colour = "b"
        else:  # Black pawns move downwards (to lower ranks)
            move_direction = -8  # Move forward 1 square
            start_rank = 6       # Starting rank for black pawns (rank 7, index 6)
            opponent_colour = "w"
        
        # Precomputed pawn attack masks for capturing moves
        pawn_attacks = self.pawn_attacks[colour][current_square]

        # Move forward by 1 square
        forward_square = current_square + move_direction
        if 0 <= forward_square < 64:  # Check bounds
            if not self.__is_piece_present(1 << forward_square):  # Square must be empty
                legal_moves_bitboard |= (1 << forward_square)  # Set the bit for this move
        
        # Move forward by 2 squares if on the starting rank
        if current_square // 8 == start_rank:
            double_forward_square = current_square + (2 * move_direction)
            if (0 <= double_forward_square < 64 and
                not self.__is_piece_present(1 << double_forward_square) and
                not self.__is_piece_present(1 << forward_square)):  # Both squares must be empty
                legal_moves_bitboard |= (1 << double_forward_square)  # Set the bit for the double move
                # Set en passant target square (just behind the pawn)
                self.en_passant_bitboards["b" if colour == "w" else "w"] = (1 << forward_square)

        # Use precomputed pawn attack mask for capture moves
        captures_bitboard = pawn_attacks & self.colour_bitboards[opponent_colour]
        legal_moves_bitboard |= captures_bitboard
        
        # Check for en passant capture
        en_passant_capture = pawn_attacks & self.en_passant_bitboards[colour]
        if en_passant_capture:
            self.print_bitboard(en_passant_capture)
            # Check that the en passant target square doesn't contain a pawn of the same colour
            en_passant_target_square = self.__get_square_from_bitboard(en_passant_capture)
            if not (self.colour_bitboards[colour] & (1 << en_passant_target_square)):  # Ensure no own pawn

                legal_moves_bitboard |= en_passant_capture

        return legal_moves_bitboard  # Return the bitboard of legal moves for the pawn


    def __generate_rook_moves(self, source_bitboard, colour):
        # Initialize the bitboard for legal moves
        legal_moves_bitboard = 0

        # Combine both white and black pieces into one bitboard
        all_pieces = self.colour_bitboards['w'] | self.colour_bitboards['b']
        
        # Get the position of the rook
        rook_position = self.__get_square_from_bitboard(source_bitboard)
        
        # Calculate the rank and file of the rook
        rook_rank = rook_position // 8  # Rank (0 to 7)
        rook_file = rook_position % 8   # File (0 to 7)
        
        # Generate horizontal (left and right) moves on the same rank
        # Left (-1)
        for i in range(rook_file - 1, -1, -1):
            target_square = rook_rank * 8 + i
            target_bitboard = 1 << target_square
            if target_bitboard & all_pieces:  # Stop if any piece is encountered
                legal_moves_bitboard |= target_bitboard
                break
            legal_moves_bitboard |= target_bitboard
        
        # Right (+1)
        for i in range(rook_file + 1, 8):
            target_square = rook_rank * 8 + i
            target_bitboard = 1 << target_square
            if target_bitboard & all_pieces:  # Stop if any piece is encountered
                legal_moves_bitboard |= target_bitboard
                break
            legal_moves_bitboard |= target_bitboard
        
        # Generate vertical (up and down) moves on the same file
        # Up (-8)
        for i in range(rook_rank - 1, -1, -1):
            target_square = i * 8 + rook_file
            target_bitboard = 1 << target_square
            if target_bitboard & all_pieces:  # Stop if any piece is encountered
                legal_moves_bitboard |= target_bitboard
                break
            legal_moves_bitboard |= target_bitboard
        
        # Down (+8)
        for i in range(rook_rank + 1, 8):
            target_square = i * 8 + rook_file
            target_bitboard = 1 << target_square
            if target_bitboard & all_pieces:  # Stop if any piece is encountered
                legal_moves_bitboard |= target_bitboard
                break
            legal_moves_bitboard |= target_bitboard
        
        # Return the bitboard of legal moves for the rook
        return legal_moves_bitboard

    def __generate_knight_moves(self, source_bitboard, colour):
        current_square = self.__get_square_from_bitboard(source_bitboard)
        legal_moves_bitboard = self.knight_attacks[current_square]

        return legal_moves_bitboard  # Return the bitboard of legal moves for the knight

    def __generate_bishop_moves(self, source_bitboard, colour):
        # Initialize the bitboard for legal moves
        legal_moves_bitboard = 0

        # Combine both white and black pieces into one bitboard
        all_pieces = self.colour_bitboards['w'] | self.colour_bitboards['b']
        
        # Get the position of the bishop
        bishop_position = self.__get_square_from_bitboard(source_bitboard)
        
        # Calculate the rank and file of the bishop
        bishop_rank = bishop_position // 8  # Rank (0 to 7)
        bishop_file = bishop_position % 8   # File (0 to 7)

        # Diagonal move directions: top-left, top-right, bottom-left, bottom-right
        directions = [(-9, -1), (-7, 1), (7, -1), (9, 1)]
        
        for direction, file_change in directions:
            current_position = bishop_position
            
            # Generate diagonal moves in the current direction
            for _ in range(1, 8):
                current_position += direction
                
                # Stop if the move goes out of bounds
                if current_position < 0 or current_position >= 64:
                    break
                
                current_rank = current_position // 8
                current_file = current_position % 8
                
                # Stop if the move crosses the board edge horizontally
                if abs(current_file - bishop_file) != abs(current_rank - bishop_rank):
                    break
                
                target_bitboard = 1 << current_position
                
                # Stop if we encounter a piece
                if target_bitboard & all_pieces:
                    legal_moves_bitboard |= target_bitboard
                    break
                
                # Add the empty square as a legal move
                legal_moves_bitboard |= target_bitboard

        # Return the bitboard of legal moves for the bishop
        return legal_moves_bitboard


    def __generate_queen_moves(self, source_bitboard, colour):
        # Combine rook and bishop moves for the queen
        return self.__generate_rook_moves(source_bitboard, colour) | self.__generate_bishop_moves(source_bitboard, colour)

    def __generate_king_moves(self, source_bitboard, colour):
        current_square = self.__get_square_from_bitboard(source_bitboard)
        legal_moves_bitboard = self.king_attacks[current_square]

        # Add castling rights if available
        legal_moves_bitboard |= self.castling_bitboards[colour]

        # Exclude squares under attack by opponent
        legal_moves_bitboard &= ~self.attack_bitboards["b" if colour == "w" else "w"]

        return legal_moves_bitboard  # Return the bitboard of legal moves for the king


    def __get_piece_at(self, target_bitboard):
        """Get the piece symbol at a specific square based on the bitboards."""
        for colour in ["w", "b"]:
            if self.colour_bitboards[colour] & target_bitboard:  # Check if the square is occupied by the current colour
                # Check each piece type for the occupied square
                for piece_type, piece_bitboard in self.piece_bitboards.items():
                    if piece_bitboard & target_bitboard:  # Check if the piece type is present at the target square
                        return piece_type, colour  # Return the piece type and colour if found

        return None  # Return None if no piece is found

    def __get_square_from_bitboard(self, bitboard):
        """Get the square index (0-63) from the bitboard."""
        return (bitboard.bit_length() - 1) if bitboard else None

    def __get_individual_bitboards(self, check_mask_bitboard):
        """Extracts individual bitboards with only one '1' bit for each piece checking the king."""
        bitboards = []

        while check_mask_bitboard:
            lsb = check_mask_bitboard & -check_mask_bitboard
            bitboards.append(lsb)
            check_mask_bitboard &= (check_mask_bitboard - 1)
        
        return bitboards

    def __is_piece_present(self, target_bitboard, colour=None):
        """Check if a piece is present on the target square."""
        if colour:
            return target_bitboard & self.colour_bitboards[colour]
        return target_bitboard & self.full_bitboard

    def __is_same_col(self, index1, index2):
        return (index1 & 7) ^ (index2 & 7)
    
    def __is_same_row(self, index1, index2):
        return (index1 >> 3) ^ (index2 >> 3)
    
    def __is_same_diag(self, index1, index2):
        positive_diag = (index1 // 8 - index1 % 8) == (index2 // 8 - index2 % 8)
        negative_diag = (index1 // 8 + index1 % 8) == (index2 // 8 + index2 % 8)
        
        return positive_diag or negative_diag


if __name__ == "__main__":
    chess_engine = ChessEngine()
    for i in range(5):
        print(f"D{i} - Nodes: ", chess_engine.perft(i))