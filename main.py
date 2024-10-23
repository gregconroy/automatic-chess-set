from ChessEngine.chess_engine import ChessEngine
from ChessEngine.chess_ui import ChessUI
from RobotInterface.chess_robot import ChessRobot

def main():
    chess_engine = ChessEngine()
    chess_robot = ChessRobot(port="COM10")
    chess_ui = ChessUI(chess_engine, chess_robot)
    chess_ui.mainloop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
 