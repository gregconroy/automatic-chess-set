import serial
import time

class ChessRobot:
    def __init__(self, port='/dev/ttyUSB0', baud_rate=115200):
        self.port = port
        self.baud_rate = baud_rate
        self.serial_connection = serial.Serial(self.port, self.baud_rate, timeout=1)
        # self.is_running = True  # Flag to control the reading thread

    def perform_move(self, message):
        formatted_message = f'{{move:{message}}}'
        self.serial_connection.write(formatted_message.encode('utf-8'))
        print(f'Sent: {formatted_message}')

    # def read_incoming(self):
    #     while self.is_running:
    #         if self.serial_connection.in_waiting > 0:
    #             incoming_data = self.serial_connection.read_until(b'}')  # Read until the closing bracket
    #             print(f'Received: {incoming_data.decode("utf-8")}')

    def close(self):
        self.serial_connection.close()
