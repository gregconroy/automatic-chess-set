import serial
import time

# Configure the serial port and baud rate
SERIAL_PORT = '/dev/ttyUSB0'  # Your ESP's USB port
BAUD_RATE = 115200

# Function to send a message to the ESP
def send_message(serial_connection, message_type, message):
    formatted_message = f'{{{message_type}:{message}}}'
    serial_connection.write(formatted_message.encode('utf-8'))
    print(f'Sent: {formatted_message}')

# Function to read incoming messages from the ESP
def read_incoming(serial_connection):
    while True:
        if serial_connection.in_waiting > 0:
            incoming_data = serial_connection.read_until(b'}')  # Read until the closing bracket
            print(f'Received: {incoming_data.decode("utf-8")}')

# Main function
def main():
    # Open the serial connection
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
        time.sleep(2)  # Wait for the serial connection to initialize

        try:
            # Example of sending a message
            send_message(ser, "move", "a1a3")  # Sending a move command
            read_incoming(ser)  # Read incoming messages
        except KeyboardInterrupt:
            print("Exiting program.")

if __name__ == "__main__":
    main()
