import Jetson.GPIO as GPIO
import time

# Pin Definitions
motor_pin_a = 18  # BOARD pin 12, BCM pin 18
motor_pin_b = 23  # BOARD pin 16, BCM pin 16

def main():
    # Pin Setup:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(motor_pin_a, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(motor_pin_b, GPIO.OUT, initial=GPIO.LOW)

    print("Starting demo now! Press CTRL+C to exit")
    
    # Start with motor_pin_a HIGH and motor_pin_b LOW
    curr_value_pin_a = GPIO.HIGH
    curr_value_pin_b = GPIO.LOW
    
    try:
        while True:
            # Toggle the output every half a second (0.5 seconds)
            time.sleep(0.5)
            print("Outputting {} to pin {} AND {} to pin {}".format(curr_value_pin_a, motor_pin_a, curr_value_pin_b, motor_pin_b))
            GPIO.output(motor_pin_a, curr_value_pin_a)
            GPIO.output(motor_pin_b, curr_value_pin_b)
            
            # Toggle the pin values
            curr_value_pin_a = GPIO.LOW if curr_value_pin_a == GPIO.HIGH else GPIO.HIGH
            curr_value_pin_b = GPIO.LOW if curr_value_pin_b == GPIO.HIGH else GPIO.HIGH
            
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
