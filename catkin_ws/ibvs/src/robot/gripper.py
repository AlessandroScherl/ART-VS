# First make sure to install pyserial:
# pip install pyserial

import serial
import time
import argparse

class Gripper:
    def __init__(self, port="/dev/ttyUSB0"):
        # Initialize connection
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=115200,
                timeout=0.2,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            # Activate Gripper
            self.ser.write(b'\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30')
            time.sleep(0.1)
            self.ser.write(b"\x09\x03\x07\xD0\x00\x01\x85\xCF")
            time.sleep(0.1)
            print("Gripper initialized successfully")
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            raise

    def close_gripper(self):
        try:
            self.ser.write(b"\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\xFF\xFF\xFF\x42\x29")
            time.sleep(1)
            print("Gripper closed")
            return True
        except serial.SerialException as e:
            print(f"Error closing gripper: {e}")
            return False

    def open_gripper(self):
        try:
            self.ser.write(b"\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\x00\xFF\xFF\x72\x19")
            time.sleep(1)
            print("Gripper opened")
            return True
        except serial.SerialException as e:
            print(f"Error opening gripper: {e}")
            return False

    def __del__(self):
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()

def main():
    parser = argparse.ArgumentParser(description='Control the gripper')
    parser.add_argument('action', choices=['open', 'close'], help='Action to perform')
    parser.add_argument('--port', default='/dev/ttyUSB0', help='Serial port (default: /dev/ttyUSB0)')
    
    args = parser.parse_args()
    
    try:
        gripper = Gripper(args.port)
        if args.action == 'open':
            gripper.open_gripper()
        else:
            gripper.close_gripper()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
