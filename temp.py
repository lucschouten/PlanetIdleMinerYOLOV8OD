import pyautogui
import time


def main():
    
    time.sleep(10)
    for i in range(0, 198):
        # Click on the screen
        pyautogui.press('R')
        # press arrow right
        time.sleep(0.1)
        pyautogui.press('right')
        time.sleep(0.9)
main()