import cv2
# per il file dialog
import tkinter as tk
from tkinter import filedialog
import time

from Libraries import plotLibrary
from Libraries import frameLibrary as frameLib
from Libraries import wordLibrary as wordLib


def getRGBPicture():
    file_path = filedialog.askopenfilename()
    img_BGR = cv2.imread(file_path)
    if img_BGR is None:
        raise FileNotFoundError()

    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    return img_RGB

def solvePuzzle():

    try:
        print("Select a picture")
        time.sleep(1)
        img_RGB = getRGBPicture()
    except FileNotFoundError:
        print("Picture not found")
        return 0

    processedImgWrapper = frameLib.processPicture(img_RGB)
    steadyImage = plotLibrary.plotSteadyImage(processedImgWrapper.img_cropped)

    print("\nPicture has been acquired correctly. Now you can start typing words.\nWhen you're done, press \"Q\" to quit.\n")

    key = ""
    while key != "q":

        key = str(input("Type a word to search for: "))

        if (key == "dog"):
            print("found")

    plotLibrary.removeSteadyImage(steadyImage)
    return


def main():
    # serve per il file dialog
    root = tk.Tk()
    root.withdraw()


    print("Welcome. This is a Crucipuzzle solver program.")

    key = ""
    while key != "2":
        print("\nWhat do you want to do")
        print("1) Solve a puzzle")
        print("2) Quit")
        key = str(input())

        if( key == "1"):
            solvePuzzle()


if __name__ == '__main__':
    main()