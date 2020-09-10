import cv2
# per il file dialog
import tkinter as tk
from tkinter import filedialog
import time

from Libraries import plotLibrary
from Libraries import frameLibrary as frameLib
from Libraries import wordLibrary as wordLib


def getBGRPicture():
    file_path = filedialog.askopenfilename()
    img_BGR = cv2.imread(file_path)
    if img_BGR is None:
        raise FileNotFoundError()

    # img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    return img_BGR

def solvePuzzle():

    try:
        print("Select a picture")
        time.sleep(1)
        img_BGR = getBGRPicture()
    except FileNotFoundError:
        print("Picture not found")
        time.sleep(2)
        return 0

    try:
        processedImgWrapper = frameLib.processImage(img_BGR)
    except ValueError:
        print("Bad picture. Puzzle not found")
        time.sleep(2)
        return 0

    lettersDictionary = wordLib.getLettersDictionary()
    # TODO: get greyscale matrix

    steadyImage = plotLibrary.plotSteadyImage(processedImgWrapper.img_cropped)
    print("\nPicture has been acquired correctly. Now you can start typing words.\nWhen you're done, press \"Q\" to quit.\n")

    word = ""
    while word != "q":
        word = str(input("Type a word to search for: "))

        if wordLib.isWordValid(word):
            # TODO: cercare la parola
            # pointA, pointB = wordLib.findWord( ???? , word, lettersDictionary)

            # TODO: disegnare le linee

            img_lines = frameLib.getFinalImage(img_BGR, processedImgWrapper.img_cropped, processedImgWrapper)
            plotLibrary.removeSteadyImage(steadyImage)
            steadyImage = plotLibrary.plotSteadyImage(img_lines)
        else:
            print("Invalid string: " + word)


    plotLibrary.removeSteadyImage(steadyImage)
    return


def main():
    # serve per il file dialog
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)


    print("Welcome. This is a Crucipuzzle solver program.")

    key = ""
    while key != "2":
        print("\nWhat do you want to do")
        print("1) Solve a puzzle")
        print("2) Quit")
        key = str(input())

        if key == "1":
            solvePuzzle()


if __name__ == '__main__':
    main()
