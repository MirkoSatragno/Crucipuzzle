import cv2
import numpy as np
import time
# per il file dialog
import tkinter as tk
from tkinter import filedialog

from Libraries import plotLibrary as plotLib
from Libraries import frameLibrary as frameLib
from Libraries import wordLibrary as wordLib
from Libraries import ocr2Library as ocrLib


def getBGRPicture():
    file_path = filedialog.askopenfilename()
    img_BGR = cv2.imread(file_path)
    if img_BGR is None:
        raise FileNotFoundError()

    return img_BGR

def getBGRCameraPicture():
    # la ragione ufficiale per il secondo parametro di VideoCapture è "Can be used to enforce a specific reader implementation if multiple are available"
    # la vera ragione è che Stack overflow m'ha detto che è così che elimino un warning fastidioso
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    captureWindowName = "Take a picture"
    cv2.namedWindow(captureWindowName)
    # serve a far comparire la window in foreground
    cv2.setWindowProperty(captureWindowName, cv2.WND_PROP_TOPMOST, 1)

    while True:
        ret, frame = capture.read()
        cv2.imshow(captureWindowName, frame)

        if cv2.waitKey(50) & 0xFF == ord('p'):
            break

    capture.release()
    cv2.destroyWindow(captureWindowName)

    return frame

def solvePuzzle(inputMethod):

    if inputMethod == "1":
        print("Press \"P\" to take a picture")
        print("(Please, mind that the pop up camera window must be your selected window, in order to take the picture)")
        img_BGR = getBGRCameraPicture()

    else:
        try:
            print("Select a picture")
            time.sleep(1)
            img_BGR = getBGRPicture()
        except FileNotFoundError:
            print("Picture not found")
            time.sleep(2)
            return 0

    print("\nProcessing image...")
    lettersDictionary = wordLib.getLettersDictionary()
    try:
        processedImgWrapper = frameLib.processImage(img_BGR)
        preprocessedParameters = ocrLib.OCRPrecomputation(processedImgWrapper.img_cropped)
        charactersMatrix = ocrLib.OCRComputation(processedImgWrapper.img_cropped, preprocessedParameters)
        grayMatrix = wordLib.matrixToGrayscale(lettersDictionary, charactersMatrix)
        for row in charactersMatrix:
            print(row)
    except Exception:
        print("Bad picture. Puzzle not found")
        time.sleep(1)
        return 0



    # serve se lo user vuole pulire l'immagine
    originalImg_cropped = processedImgWrapper.img_cropped.copy()
    steadyImage = plotLib.plotSteadyImage(img_BGR)
    print("\nPicture has been acquired correctly. Now you can start searching words.")
    print("If you want to clean the picture press \"C\".\nWhen you're done, press \"Q\" to quit.\n")

    word = ""
    while word != "q":
        word = str(input("Type a word to search for: "))

        if wordLib.isWordValid(word):
            try:
                line = wordLib.findWord(grayMatrix , word, lettersDictionary)
            except FileNotFoundError:
                print("Word not found")
                continue
            plotLib.drawLine(processedImgWrapper.img_cropped, grayMatrix, line, preprocessedParameters)

            img_lines = frameLib.getFinalImage(img_BGR, processedImgWrapper.img_cropped, processedImgWrapper)
            plotLib.changeSteadyImage(steadyImage, img_lines)
        else:
            if word == "c":
                processedImgWrapper.img_cropped = originalImg_cropped.copy()
                plotLib.changeSteadyImage(steadyImage, img_BGR)
            else:
                if word != "q" and word != "c":
                    print("Invalid string: " + word)


    plotLib.removeSteadyImage(steadyImage)
    return


def main():
    # serve per il file dialog
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)

    print("Welcome. This is a Crucipuzzle solver program.")

    key = ""
    while key != "3":
        print("\nWhat do you want to do")
        print("1) Solve a puzzle from camera")
        print("2) Solve a puzzle from file")
        print("3) Quit")

        key = str(input())

        if key == "1" or key == "2":
            solvePuzzle(key)


if __name__ == '__main__':
    main()
