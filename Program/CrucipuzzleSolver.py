import cv2
import time
# per il file dialog
import tkinter as tk
from tkinter import filedialog

from Libraries import plotLibrary as plotLib
from Libraries import frameLibrary as frameLib
from Libraries import wordLibrary as wordLib
from Libraries import ocr2Library as ocrLib


'''Questa funzione serve ad acquisire un'immagine da file.
Restituisce un immagine in BGR'''
def getBGRPicture():
    file_path = filedialog.askopenfilename(title = "Select image", initialdir = "./../Sample pictures")
    img_BGR = cv2.imread(file_path)
    if img_BGR is None:
        raise FileNotFoundError()

    return img_BGR

'''Questa funzione serve ad acquisire un'immagine sul momento usando una camera integrata o connessa al computer.
Restituisce un'immagine BGR'''
def getBGRCameraPicture():
    # il secondo parametro di VideoCapture viene usato perchè "Can be used to enforce a specific reader implementation if multiple are available"
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    captureWindowName = "Take a picture"
    cv2.namedWindow(captureWindowName)
    # serve a far comparire la window in foreground
    cv2.setWindowProperty(captureWindowName, cv2.WND_PROP_TOPMOST, 1)

    while True:
        ret, frame = capture.read()
        cv2.imshow(captureWindowName, frame)

        if cv2.waitKey(33) & 0xFF == ord('p'):  # Uso 33 come input delay per avere idealmente 30 fps
            break

    capture.release()
    cv2.destroyWindow(captureWindowName)

    return frame



'''Questa è la funzione principale del programma. Riceve in input solo il metodo di acquisizione dell'immagine.
Si occupa di chiamare le funzioni di alto livello e ne gestisce eventuali eccezioni'''
def solvePuzzle(inputMethod):

    # 1 IMAGE ACQUISITION
    if inputMethod == "1":
        try:
            print("Select a picture.")
            time.sleep(1)
            img_BGR = getBGRPicture()
        except FileNotFoundError:
            print("Picture not found.")
            time.sleep(1)
            return 0
    else:
        print("Press \"P\" to take a picture.")
        print("(Please, mind that the pop up camera window must be your selected window, in order to take the picture)")
        img_BGR = getBGRCameraPicture()


    # 2 PUZZLE RECOGNITION
    print("\nProcessing image...")
    # contiene una codifica delle lettere in scala di grigi
    lettersDictionary = wordLib.getLettersDictionary()
    try:
        # per prima cosa identifichiamo la ROI del puzzle
        processedImgWrapper = frameLib.processImage(img_BGR)
        # poi estrapoliamo delle informazioni dall'immagine del puzzle
        preprocessedParameters = ocrLib.OCRPrecomputation(processedImgWrapper.img_cropped)
        # eseguiamo il riconoscimento dei caratteri
        charactersMatrix = ocrLib.OCRComputation(processedImgWrapper.img_cropped, preprocessedParameters)
        # infine codifichiamo il puzzle come immagine
        grayMatrix = wordLib.matrixToGrayscale(lettersDictionary, charactersMatrix)
    except Exception:
        print("Bad picture. Puzzle not found")
        time.sleep(1)
        return 0


    # 3 WORD SEARCH
    print("\nPicture has been acquired correctly. Now you can start searching words.")
    print("If you want to clean the picture press \"C\".\nWhen you're done, press \"Q\" to quit.\n")
    # serve se lo user vuole pulire l'immagine
    originalImg_cropped = processedImgWrapper.img_cropped.copy()
    steadyImage = plotLib.plotSteadyImage(img_BGR)
    word = ""
    while word != "q":
        word = str(input("Type a word to search for: "))

        if wordLib.isWordValid(word):
            try:
                line = wordLib.findWord(grayMatrix , word, lettersDictionary)
            except FileNotFoundError:
                print("Word not found.")
                continue
            plotLib.drawLine(processedImgWrapper.img_cropped, line, preprocessedParameters)

            img_lines = frameLib.getFinalImage(img_BGR, processedImgWrapper.img_cropped, processedImgWrapper)
            plotLib.changeSteadyImage(steadyImage, img_lines)
        else:
            if word == "c":
                # ripristino l'immagine originale senza le righe delle parole già trovate
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
        print("\nWhat do you want to do:")
        print("1) Solve a puzzle from file")
        print("2) Solve a puzzle from camera")
        print("3) Quit")

        key = str(input())

        if key == "1" or key == "2":
            solvePuzzle(key)


if __name__ == '__main__':
    main()
