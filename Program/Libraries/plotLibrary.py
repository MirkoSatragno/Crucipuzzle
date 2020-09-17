import cv2
import numpy as np
import threading

'''Questa classe definisce il thread da lanciare
Il thread si occupa di stampare continuamente un'immagine'''
class steadyImage(threading.Thread):
    def __init__(self, name, img):
        threading.Thread.__init__(self)
        self.running = True
        self.img = img
        self.name = name

    def run(self):
        while self.running:
            # il flag di window normal serve a poter regolare la size dell'immagine a piacere
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.name, cv2.WND_PROP_TOPMOST, 1)
            cv2.imshow(self.name, self.img)
            cv2.waitKey(500)
            
    def changeImage(self, newImg):
        self.img = newImg

    def stop(self):
        self.running = False
        cv2.destroyWindow(self.name)


'''Faccio partire il thread che stampa l'immagine.
Come valore di ritorno passo l'oggetto thread che sta stampando così in un secondo momento lo fremerò'''
def plotSteadyImage(img, windowName = "Puzzle"):
    myThread = steadyImage(windowName, img)
    myThread.start()
    return myThread

def changeSteadyImage(thread, newImg):
    thread.changeImage(newImg)


'''Ricevo il thread che sta stampando e lo fermo'''
def removeSteadyImage(thread):
    thread.stop()
    thread.join()

'''img è l'immagine dove devo disegnare
matrixBW è la matrice in scala d grigi
line sono gli estremi della parola trovata nella matrice grigia
HSX e LDX sono le coordinate delle lettere estreme nell'immagine'''
def drawLine(img, matrixBW, line, preParameters):
    heightOffset = preParameters.top #HSX[1]
    widthOffset = preParameters.left #HSX[0]
    deltaHeight = preParameters.meanHeight
    deltaWidth = preParameters.meanWidth

    point1 = (int(widthOffset + deltaWidth * line[0][0]), int(heightOffset + deltaHeight * line[0][1]))
    point2 = (int(widthOffset + deltaWidth * line[1][0]), int(heightOffset + deltaHeight * line[1][1]))

    #lineThickness = int(np.sqrt(height * width / (matrixBW.shape[0] * matrixBW.shape[1])) / 8)
    lineThickness = int(min(deltaHeight, deltaWidth) / 8)

    cv2.line(img,  point1, point2, (0, 255, 255), lineThickness)

