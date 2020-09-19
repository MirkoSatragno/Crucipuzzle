import cv2
import threading

'''Questa classe definisce il thread da lanciare
Il thread si occupa di stampare continuamente un'immagine'''
class steadyImage(threading.Thread):
    def __init__(self, windowName, img):
        threading.Thread.__init__(self)
        self.running = True
        self.img = img
        self.windowName = windowName

    def run(self):
        while self.running:
            # ridefinisco nel ciclo la named window perchè se l'utente chiude per sbaglio la window la si riapre immediatamente con questi parametri specifici
            # il flag di window normal serve a poter regolare la size dell'immagine a piacere
            cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.windowName, cv2.WND_PROP_TOPMOST, 1)
            cv2.imshow(self.windowName, self.img)
            cv2.waitKey(500)
            
    def changeImage(self, newImg):
        self.img = newImg

    def stop(self):
        self.running = False
        cv2.destroyWindow(self.windowName)


'''Faccio partire il thread che stampa l'immagine.
Come valore di ritorno passo l'oggetto thread che sta stampando così in un secondo momento lo fermerò'''
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
line sono gli estremi della parola trovata nella matrice grigia'''
def drawLine(img, line, preParameters):
    heightOffset = preParameters.top
    widthOffset = preParameters.left
    deltaHeight = preParameters.meanHeight
    deltaWidth = preParameters.meanWidth

    point1 = (int(widthOffset + deltaWidth * line[0][0]), int(heightOffset + deltaHeight * line[0][1]))
    point2 = (int(widthOffset + deltaWidth * line[1][0]), int(heightOffset + deltaHeight * line[1][1]))

    lineThickness = int(min(deltaHeight, deltaWidth) / 8)

    cv2.line(img,  point1, point2, (0, 255, 255), lineThickness)
