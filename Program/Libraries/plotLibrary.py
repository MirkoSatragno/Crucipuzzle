import cv2
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
            cv2.imshow(self.name, self.img)
            cv2.waitKey()

    def stop(self):
        self.running = False
        cv2.destroyWindow(self.name)


'''Faccio partire il thread che stampa l'immagine.
Come valore di ritorno passo l'oggetto thread che sta stampando così in un secondo momento lo fremerò'''
def plotSteadyImage(img, windowName = "Puzzle"):
    myThread = steadyImage(windowName, img)
    myThread.start()
    return myThread


'''Ricevo il thread che sta stampando e lo fermo'''
def removeSteadyImage(thread):
    thread.stop()
    thread.join()


