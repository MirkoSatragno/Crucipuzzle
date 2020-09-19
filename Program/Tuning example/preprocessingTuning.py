import cv2
import numpy as np
import glob

###################################################################################################################
# QUESTO FILE E' SOLO A SCOPO ESEMPLIFICATIVO DI COME ABBIAMO LAVORATO NELLO STUDIO DI IMMAGINI CAMPIONE
# PER STABILIRE I MIGLIORI ALGORITMI E PARAMETRI DA UTILIZZARE PER IL PROCESSAMENTO DELLE IMMAGINI.
# E' UNO DEI TANTI CHE ABBIAMO SCRITTO, E NON FACENDO PARTE DEL PROGRAMMA FINALE NON ABBIAMO CURATO LA LEGGIBILITà DEL CODICE,
# MA DA UN'IDEA DEL LAVORO CHE STA DIETRO AL PRODOTTO FINITO CHE CONSEGNAMO
###################################################################################################################


def nothing():
    pass

def resizeProportional(image, newWidth = None, newHeight = None):
    (h, w) = image.shape[:2]

    if newWidth is None and newHeight is None:
        return image
    if newWidth is None:
        r = newHeight / float(h)
        dim = (int(w * r), newHeight)
    else:
        r = newWidth / float(w)
        dim = (newWidth, int(h * r))

    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)



#1 IMPORT

images_BW = []
names = []
for file in glob.glob("Cropped/*"):
    names.append(file)
    images_BW.append(cv2.imread(file, 0))


#2 TRACKBAR
trackbarWindowName = "Trackbars"
trackbarName1 = "Unsharp iterations"
trackbarName2 = "Blur iteratios"
trackbarName3 = "Block size/2"
trackbarName4 = "Bho"
trackbarName5 = "Open kernel"
trackbarName6 = "Open iterations"
trackbarName7 = "Close kernel"
trackbarName8 = "Close iterations"
cv2.namedWindow(trackbarWindowName)
cv2.createTrackbar(trackbarName1, trackbarWindowName, 0, 10, nothing)
cv2.createTrackbar(trackbarName2, trackbarWindowName, 2, 10, nothing)
cv2.createTrackbar(trackbarName3, trackbarWindowName, 10, 60, nothing)
cv2.createTrackbar(trackbarName4, trackbarWindowName, 10, 100, nothing)
cv2.createTrackbar(trackbarName5, trackbarWindowName, 1, 10, nothing)
cv2.createTrackbar(trackbarName6, trackbarWindowName, 0, 10, nothing)
cv2.createTrackbar(trackbarName7, trackbarWindowName, 2, 10, nothing)
cv2.createTrackbar(trackbarName8, trackbarWindowName, 1, 10, nothing)


puzzleWindowName = "Puzzle"
cv2.namedWindow(puzzleWindowName, cv2.WINDOW_NORMAL)
currentIndex = 0
while True:

    linearImageLength = int(np.sqrt(images_BW[currentIndex].shape[0] * images_BW[currentIndex].shape[1]))
    unsharpIterations = cv2.getTrackbarPos(trackbarName1, trackbarWindowName)
    blurIterations = cv2.getTrackbarPos(trackbarName2, trackbarWindowName)
    blockSize = cv2.getTrackbarPos(trackbarName3, trackbarWindowName)
    boh = cv2.getTrackbarPos(trackbarName4, trackbarWindowName)
    openKernelSize = cv2.getTrackbarPos(trackbarName5, trackbarWindowName)
    openIterations = cv2.getTrackbarPos(trackbarName6, trackbarWindowName)
    closeKernelSize = cv2.getTrackbarPos(trackbarName7, trackbarWindowName)
    closeIterations = cv2.getTrackbarPos(trackbarName8, trackbarWindowName)

    img = images_BW[currentIndex]

    ugly_picture = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 81, 15)

    # 1 UNSHARP
    if unsharpIterations != 0:
        for i in range(0, unsharpIterations):
            img_blurred = cv2.GaussianBlur(img, (3, 3), 0)
            img = cv2.addWeighted(img, 1.5, img_blurred, -0.5, 0)

    # 2 BLUR
    if blurIterations != 0:
        for i in range(0, blurIterations):
            img= cv2.GaussianBlur(img, (3, 3), 0)


    # 3 ADAPTIVE THRESHOLDING
    oddBlockSize = int(linearImageLength * blockSize/800) * 2 + 3
    bohValue = int(linearImageLength * boh/10000) * 2 + 3
    img_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, oddBlockSize, bohValue)

    # 4 OPEN CLOSE
    openKernelSize2 = int(linearImageLength * openKernelSize /3000) * 2 + 1
    openKernel = np.ones((openKernelSize2, openKernelSize2))
    img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, openKernel, iterations = openIterations)
    closeKernelSize2 = int(linearImageLength * closeKernelSize /3000) * 2 + 1
    closeKernel = np.ones((closeKernelSize2, closeKernelSize2))
    img_openClosed = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, closeKernel, iterations = closeIterations)

    img_openClosed = 255 - img_openClosed
    cv2.imshow(puzzleWindowName, img_openClosed)

    capture = cv2.waitKey(100)
    if capture & 0xFF == ord('4'):
        currentIndex = (currentIndex + 1) % images_BW.__len__()
    if capture & 0xFF == ord('6'):
        currentIndex = (currentIndex - 1) % images_BW.__len__()
    if capture & 0xFF == ord('q'):
        break


#cv2.waitKey(0)
cv2.destroyAllWindows()
