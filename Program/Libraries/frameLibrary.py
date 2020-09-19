import cv2
import numpy as np

'''Questo mi contiene in maniera più comoda i parametri che poi mi serviranno  per i procedimenti inversi di warp e crop'''
class processedPicture:
    def __init__(self, img_warped, img_cropped, oldVertexes, newVertexes):
        self.img_warped = img_warped
        self.img_cropped = img_cropped
        self.oldVertexes = oldVertexes
        self.newVertexes = newVertexes

#################################### PUBLIC METHODS#####################################################
'''Questo serve a riconoscere la cornice del puzzle, fare il crop per avere un'immagine stiracchiata per il riconosciemtno delle lettere
e si salva anche qualche parametro sulla posizione di punti chiave per fare il processo inverso al crop e allo stiracchiamento quando bisognerà stamparla'''
def processImage(img_BGR):
    # 1 COLORS
    img_BW = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)

    # 2 VERTEXES
    # questa è la posizione dei 4 vertici del quadrilatero originale della cornice del puzzle
    sortedVertexes = getFrameVertexes(img_BW)
    # questa è la nuova posizione dei vertici che mi server per fare il wrap
    newSortedVertexes = getNewFrameVertexes(sortedVertexes)

    # 3 AFFINE IMAGE
    # stiracchio l'immagine per raddrizzare un po' il quadrilatero del puzzle
    img_warped = getWarpedImage(img_BGR, sortedVertexes, newSortedVertexes)

    # 4 CROP
    # ritaglio solo la ROI che mi interessa
    img_cropped = cropImage(img_warped, newSortedVertexes)

    return processedPicture(img_warped, img_cropped, sortedVertexes, newSortedVertexes)


'''Ricevo l'immagine colla ROI del puzzle e le linee aggiuntive delle parole trovate 
e devo ritrasformarla in base alle coordinate originali'''
def getFinalImage(img_BGR, img_cropped_lines, processedPictureWrapper):
    # 1 CROP INVERSO
    img_crop_lines = cropImageInverse(processedPictureWrapper.img_warped, img_cropped_lines, processedPictureWrapper.newVertexes)

    # 2 WARP INVERSO
    img_lines = getInverseWarpedImage(img_BGR, img_crop_lines, processedPictureWrapper.newVertexes, processedPictureWrapper.oldVertexes)

    return img_lines



############################################## PRIVATE METHODS ##########################################################
# in realtà sono tutti public, ma volevo distinguere visivamente tra quali devono essere chiamati all'esterno e quali no

'''Questa funzione riceve in input l'immagine in bianco e nero (perchè sì)
E restituisce in output un vettore dei 4 vertici della cornice del puzzle, ordinati'''
def getFrameVertexes(img_BW):

    '''Tutti i seguenti parametri sono stati scelti dopo aver fatto molti tentativi a mano.
    Per velocizzare il processo ci siamo scritti un altro programmino in python che utilizza delle trackbar.'''
    # 1 BLUR
    img_blurred = cv2.GaussianBlur(img_BW, (7, 7), 0)

    # 2 ADAPTIVE THRESHOLD
    img_thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 81, 15)

    # 3 OPEN CLOSE
    # nota: questo processo potrebbe peggiorare la leggibilità delle lettere,
    # ma non è un problema perchè qui stiamo solo cercando di riconoscere la cornice del puzzle
    kernelSize = 2
    openKernel = np.ones((kernelSize, kernelSize))
    img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, openKernel, iterations = 1)
    kernelSize = 3
    closeKernel = np.ones((kernelSize, kernelSize))
    img_openClosed = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, closeKernel, iterations = 2)

    # 4 CONTOURS
    # qui cerchiamo di approssimare la figura del quadrilatero per estrapolarne i vertici
    # fisso una dimensione minima per l'area del puzzle. Se non troviamo un quadrilatero abbastanza grande l'immagine è da scartare
    shorterSide = np.minimum(img_BW.shape[0], img_BW.shape[1])
    thresholdArea = shorterSide * shorterSide / 5
    fourVertexes = []

    contoursList, hierarchy = cv2.findContours(img_openClosed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contoursList:
        area = cv2.contourArea(contour)
        # vogliamo che il rettangolo sia abbastanza grande, e sia il più grande dell'immagine
        if thresholdArea < area:
            perimeter = cv2.arcLength(contour, True)
            vertexes = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            # vogliamo solo quadrilateri
            if len(vertexes) == 4:
                fourVertexes = vertexes
                thresholdArea = area

    if len(fourVertexes) != 4:
        raise ValueError()
    fourVertexes = [vertex[0] for vertex in fourVertexes]

    # 5 SORT VERTEXES
    # li ordino in modo che seguano un ordine circolare partendo dal pivot
    # il pivot point ha queste proprietà: appartiene al lato più lungo del quadrilatero ed è il primo dei due in base all'ordine del vettore
    sortedVertexes = sortVertexes(fourVertexes)

    return sortedVertexes


'''Lo scopo di questa funzione è quello restituire in output il vettore colle nuove posizioni dei vertici per fare il wrap,
in modo che il nuovo quadrilatero che formeranno avrà la size desiderata.
Per farlo sposto tutti i vertici tranne il pivot (quindi ne sposto 3)'''
def getNewFrameVertexes(sortedVertexes):
    newSortedVertexes = sortedVertexes.copy()

    # "vector" nel senso che non è un valore assoluto ma è con segno
    finalWidthVector, finalHeightVector = croppedDimensions(sortedVertexes)
    pivotVertex = sortedVertexes[0]
    followingVertex = sortedVertexes[1]

    newSortedVertexes[0] = pivotVertex

    # devo capire dove sta il pivot (alto, basso, SX, DX) e lo capisco confrontandolo col suo successore
    xDifference = abs(pivotVertex[0] - followingVertex[0])
    yDifference = abs(pivotVertex[1] - followingVertex[1])
    if xDifference < yDifference:
        # i punti sono uno sopra l'altro'
        newSortedVertexes[1] = pivotVertex + [0, finalHeightVector]
        newSortedVertexes[2] = pivotVertex + [finalWidthVector, finalHeightVector]
        newSortedVertexes[3] = pivotVertex + [finalWidthVector, 0]
    else:
        newSortedVertexes[1] = pivotVertex + [finalWidthVector, 0]
        newSortedVertexes[2] = pivotVertex + [finalWidthVector, finalHeightVector]
        newSortedVertexes[3] = pivotVertex + [0, finalHeightVector]

    return newSortedVertexes


'''Questa funzione riordina i vertici del quadrilatero in modo da averli adiacenti uno dopo l'altro.
Non so se alla fine sono in ordine orario o antioriario, nè in che posizione sia il primo, ma non mi interessa'''
def sortVertexes(vertexes):
    sortedVertexes = []
    vertexesList = vertexes.copy()

    # parto da uno a caso e itero cercando di volta in volta il vertice più vicino
    currentVertex = vertexesList.pop()
    sortedVertexes.append(currentVertex)
    while 0 < len(vertexesList):
        closestIndex = None
        closestDistance = None

        for index in range(len(vertexesList)):
            distance = cv2.norm(currentVertex - vertexesList[index], normType = cv2.NORM_L2)
            if closestDistance is None or distance < closestDistance:
                closestDistance = distance
                closestIndex = index

        currentVertex = vertexesList[closestIndex]
        vertexesList.pop(closestIndex)
        sortedVertexes.append(currentVertex)

    # voglio che il pivot vertex sia il primo
    pivotVertexIndex = longestSideVertexIndex(sortedVertexes)
    for i in range(pivotVertexIndex):
        firstVertex = sortedVertexes.pop(0)
        sortedVertexes.append(firstVertex)

    return sortedVertexes

'''dato un vettore di vertici mi ritorna l'indice del vettore che collegato col suo successore
forma il lato più lungo del quadrilatero'''
def longestSideVertexIndex(vertexes):
    indexMax = 0
    distanceMax = 0

    for index in range(len(vertexes)):
        vertex1 = vertexes[index]
        vertex2 = vertexes[(index + 1) % len(vertexes)]

        distance = cv2.norm(vertex1 - vertex2, normType = cv2.NORM_L2)
        if distanceMax < distance:
            indexMax = index
            distanceMax = distance

    return indexMax

'''Riceve in input 4 vertici del quadrilatero sbilenco e mi calcola quale sarà la dimensione del quadrilatero finale
restituendo in output width e height con segno rispetto al pivot'''
def croppedDimensions(sortedVertexes):
    # Parto dal pivot vertex e guardo com'è messo (orizzontale verticale) in base agli altri 2 vertci che gli stanno vicini
    previousVertex = sortedVertexes[3]
    currentVertex = sortedVertexes[0]
    followingVertex = sortedVertexes[1]

    distance1 = cv2.norm(currentVertex - followingVertex, normType = cv2.NORM_L2)
    distance2 = cv2.norm(currentVertex - previousVertex, normType = cv2.NORM_L2)

    xDifference = abs(currentVertex[0] - followingVertex[0])
    yDifference = abs(currentVertex[1] - followingVertex[1])

    # prima devo capire quali coppie di punti dovrebbero essere in verticale e quale in orizzontale ...
    if xDifference < yDifference:
        # poi devo capire quale dei 2 sta sopra l'altro, da sapere come sono posizionati l'uno rispetto all'altro
        if currentVertex[1] < followingVertex[1]:
            heightVector = distance1
        else:
            heightVector = -distance1

        # ora, per esclusione glia altri 2 sono in orizzontale, e devo capire quale sta a SX
        if currentVertex[0] < previousVertex[0]:
            widthVector = distance2
        else:
            widthVector = -distance2
    # qui invece siamo nel caso invertito in cui il punto con indice i+1 sta in orizzontale rispetto all'altro
    else:
        if currentVertex[0] < followingVertex[0]:
            widthVector = distance1
        else:
            widthVector = -distance1

        if currentVertex[1] < previousVertex[1]:
            heightVector = distance2
        else:
            heightVector = -distance2

    return int(widthVector), int(heightVector)


'''Serve a trovare il vertice in alto a sinistra partendo da 4 vertici disposti a rettangolo'''
def getHighSX(vertexes):
    minIndex = 0
    minValue = cv2.norm(vertexes[minIndex], normType = cv2.NORM_L2)

    for index in range (1, 4):
        value = cv2.norm(vertexes[index], normType = cv2.NORM_L2)
        if value < minValue:
            minIndex = index
            minValue = value

    return vertexes[minIndex]


def cropImage(img, croppedVertexes):
    highSXCroppedPoint = getHighSX(croppedVertexes)
    croppedWidth, croppedHeight = croppedDimensions(croppedVertexes)
    croppedWidth = abs(croppedWidth)
    croppedHeight = abs(croppedHeight)

    img_cropped = img[highSXCroppedPoint[1]: highSXCroppedPoint[1] + croppedHeight, highSXCroppedPoint[0]: highSXCroppedPoint[0] + croppedWidth]

    return img_cropped

def cropImageInverse(oldImg, croppedImg, CroppedVertexes):
    highSXCroppedPoint = getHighSX(CroppedVertexes)
    croppedWidth, croppedHeight = croppedDimensions(CroppedVertexes)
    croppedWidth = abs(croppedWidth)
    croppedHeight = abs(croppedHeight)

    newImg = oldImg.copy()
    newImg[highSXCroppedPoint[1]: highSXCroppedPoint[1] + croppedHeight, highSXCroppedPoint[0]: highSXCroppedPoint[0] + croppedWidth] = croppedImg

    return newImg

def getWarpedImage(img_BGR, sortedVertexes, newSortedVertexes):
    croppedMatrix = cv2.getPerspectiveTransform(np.asarray(sortedVertexes, np.float32), np.asarray(newSortedVertexes, np.float32))
    img_warped = cv2.warpPerspective(img_BGR, croppedMatrix, (img_BGR.shape[1], img_BGR.shape[0]))

    return img_warped


def getInverseWarpedImage(img_original, img_warped, newSortedVertexes, oldSortedVertexes):
    # questa immagine è ancora troncata ai bordi per colpa del warp diretto che ha fatto perdere informazioni
    img_unwarped = getWarpedImage(img_warped, newSortedVertexes, oldSortedVertexes)

    # il parametro axis riguarda la gerarchia con cui viene verificato se la condizione è vera o falsa
    # la condizione si basa sul colore
    colors_match = np.all(img_unwarped < 200, axis = 2)
    img_unwarped[colors_match] = img_original[colors_match]

    return img_unwarped
