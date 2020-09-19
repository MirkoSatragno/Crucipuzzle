import cv2
import numpy as np

'''Questo mi contiene in maniera elegante i parametri che poi mi serviranno  per i procedimenti inversi di warp e crop'''
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
    # questa è la posizione dei 4 vertici del quadrilatero originale
    sortedVertexes = getFrameVertexes(img_BW)
    # questa è la nuova posizione dei vertici che mi server per fare il wrap
    newSortedVertexes = getNewFrameVertexes(sortedVertexes)

    # 3 AFFINE IMAGE
    # stiracchio l'immagine, per raddrizzare un po' il quadrilatero centrale
    img_warped = getWarpedImage(img_BGR, sortedVertexes, newSortedVertexes)

    # 4 CROP
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
    # 1 BLUR
    # non so se e quanto sia utile, ma gli youtuber più esperti di me lo usano e per ora mi fido
    img_blurred = cv2.GaussianBlur(img_BW, (7, 7), 0)

    # 2 ADAPTIVE THRESHOLD
    # i valori sono stati decisi in maniera sperimentale sfruttando l'altro programma di AdaptiveThresholdTuning (al quale forse cambierò il nome, non lo so)
    # alla fine molti valori sono stati trovati in maniera sperimentale con un altro programmino scritto da me
    img_thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 81, 15)

    # 3 OPEN CLOSE
    # dovrebbe migliorare un po' l'immagine, togliere un po' di rumore, tappare buchetti
    kernelSize = 2
    openKernel = np.ones((kernelSize, kernelSize))
    img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, openKernel, iterations = 1)
    # la chiusura potrebbe chiudere il buco di qualche "A" e altra roba, ma chissenefrega perchè ora stiamo cercando di trovare il rettangolo
    kernelSize = 3
    closeKernel = np.ones((kernelSize, kernelSize))
    img_openClosed = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, closeKernel, iterations = 2)

    # 4 CONTOURS
    # qui cerchiamo di approssimare la figura del quadrilatero per estrapolarne i vertici
    # fisso una dimensione minima per l'area. Se non troviamo un quadrilatero grande almeno così allora la foto fa schifo
    shorterSide = np.minimum(img_BW.shape[0], img_BW.shape[1])
    minimumArea = shorterSide * shorterSide / 5

    contoursList, hierarchy = cv2.findContours(img_openClosed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # questi servono a trovare il rettangolo più grande, nel malaugurato caso ce ne sia più di uno
    fourVertexes = []
    thresholdArea = minimumArea

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

    # 5 SORT VERTEXES
    # li ordino a modo mio, in modo che seguano un ordine circolare
    sortedVertexes = []
    sortVertexes(fourVertexes, sortedVertexes)

    return sortedVertexes


'''Lo scopo di questa funzione è quello restituire in output il vettore colle nuove posizioni dei vertici per fare il wrap'''
def getNewFrameVertexes(sortedVertexes):
    # 1 FINAL FRAME SIZE
    # il mio pivot point ha queste proprietà: appartiene al lato più lungo del quadrilatero ed è il primo dei due in base all'ordine del vettore
    pivotVertexIndex = longestSideVertexIndex(sortedVertexes)
    finalWidthVector, finalHeightVector = finalDimensions(sortedVertexes, pivotVertexIndex)

    # 2 NEW VERTEXES
    newSortedVertexes = sortedVertexes.copy()
    getNewSortedVertexes(sortedVertexes, newSortedVertexes, pivotVertexIndex, finalWidthVector, finalHeightVector)

    return newSortedVertexes


'''Questa funzione riordina i vertici del quadrilatero in modo da averli adiacenti uno dopo l'altro.
Non so se alla fine sono in ordine orario o antioriario, nè in che posizione sia il primo, ma non mi interessa'''
def sortVertexes(vertexes, sortedVertexes):
    # li metto tutti in un vettore per lavorarci meglio
    vertexesList = [ vertex for vertex in vertexes]

    # parto da uno a caso e itero cercando di volta in volta il vertice più vicino
    currentVertex = vertexesList.pop()
    sortedVertexes.append(currentVertex)
    while 0 < len(vertexesList):
        closestIndex = None
        closestDistance = None

        for index in range (0, len(vertexesList)):
            distance = cv2.norm(currentVertex - vertexesList[index], normType = cv2.NORM_L2)
            if closestDistance is None or distance < closestDistance:
                closestDistance = distance
                closestIndex = index

        currentVertex = vertexesList[closestIndex]
        vertexesList.pop(closestIndex)
        sortedVertexes.append(currentVertex)

    return

'''dato un vettore di vertici mi ritorna l'indice del vettore che collegato col suo successore
forma il lato più lungo del quadrilatero'''
def longestSideVertexIndex(vertexes):
    indexMax = 0
    distanceMax = 0

    for index in range(0, len(vertexes)):
        vertex1 = vertexes[index]
        vertex2 = vertexes[(index + 1) % len(vertexes)]

        distance = cv2.norm(vertex1 - vertex2, normType = cv2.NORM_L2)
        if distanceMax < distance:
            indexMax = index
            distanceMax = distance

    return indexMax

'''Riceve in input 4 vertici del quadrilatero sbilenco e mi calcola quale sarà la dimensione del quadrilatero finale
restituendo in output width e height'''
def finalDimensions(sortedVertexes, pivotVertexIndex):
    widthVector = 0
    heightVector = 0

    #Parto dal pivot vertex e guardo com'è messo (orizzontale verticale) in base agli altri 2 vertci che gli stanno vicini
    # per ragioni di datatype, ogni vertex è in realtà vettore di punti (con un solo punto) e per questo aggiungo uno [0] alla fine
    previousVertex = sortedVertexes[(pivotVertexIndex - 1) % len(sortedVertexes)][0]
    currentVertex = sortedVertexes[pivotVertexIndex][0]
    followingVertex = sortedVertexes[(pivotVertexIndex + 1) % len(sortedVertexes)][0]

    distance1 = cv2.norm(currentVertex - followingVertex, normType=cv2.NORM_L2)
    distance2 = cv2.norm(currentVertex - previousVertex, normType=cv2.NORM_L2)

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


'''Qui l'obiettivo è spostare i vertici in modo che il nuovo quadrilatero che formeranno avrà la size desiderata.
Per farlo sposto tutti i vertici tranne il pivot (quindi ne sposto 3)'''
def getNewSortedVertexes(sortedVertexes, newSortedVertexes, pivotVertexIndex, finalWidthVector, finalHeightVector):
    pivotVertex = sortedVertexes[pivotVertexIndex]

    currentVertex = sortedVertexes[pivotVertexIndex][0]
    followingVertex = sortedVertexes[(pivotVertexIndex + 1) % len(sortedVertexes)][0]

    # devo capire dove sta il pivot (alto, basso, SX, DX) e lo capisco confrontandolo col suo successore
    xDifference = abs(currentVertex[0] - followingVertex[0])
    yDifference = abs(currentVertex[1] - followingVertex[1])
    if xDifference < yDifference:
        # i punti sono uno sopra l'altro'
        newSortedVertexes[(pivotVertexIndex + 1) % 4] = pivotVertex + [0, finalHeightVector]
        newSortedVertexes[(pivotVertexIndex + 2) % 4] = pivotVertex + [finalWidthVector, finalHeightVector]
        newSortedVertexes[(pivotVertexIndex + 3) % 4] = pivotVertex + [finalWidthVector, 0]
    else:
        newSortedVertexes[(pivotVertexIndex + 1) % 4] = pivotVertex + [finalWidthVector, 0]
        newSortedVertexes[(pivotVertexIndex + 2) % 4] = pivotVertex + [finalWidthVector, finalHeightVector]
        newSortedVertexes[(pivotVertexIndex + 3) % 4] = pivotVertex + [0, finalHeightVector]

    return

'''Serve a trovare l'indice del vertice in alto a destra partendo da 4 vertici disposti a rettangolo'''
def getHighSX(vertexes):
    minIndex = 0
    minValue = cv2.norm(vertexes[minIndex], normType=cv2.NORM_L2)

    for index in range (1, 4):
        value = cv2.norm(vertexes[index], normType=cv2.NORM_L2)
        if value < minValue:
            minIndex = index
            minValue = value

    return minIndex


def cropImage(img, croppedVertexes):
    highSXCroppedPointIndex = getHighSX(croppedVertexes)
    highSXCroppedPoint = croppedVertexes[highSXCroppedPointIndex]
    # questa funzione non era stata pensata per questo, ma mi torna comodo anche così
    croppedWidth, croppedHeight = finalDimensions(croppedVertexes, highSXCroppedPointIndex)

    img_cropped = img[highSXCroppedPoint[0][1]: highSXCroppedPoint[0][1] + croppedHeight, highSXCroppedPoint[0][0]: highSXCroppedPoint[0][0] + croppedWidth]

    return img_cropped

def cropImageInverse(oldImg, croppedImg, CroppedVertexes):
    highSXCroppedePointIndex = getHighSX(CroppedVertexes)
    highSXCroppedPoint = CroppedVertexes[highSXCroppedePointIndex]
    # questa funzione non era stata pensata per questo, ma mi torna comodo anche così
    croppedWidth, croppedHeight = finalDimensions(CroppedVertexes, highSXCroppedePointIndex)

    newImg = oldImg.copy()
    newImg[highSXCroppedPoint[0][1]: highSXCroppedPoint[0][1] + croppedHeight, highSXCroppedPoint[0][0]: highSXCroppedPoint[0][0] + croppedWidth] = croppedImg

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
    img_unwarped[colors_match] = img_original

    return img_unwarped
