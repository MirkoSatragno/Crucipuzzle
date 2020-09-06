import cv2
import numpy as np

#################################### PUBLIC METHODS#####################################################
'''Questa funzione riceve in input l'immagine in bianco e nero (perchè sì)
E restituisce in output un vettore dei 4 vertici della cornice del puzzle, ordinati'''
def getFrameVertexes(img_BW):
    # 1 non so se e quanto sia utile, ma gli youtuber più esperti di me lo usano e per ora mi fido
    img_BWBlurred = cv2.GaussianBlur(img_BW, (7, 7), 0)

    # 2 ADAPTIVE THRESHOLD
    # i valori sono stati decisi in maniera sperimentale sfruttando l'altro programma di AdaptiveThresholdTuning (al quale forse cambierò il nome, non lo so)
    # alla fine molti valori sono stati trovati in maniera sperimentale con un altro programmino
    img_thresh = cv2.adaptiveThreshold(img_BWBlurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 81, 15)

    # 3 OPEN CLOSE
    # dovrebbe migliorare un po' l'immagine, togliere un po' di rumore, tappare buchetti
    kernelSize = 2
    openKernel = np.ones((kernelSize, kernelSize))
    img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, openKernel, iterations=1)
    # la chiusura potrebbe chiudere il buco di qualche "A" e altra roba, ma chissenefrega perchè ora stiamo cercando di trovare il rettangolo
    kernelSize = 3
    closeKernel = np.ones((kernelSize, kernelSize))
    img_openClosed = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, closeKernel, iterations=2)

    # 4 CONTOURS
    # il risultato di questa parte sarà una nuova immagine con solo i bordi del quadrilatero identificato come frame
    img_contours = np.zeros(img_BW.shape, dtype=np.uint8)
    startingValue = np.minimum(img_BW.shape[0], img_BW.shape[1])

    contoursList, hierarchy = cv2.findContours(img_openClosed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # questi servono a trovare il rettangolo più grande, nel malaugurato caso ce ne sia più di uno
    maxContour = []
    maxArea = 0

    for contour in contoursList:
        area = cv2.contourArea(contour)
        # vogliamo che il rettangolo sia abbastanza grande
        if (area > startingValue * startingValue / 5):
            perimeter = cv2.arcLength(contour, True)
            vertexes = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            # vogliamo solo rettangoli
            if len(vertexes) == 4:
                if maxArea < area:
                    maxContour = contour.copy()
                    maxArea = area

    if len(maxContour) == 0:
        # TODO: in realtà dovrò fare una return con un qualche codice di errore
        maxArea = 123456
    else:
        cv2.drawContours(img_contours, maxContour, -1, (255, 255, 255), 2)

    # 5 VERTEXES
    # qui iddentifico i vertici del quadrilatero, ma nel dubbio li assumo ancora disordinati
    img_vertexes = img_contours.copy()
    perimeter = cv2.arcLength(maxContour, True)
    vertexes = cv2.approxPolyDP(maxContour, 0.02 * perimeter, True)
    for vertex in vertexes:
        cv2.circle(img_vertexes, (vertex[0][0], vertex[0][1]), 20, (255, 255, 255), -1)

    # 6 SORT VERTEXES
    # forse sono già in ordine, ma la documentazione non lo dice
    # nel dubbio li ordino a modo mio
    sortedVertexes = []
    sortVertexes(vertexes, sortedVertexes)

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


def getWarpedImage(img_RGB, sortedVertexes, newSortedVertexes):

    # no, questo non fa al caso nostro
    """pivotVertexIndex = longestSideVertexIndex(sortedVertexes)
    originalPoints = np.float32([sortedVertexes[(pivotVertexIndex - 1) % 4], sortedVertexes[pivotVertexIndex], sortedVertexes[(pivotVertexIndex + 1) % 4]])
    traslatedPoints = np.float32([newSortedVertexes[(pivotVertexIndex - 1) % 4], newSortedVertexes[pivotVertexIndex], newSortedVertexes[(pivotVertexIndex + 1) % 4]])
    affineMatrix = cv2.getAffineTransform(originalPoints, traslatedPoints)
    img_affine = cv2.warpAffine(img_RGB, affineMatrix, (img_RGB.shape[1], img_RGB.shape[0]))"""

    perspectiveMatrix = cv2.getPerspectiveTransform(np.asarray(sortedVertexes, np.float32), np.asarray(newSortedVertexes, np.float32))
    img_perspective = cv2.warpPerspective(img_RGB, perspectiveMatrix, (img_RGB.shape[1], img_RGB.shape[0]))

    return img_perspective


def getInverseWarpedImage(img_original, img_warped, newSortedVertexes, oldSortedVertexes):

    # questa è ancora troncata
    img_perspectiveCropped = getWarpedImage(img_warped, newSortedVertexes, oldSortedVertexes)

    img_perspective = img_perspectiveCropped.copy()
    # è per il colore nero
    zeroArray = np.array([0, 0, 0])

    # axis ha a che fare col metodo di riduzione, perchè se non lo specifico mi restituisce un univo boolean finale per tutta la matrix
    # non sono certo del perchè vada = 2, ma mi sembra fondamentale altrimenti non funziona
    colors_match = np.all(img_perspective[:, :] == zeroArray, axis = 2)
    img_perspective[colors_match] = img_original[colors_match]
    return img_perspective


############################################## PRIVATE METHODS ##########################################################
# in realtà sono tutti public, ma volevo distinguere visivamente tra quali devono essere chiamati all'esterno e quali no

'''Questa funzione riordina i vertici del quadrilatero in modo da averli adiacenti uno dopo l'altro.
Non so se alla fine sono in ordine orario o antioriario, nè in che posizione sia il primo, ma non mi interessa'''
def sortVertexes(sortVertexes_vertexes, sortVertexes_sorted):
    # li metto tutti in un vettore per lavorarci meglio
    sortVertexes_vertexesList = []
    for sortVertexes_vertex in sortVertexes_vertexes:
        sortVertexes_vertexesList.append(sortVertexes_vertex)

    # parto da uno a caso e itero cercando di volta in volta il vertice più vicino
    sortVertexes_currentVertex = sortVertexes_vertexesList.pop()
    sortVertexes_sorted.append(sortVertexes_currentVertex)
    while 0 < len(sortVertexes_vertexesList):
        sortVertexes_closestIndex = None
        sortVertexes_closestDistance = -1

        sortVertexes_index = 0
        while sortVertexes_index < len(sortVertexes_vertexesList):
            sortVertexes_distance = cv2.norm(sortVertexes_currentVertex - sortVertexes_vertexesList[sortVertexes_index],
                                             normType=cv2.NORM_L2)
            if (sortVertexes_distance < sortVertexes_closestDistance or sortVertexes_closestDistance == -1):
                sortVertexes_closestDistance = sortVertexes_distance
                sortVertexes_closestIndex = sortVertexes_index
            sortVertexes_index += 1

        sortVertexes_currentVertex = sortVertexes_vertexesList[sortVertexes_closestIndex]
        sortVertexes_vertexesList.pop(sortVertexes_closestIndex)
        sortVertexes_sorted.append(sortVertexes_currentVertex)

    return

'''dato un vettore di vertici mi ritorna l'indice del vettore che collegato col suo successore
forma il lato più lungo del quadrilatero'''
def longestSideVertexIndex(vertexes):
    indexMax = 0
    distanceMax = 0

    for index in range(0, len(vertexes)):
        vertex1 = vertexes[index]
        vertex2 = vertexes[(index + 1) % len(vertexes)]

        distance = cv2.norm(vertex1 - vertex2, normType=cv2.NORM_L2)
        if (distanceMax < distance):
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
    if (xDifference < yDifference):
        # poi devo capire quale dei 2 sta sopra l'altro, da sapere come sono posizionati l'uno rispetto all'altro
        if (currentVertex[1] < followingVertex[1]):
            heightVector = distance1
        else:
            heightVector = -distance1

        # ora, per esclusione glia altri 2 sono in orizzontale, e devo capire quale sta a SX
        if (currentVertex[0] < previousVertex[0]):
            widthVector = distance2
        else:
            widthVector = -distance2
    # qui invece siamo nel caso invertito in cui il punto con indice i+1 sta in orizzontale rispetto all'altro
    else:
        if (currentVertex[0] < followingVertex[0]):
            widthVector = distance1
        else:
            widthVector = -distance1

        if (currentVertex[1] < previousVertex[1]):
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

    #devo capire dove sta il pivot (alto, basso, SX, DX) e lo capisco confrontandolo col suo successore
    xDifference = abs(currentVertex[0] - followingVertex[0])
    yDifference = abs(currentVertex[1] - followingVertex[1])
    if (xDifference < yDifference):
        # i punti sono uno sopra l'altro'
        newSortedVertexes[(pivotVertexIndex + 1) % 4] = pivotVertex + [0, finalHeightVector]
        newSortedVertexes[(pivotVertexIndex + 2) % 4] = pivotVertex + [finalWidthVector, finalHeightVector]
        newSortedVertexes[(pivotVertexIndex + 3) % 4] = pivotVertex + [finalWidthVector, 0]
    else:
        newSortedVertexes[(pivotVertexIndex + 1) % 4] = pivotVertex + [finalWidthVector, 0]
        newSortedVertexes[(pivotVertexIndex + 2) % 4] = pivotVertex + [finalWidthVector, finalHeightVector]
        newSortedVertexes[(pivotVertexIndex + 3) % 4] = pivotVertex + [0, finalHeightVector]

    return


'''Questo non mi ricordo dove lo usavo. Nel dubbio per ora lo tengo qui che può tornar buono
Serve a fare un resize proporzionale dell'immagine specificando anche solo una delle 2 nuove dimensioni (altezza o larghezza)'''
def resizeProportional(image, newWidth=None, newHeight=None):
    (h, w) = image.shape[:2]

    if newWidth is None and newHeight is None:
        return image
    if newWidth is None:
        r = newHeight / float(h)
        dim = (int(w * r), newHeight)
    else:
        r = newWidth / float(w)
        dim = (newWidth, int(h * r))

    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)