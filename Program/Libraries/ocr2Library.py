import cv2
import pytesseract
import statistics
import matplotlib.pyplot as plot

from Libraries import frameLibrary as frameLib

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


class CharacterWrapper:
    def __init__(self, char, left, top, right, bottom):
        self.char = char
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

        self.width = right - left
        self.height = top - bottom
        self.pos = [int((right + left)/2), int((bottom + top)/2)]

class PreOCRParameters:
    def __init__(self, rows, columns, left, top, right, bottom):
        self.rows = rows
        self.columns = columns
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

        # non li casto a int o poi mi trascino dietro errori di approssimazione che crescono linearmente
        self.meanWidth = (right - left)/(columns - 1)
        self.meanHeight = (bottom - top)/(rows - 1)


'''E' un metodo abbastanza preciso per trovare il numero di righe di un puzzle.
Riceve in input un'immagine binaria'''
def getRowsNumber(img_thresh):
    # data contiene una stringa unica con tutto il risultato
    config = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6"
    data = pytesseract.image_to_string(img_thresh, config = config)

    word_strings = [w for w in data.split("\n")]
    word_strings = list(filter(lambda w: w != "\f", word_strings))

    # filtro via tutte le righe troppo corte perchè verosimilmente sono outliers
    meanLength = statistics.mean([len(word) for word in word_strings])
    lengthThreshold = int(meanLength * 0.5)
    word_strings = list(filter(lambda word: len(word) > lengthThreshold, word_strings))
    rowsNumber = len(word_strings)

    return rowsNumber


'''Questa funzione legge tutte le lettere possibili, le trasforma in oggetti CharacterWrapper e ne fa una lista.
In input riceve un'immagine binaria del puzzle'''
def getCharactersLists(img_thresh):
    config = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6"
    data = pytesseract.image_to_boxes(img_thresh, config = config)

    data_split = list([row] for row in data.split("\n"))
    data_split_filtered = list(filter(lambda row: row[0] != "", data_split))
    data_matrix = [[element for element in row[0].split(" ")] for row in data_split_filtered]

    # il sistema di riferimento usato da pytesseract per le righe è invertito
    charactersLists = [ CharacterWrapper(el[0], int(el[1]), img_thresh.shape[0] - int(el[2]), int(el[3]), img_thresh.shape[0] - int(el[4])) for el in data_matrix ]

    return charactersLists


'''Questo metodo riceve in input una lista di CharactersWrapper e deve filtrare via quelli troppo grandi e troppo piccoli.
Il risultato è una lista di CharacterWrapper con solo caratteri di dimensione uniforme fra loro'''
def filterBySize(charactersList):

    # le I son troppo piccole e le scarto
    chars_noI = list(filter(lambda el:  el.char != "I", charactersList))


    # scarto tutte le aree troppo grandi o troppo piccole
    charAreas = list(el.width * el.height for el in chars_noI)
    meanArea = statistics.mean(charAreas)
    chars_filteredAreas = list(filter(lambda el: el.width * el.height < meanArea * 1.5, chars_noI))
    newCharAreas =  list(el.width * el.height for el in chars_filteredAreas)
    newMeanArea = statistics.mean(newCharAreas)

    while meanArea != newMeanArea:
        meanArea = newMeanArea

        chars_filteredAreas = list(filter(lambda el: meanArea * 0.5 < el.width * el.height < meanArea * 1.5, chars_filteredAreas))
        newCharAreas = list(el.width * el.height for el in chars_filteredAreas)
        newMeanArea = statistics.mean(newCharAreas)


    # scarto tutte le aree con ampiezza troppo piccola
    charWidths = list(el.width for el in chars_filteredAreas)
    meanWidth = statistics.mean(charWidths)
    chars_filteredWidth  = list(filter(lambda el: meanWidth * 0.7 <el.width < meanWidth * 1.4 , chars_filteredAreas))
    newCharWidths = list(el.width for el in chars_filteredWidth)
    newMeanWidth = statistics.mean(newCharWidths)

    while meanWidth != newMeanWidth:
        meanWidth = newMeanWidth

        chars_filteredWidth = list(filter(lambda el: meanWidth * 0.7 < el.width < meanWidth * 1.4, chars_filteredWidth))
        newCharWidths = list(el.width for el in chars_filteredWidth)
        newMeanWidth = statistics.mean(newCharWidths)


    # scarto tutte le aree con altezza troppo piccola
    charHeights = list(el.width for el in chars_filteredWidth)
    meanHeight = statistics.mean(charHeights)
    chars_filteredHeight = list(filter(lambda el: meanHeight * 0.7 < el.width < meanHeight * 1.3, chars_filteredWidth))
    newCharHeights = list(el.width for el in chars_filteredHeight)
    newMeanHeight = statistics.mean(newCharHeights)

    while meanHeight != newMeanHeight:
        meanHeight = newMeanHeight

        chars_filteredHeight = list(filter(lambda el: meanHeight * 0.7 < el.width < meanHeight * 1.3, chars_filteredHeight))
        newCharHeights = list(el.width for el in chars_filteredHeight)
        newMeanHeight = statistics.mean(newCharHeights)

    return chars_filteredHeight



'''Questa funzione serve a trasformare un lista di caratteri in una matrice di righe del puzzle'''
def getHorizontalMatrix(charactersList):

    '''Per riordinarle mi baso sull'assunzione che le lettere sono già ordinate per righe quando vengono riconosciute.
    Tutto ciò che devo fare è capire dove finiscono le righe, perchè nelle varie righe non tutte e lettere sono state riconosciute'''
    horizMatrix = []
    rowIndex = 0
    previousEl = None
    for el in charactersList:
        if len(horizMatrix) == 0:
            horizMatrix.append([el])
            previousEl = el
            continue

        if previousEl.pos[0] < el.pos[0]:
            horizMatrix[rowIndex].append(el)
            previousEl = el
        else:
            horizMatrix.append([el])
            previousEl = el
            rowIndex = rowIndex + 1

    # alla fine filtro sulla lunghezza delle righe per rimuovere gli outliers
    maxRowLength = max([len(row) for row in horizMatrix])
    horizMatrix = list(filter(lambda row: maxRowLength * 0.5 < len(row), horizMatrix))

    return horizMatrix


'''Questa funzione serve alla funzione di getVerticalMatrix per capire se una lettera appartiene ad una specifica colonna'''
def doesBelongToColumn(currentChar, vertMatrix, columnIndex):
    lastCharAddedInColumn_Index = len(vertMatrix[columnIndex]) - 1
    lastCharAddedInColumn = vertMatrix[columnIndex][lastCharAddedInColumn_Index]

    if lastCharAddedInColumn.left < currentChar.pos[0] < lastCharAddedInColumn.right:
        return True

    return False


'''Questa funzione serve a trasformare un lista di caratteri in una matrice di colonne del puzzle'''
def getVerticalMatrix(charactersList):

    # per dividerli in colonne confronto se la posizione di una lettera è compresa fra i limiti destro e sinistro della lettera sopra
    vertMatrix = []
    for el in charactersList:
        found = False
        for columnIndex in range(len(vertMatrix)):
            # se riesco a piazzare una lettera solla alla lettera di una colonna già iniziata ce la piazzo
            if doesBelongToColumn(el, vertMatrix, columnIndex) and not found:
                vertMatrix[columnIndex].append(el)
                found = True

        # se non sono riuscito a piazzare la lettera in nessuna colonna esistente ne creo una nuova
        if not found:
            vertMatrix.append([el])

    # alla fine filtro sull'altezza' delle colonne per rimuovere gli outliers
    maxColumnLen = max([len(col) for col in vertMatrix])
    vertMatrix = list(filter(lambda col: maxColumnLen * 0.5 < len(col), vertMatrix))

    return vertMatrix


'''Questa funzione serve a trovare gli estremi destro e sinistro, o alto e basso, dei caratteri di un puzzle.
Per decidere se la funzione lavorerà sulle righe o sulle colonne riceve in input un indice'''
def getMatrixMeanMinMeanMax(matrix, shapeIndex):
    # questo array values contiene o le coordinate delle righe o le coordinate delle colonne
    values = [ statistics.mean([ el.pos[shapeIndex] for el in array]) for array in matrix]
    values = sorted(values)

    return int(min(values)), int(max(values))


'''Questa funzione serve ad estrapolare dei parametri utili al riconoscimento dei caratteri del puzzle.
Non esegue ancora il riconoscimento vero e proprio, però lo sfrutta a suo modo per ottenere le informazioni che le servono.
Alla fine restituisce un oggetto di tipo PreOCRParameters che contiene dati sul puzzle che servono ad effettuare poi il riconoscimento vero e proprio.'''
def OCRPrecomputation(img_BGR):
    # 1 THRESHOLD
    img_BGR = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.adaptiveThreshold(img_BGR, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 81, 15)


    # 2 GET CHARACTERS LIST
    # ottengo una lista di CharacterWrapper, con solo caratteri di dimensione media simile fra loro.
    # L'obiettivo è identificare solo i caratteri che sono stati riconosciuti con certezza, escludendo via via quelli incerti.
    charactersList = getCharactersLists(img_thresh)
    charactersList_Filtered = filterBySize(charactersList)


    # 3 CHARACTERS MATRIXES
    horizontalMatrix = getHorizontalMatrix(charactersList_Filtered)
    # questo è un controllo in più che faccio per essere sicuro sul numero di righe
    if len(horizontalMatrix) != getRowsNumber(img_thresh):
        raise Exception()

    verticalMatrix = getVerticalMatrix(charactersList_Filtered)


    # 4 PUZZLE PARAMETERS
    rows = len(horizontalMatrix)
    columns = len(verticalMatrix)

    top, bottom = getMatrixMeanMinMeanMax(horizontalMatrix, 1)
    left, right = getMatrixMeanMinMeanMax(verticalMatrix, 0)

    return PreOCRParameters(rows, columns, left, top, right, bottom)




def OCRComputation(img, preParameters):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 81, 15)
    #img_thresh = frameLib.preprocessingOCRImage(img)

    config = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6"

    left0 = preParameters.left - int(preParameters.meanWidth / 2)
    if (left0 < 0):
        left0 = 0
    right0 = preParameters.left + int(preParameters.meanWidth / 2)
    top0 = preParameters.top - int(preParameters.meanHeight / 2)
    if top0 < 0:
        top0 = 0
    bottom0 = preParameters.top + int(preParameters.meanHeight / 2)

    matrix = []

    for index1 in range(preParameters.rows):
        matrix.append([])

        top1 = top0 + int(preParameters.meanHeight * index1)
        bottom1 = bottom0 + int(preParameters.meanHeight * index1)
        if img.shape[0] < bottom1:
            bottom1 = img.shape[0]

        for index2 in range(preParameters.columns):
            left1 = left0 + int(preParameters.meanWidth * index2)
            right1 = right0 + int(preParameters.meanWidth * index2)
            if (img.shape[1] < right1):
                right1 = img.shape[1]

            img_crop = img[top1: bottom1, left1: right1]

            recognizedChar = False
            boxSize = int(img_crop.shape[0]/2)*2 + 1
            offsetIndex = 2
            while (not recognizedChar and offsetIndex < 10):

                img_thresh = cv2.adaptiveThreshold(img_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, boxSize, offsetIndex * 10)
                d = pytesseract.image_to_string(img_thresh, config=config)

                d = d[0]
                if d != "\f":
                    recognizedChar = True
                else:
                    img_threshInv = img_thresh.copy()
                    img_threshInv[:] = 255 - img_threshInv

                    contoursList, hierarchy = cv2.findContours(img_threshInv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    # questi servono a trovare il rettangolo più grande, nel malaugurato caso ce ne sia più di uno
                    maxContour = []
                    maxArea = 0

                    for contour in contoursList:
                        area = cv2.contourArea(contour)

                        # vogliamo che il rettangolo sia abbastanza grande, e sia il più grande dell'immagine
                        if area > maxArea:
                            perimeter = cv2.arcLength(contour, True)
                            vertexes = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
                            # vogliamo solo quadrilateri
                            if len(vertexes) == 4 or len(vertexes) == 5:
                                maxContour = contour.copy()
                                maxArea = area

                    if len(maxContour) != 0:
                        d = "I"
                        recognizedChar = True
                    else:
                        offsetIndex = offsetIndex + 1
                        '''print(offsetIndex)
                        print(recognizedChar)
                        plot.imshow(img_thresh, "gray")
                        plot.show()'''

            if not recognizedChar:
                d = "_"


            matrix[index1].append( d )



    for row in matrix:
        print(row)

    plot.imshow(img, "gray")
    plot.show()


import glob

paths = glob.glob("../../Sample pictures/*")
for path in paths:
    print("******" + path + "*********")
    img = cv2.imread(path)
    imgWrapper = frameLib.processImage(img)

    resultParameters = OCRPrecomputation(imgWrapper.img_cropped)
    OCRComputation(imgWrapper.img_cropped, resultParameters)