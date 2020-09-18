import cv2
import pytesseract
import statistics
import matplotlib.pyplot as plot

from Libraries import frameLibrary as frameLib

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'



################################## PUBLIC METHODS ###############################################


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
    # trasformo la lista di caratteri in 2 matrici di caratteri: una lista di righe ed una lista di colonne
    # Di nuovo, non ci sono tutti i caratteri, ma solo quelli che ho filtrato prima
    horizontalMatrix = getHorizontalMatrix(charactersList_Filtered)
    verticalMatrix = getVerticalMatrix(charactersList_Filtered)


    # 4 PUZZLE PARAMETERS
    rows = getRowsNumber(img_thresh)
    columns = len(verticalMatrix)

    top, bottom = getMatrixMeanMinMeanMax(horizontalMatrix, 1)
    left, right = getMatrixMeanMinMeanMax(verticalMatrix, 0)

    return PreOCRParameters(rows, columns, left, top, right, bottom)



'''Questa è la vera e propria funzione che si occupa di riconoscere i caratteri del puzzle.
Riceve in input l'immagine ed un oggetto che contiene dei parametri estrapolati grazie alla funzione di OCRPrecomputation.
Il risultato è una matrice di caratteri'''
def OCRComputation(img_BGR, preParameters):

    '''La funzione opera il riconoscimento di caratteri lavorando su di loro singolarmente.
    Ogni immagine di carattere viene ritagliata e poi con pytesseract si prova a riconoscere il contenuto.
    Riusciamo a ritagliare con buona precisione i singoli caratteri grazie al risultato del preprocessing'''
    img_BW = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)

    # preparo i parametri base per ritagliare una a una le immagini delle singole lettere da riconoscere
    left0 = preParameters.left - int(preParameters.meanWidth / 2)
    right0 = preParameters.left + int(preParameters.meanWidth / 2)
    top0 = preParameters.top - int(preParameters.meanHeight / 2)
    bottom0 = preParameters.top + int(preParameters.meanHeight / 2)

    matrix = []

    # con un doppio ciclo riempo la matrice coi caratteri riconosciuti uno a uno
    for rowIndex in range(preParameters.rows):
        matrix.append([])

        # non faccio una somma incrementale ma me lo moltiplico ogni volta per evitare errori di approssimazione castando a int
        top1 = max(top0 + int(preParameters.meanHeight * rowIndex), 0)
        bottom1 = min(bottom0 + int(preParameters.meanHeight * rowIndex), img_BW.shape[0])

        for columnIndex in range(preParameters.columns):
            left1 = max(left0 + int(preParameters.meanWidth * columnIndex), 0)
            right1 = min(right0 + int(preParameters.meanWidth * columnIndex), img_BW.shape[1])

            img_crop = img_BW[top1: bottom1, left1: right1]
            char = recognizeCharacter(img_crop)
            matrix[rowIndex].append(char)


    return matrix





################################## PRIVATE METHODS ###############################################
# in realtà non sono privati. E' una distinzione estetica

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



'''Questa funzione riceve in input l'immagine in scala di grigi di un singolo carattere e deve riconoscerlo. L'output è il carattere riconosciuto'''
def recognizeCharacter(img_char):
    config = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6"

    # serve giusto ad avere una size di dimensione dispari
    thresholdBoxSize = int(img_char.shape[0] / 2) * 2 + 1
    # questo valore serve a "filtrare" il risultato del thresholding (circa. Vedi documentazione di opencv per la descrizione seria)
    offsetValue = 2

    # faccio un po' di tentativi di riconoscimento adattivo cambiando parametri. Se dopo un po' di tentativi non riconosco nulla mi arrendo
    while offsetValue < 10:

        img_thresh = cv2.adaptiveThreshold(img_char, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, thresholdBoxSize, offsetValue * 10)
        d = pytesseract.image_to_string(img_thresh, config = config)

        if d != "\f":
            # controllo di non aver riconosciuto più di un carattere. Nel qual caso continuo col while finchè non ne riconosco esattamente solo 1
            characters = d.split("\n")[0]
            if len(characters) == 1:
                return characters[0]
            if 1 < len(characters):
                offsetValue = offsetValue + 1
                continue


        '''se fallisco nel trovare un carattere generico mi concentro sulla "I", 
        perchè è quella più difficile da riconoscere con pytesseract, ma la più facile da riconoscere "a mano"'''
        # inverto bianco/nero perchè pytesseract lavora col nero, opencv col bianco
        img_thresh[:] = 255 - img_thresh

        contoursList, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contoursList:
            area = cv2.contourArea(contour)
            if 0 < area:
                perimeter = cv2.arcLength(contour, True)
                vertexes = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
                # vogliamo solo quadrilateri, o simili
                if len(vertexes) == 4 or len(vertexes) == 5:
                    return "I"


        # se ho fallito nel riconoscere anche la "I" ci riprovo con un nuovo thresholding
        offsetValue = offsetValue + 1

    # se arrivo alla fine senza aver riconosciuto nessun carattere lancio un'eccezione
    raise Exception

