import cv2
import numpy as np
import re


#################################### PUBLIC METHODS#####################################################
'''Costruisco una mappa (lettera, colore in scala di grigio)'''
def getLettersDictionary():
    numbersList = range(0, 26)
    bigNumbersList = [number * 10 for number in numbersList]
    lettersList = [getLetter(number) for number in numbersList]

    lettersDictionary = dict(zip(lettersList, bigNumbersList))

    return lettersDictionary


'''Questa funzione serve a controllare che la stringa inserita dall'utente come parola da cercare sia valida'''
def isWordValid(string):
    if len(string) < 2:
        return False
    return re.search(r"[^A-Za-z]", string) is None


def matrixToGrayscale(lettersDictionary, matrix):

    grayMatrix = np.zeros((len(matrix), len(matrix[0])), np.uint8)
    for rowIndex in range(len(matrix)):
        for columnIndex in range(len(matrix[0])):

            grayMatrix[rowIndex][columnIndex] = int(lettersDictionary[matrix[rowIndex][columnIndex]])

    return grayMatrix



'''Questo è l'algoritmo che cerca una parola dentro l'immagine grigia.
Ritorna la posizione dei due punti agli estremi della parola trovata'''
def findWord(img_BW, word, lettersDictionary):
    # 1 WORD TEMPLATES
    # questo cambio mi serve per le funzioni di match che userò dopo
    img_BW = img_BW.astype(np.uint8)
    word = word.upper()

    # i templates sono le immagini della parola da cercare
    # le maschere mi servono per le parole in diagonale, perchè i template delle parole in diagonale devono essere quadrati
    templates, masks = getTemplatesAndMasks(word, lettersDictionary)


    # 2 MATCH
    # inizio coi primi 6 templates
    for index in range(0, 6):
        template = templates[index].astype(np.uint8)
        mask = masks[index].astype(np.uint8)

        matchResult = cv2.matchTemplate(img_BW, template, cv2.TM_SQDIFF, mask = mask)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matchResult)
        print(min_val)
        if min_val <= 0:
            pointA = min_loc
            pointB = (pointA[0] + template.shape[1] - 1, pointA[1] + template.shape[0] - 1)

            return pointA, pointB

    # le due antidiagonali danno qualche problemino per identificare pointA e pointB, allora le faccio a parte
    for index in range (6, 8):
        template = templates[index].astype(np.uint8)
        mask = masks[index].astype(np.uint8)

        matchResult = cv2.matchTemplate(img_BW, template, cv2.TM_SQDIFF, mask=mask)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matchResult)
        print(min_val)
        if min_val <= 0:
            pointA = (min_loc[0], min_loc[1] + template.shape[0] - 1)
            pointB = (min_loc[0] + template.shape[1] - 1, min_loc[1])

            return [pointA, pointB]

    # non è un vero e proprio file not found
    raise FileNotFoundError


############################################## PRIVATE METHODS ##########################################################
'''Questo serve ad ottenere una lettera giocando coi valori ASCII'''
def getLetter(offset):
    return chr(ord("A") + offset)

'''Serve a costruire i template e le maschere per la funzione di templateMatch.
Riceve in input la parola ed una mappa <lettera, valore>, e restituisce le due liste'''
def getTemplatesAndMasks(word, lettersDictionary):
    # creo un template per ogni possibile orientamento della parola (orizz, vert, diag, anti-diag, al contrario)
    templates = []

    # mettto tutte ste parentesi quadre perchè servono dopo per il np.array
    horizontalColors = [[lettersDictionary[letter] for letter in word]]
    # questo serve a creare un secondo vettore invertito
    horizontalColorsInverse = [row[::-1] for row in horizontalColors]
    verticalColors = [ [lettersDictionary[letter]] for letter in word ]
    verticalColorsinverse = verticalColors[::-1]
    diagonalColors = np.diag([lettersDictionary[letter] for letter in word])
    diagonalColorsInverse = np.diag(*[row[::-1] for row in horizontalColors])
    antiDiagonalColors = np.fliplr(diagonalColors)
    antiDiagonalColorsInverse = np.fliplr(diagonalColorsInverse)

    # horizontal
    templates.append(np.array(horizontalColors))
    templates.append(np.array(horizontalColorsInverse))
    # vertical
    templates.append(np.array(verticalColors))
    templates.append(np.array(verticalColorsinverse))

    masks = [np.ones(template.shape) for template in templates]

    # diagonal
    templates.append(diagonalColors)
    templates.append(diagonalColorsInverse)
    templates.append(antiDiagonalColors)
    templates.append(antiDiagonalColorsInverse)

    eyeMatr = np.eye(len(word))
    masks.append(eyeMatr)
    masks.append(eyeMatr)
    masks.append(np.fliplr(eyeMatr))
    masks.append(np.fliplr(eyeMatr))

    return templates, masks



