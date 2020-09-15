import cv2
import pytesseract
import statistics
import matplotlib.pyplot as plot

from Libraries import frameLibrary as frameLib

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

def wordsList(data):
    words = [w for w in data.split("\n")]
    words = list(filter(lambda w: w != "\f", words))

    mean = statistics.mean([len(word) for word in words])
    threshold = int(mean * 0.5)
    words = list(filter(lambda word: len(word) > threshold, words))

    return words

def getCharactersLists(img):
    config = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6"
    data = pytesseract.image_to_boxes(img, config=config)

    data_split = list([row] for row in data.split("\n"))
    data_split_filtered = list(filter(lambda lis: lis[0] != "", data_split))
    data_matrix = [[element for element in row[0].split(" ")] for row in data_split_filtered]

    lists = [ CharacterWrapper(el[0], int(el[1]), img.shape[0] - int(el[2]), int(el[3]), img.shape[0] - int(el[4])) for el in data_matrix ]

    return lists

def filterBySize(data):

    # le I son troppo piccole
    data_noI = list(filter(lambda el:  el.char != "I", data))

    data_areas = list(el.width * el.height for el in data_noI)
    meanArea = statistics.mean(data_areas)
    data_filteredAreas = list(filter(lambda el: el.width * el.height < meanArea * 1.5, data_noI))
    newData_areas =  list(el.width * el.height for el in data_filteredAreas)
    newMeanArea = statistics.mean(newData_areas)

    while meanArea != newMeanArea:
        meanArea = newMeanArea

        data_filteredAreas = list(filter(lambda el: meanArea * 0.5 < el.width * el.height < meanArea * 1.5, data_filteredAreas))
        newData_areas = list(el.width * el.height for el in data_filteredAreas)
        newMeanArea = statistics.mean(newData_areas)

    data_width = list(el.width for el in data_filteredAreas)
    meanWidth = statistics.mean(data_width)
    data_filteredWidth  = list(filter(lambda el: meanWidth * 0.7 <el.width < meanWidth * 1.4 , data_filteredAreas))
    newdataWidth = list(el.width for el in data_filteredWidth)
    newMeanWidth = statistics.mean(newdataWidth)

    while meanWidth != newMeanWidth:
        meanWidth = newMeanWidth

        data_filteredWidth = list(filter(lambda el: meanWidth * 0.7 < el.width < meanWidth * 1.4, data_filteredWidth))
        newdataWidth = list(el.width for el in data_filteredWidth)
        newMeanWidth = statistics.mean(newdataWidth)

    return data_filteredWidth

def isSameColumn(el1, el2):
    if el2.left < el1.pos[0] < el2.right:
        return True

    return False

def getHorizontalMatrix(lists):
    horizMatrix = []
    index = 0
    previousEl = None
    for el in lists:
        if len(horizMatrix) == 0:
            horizMatrix.append([el])
            previousEl = el
            continue

        if previousEl.pos[0] < el.pos[0]:
            horizMatrix[index].append(el)
            previousEl = el
        else:
            horizMatrix.append([el])
            previousEl = el
            index = index + 1

    maxLen = max([len(row) for row in horizMatrix])
    horizMatrix = list(filter(lambda row: maxLen * 0.5 < len(row), horizMatrix))

    return horizMatrix

def getVerticalMatrix(lists):
    vertMatrix = []
    for el in lists:
        found = False
        for matrixIndex in range(len(vertMatrix)):
            lastIndex = len(vertMatrix[matrixIndex]) - 1
            if isSameColumn(el, vertMatrix[matrixIndex][lastIndex]) and not found:
                vertMatrix[matrixIndex].append(el)
                found = True

        if not found:
            vertMatrix.append([el])

    maxLen = max([len(col) for col in vertMatrix])
    vertMatrix = list(filter(lambda col: maxLen * 0.5 < len(col), vertMatrix))

    return vertMatrix

def getMatrixMeanMinMeanMax(matrix, index):
    values = [ statistics.mean([ el.pos[index] for el in array]) for array in matrix]
    values = sorted(values)

    return int(min(values)), int(max(values))

def OCRPrecomputation(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 81, 15)
    #img_thresh = frameLib.preprocessingOCRImage(img)

    config = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6"
    d = pytesseract.image_to_string(img_thresh, config=config)

    #serve giusto da check aggiuntivo qualche riga più sotto
    horizWords = wordsList(d)
    rowsNumber = len(horizWords)

    # FILTER
    lists = getCharactersLists(img_thresh)
    lists_filtered = filterBySize(lists)

    newImg = img_thresh.copy()
    for el in lists_filtered:
        cv2.rectangle(newImg, (el.left, el.top), (el.right, el.bottom), (0, 0, 255), 3)
        #cv2.putText(newImg, el.char, (el.pos[0], el.pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))


    # MATRIXES
    horizMatrix = getHorizontalMatrix(lists_filtered)
    # se fallisce già questo è perchè il puzzle è proprio una porcheria ed è meglio se mi fermo
    if len(horizMatrix) != rowsNumber:
        raise Exception()

    vertMatrix = getVerticalMatrix(lists_filtered)


    # FINAL PARAMETERS
    rows = len(horizMatrix)
    columns = len(vertMatrix)

    top, bottom = getMatrixMeanMinMeanMax(horizMatrix, 1)
    left, right = getMatrixMeanMinMeanMax(vertMatrix, 0)

    result = PreOCRParameters(rows, columns, left, top, right, bottom)

    return result




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