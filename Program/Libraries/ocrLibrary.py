import math
import cv2
import pytesseract
import numpy as np
from Libraries import frameLibrary as frameLib

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
'''
Expect crucipuzzle cropped opencv image
Returns matrix of grayscale pixels, each pixel corresponds to a letter, check wordLibrary.py for further details
'''
def getCharMap(img=None, debug=False):
    d, img = ImgToPytesseractDict(img)


    if debug:
        # Used only for developer visualization
        imgPreFilter = np.ones((img.shape[0], img.shape[1], 3), np.uint8)
        imgPreFilter[:] = 255
        imgPostFilter = imgPreFilter.copy()
        # print(d)
        matrix, charPerRow = ReorderDictIntoMatrix(d, img, imgPreFilter, debug)
    else:
        matrix, charPerRow = ReorderDictIntoMatrix(d, img)


    matrix = RemoveOutliers(matrix, charPerRow, img)

    charMap = np.ones((len(matrix['rows']), charPerRow, 1), np.uint8)
    if debug:
        scale = 50
        charMapScaled = np.ones((len(matrix['rows']) * scale, charPerRow * scale, 1), np.uint8)

    # Map char matrix to pixel matrix
    for i in range(0, len(matrix['rows'])):
        for j in range(0, len(matrix['rows'][i]['chars'])):
            charMap[i, j] = (ord(matrix['rows'][i]['chars'][j]) - ord('A')) * 10
            if debug:
                cv2.putText(imgPostFilter, matrix['rows'][i]['chars'][j],  # Image just for developer visualization
                            (math.floor(matrix['rows'][0]['charHPos'][j]),
                             math.floor(matrix['rowVPos'][i])), cv2.FONT_HERSHEY_SIMPLEX, .4, 0)
                charMapScaled[(i * scale):((i + 1) * scale), (j * scale):((j + 1) * scale)] = (ord(
                    matrix['rows'][i]['chars'][j]) - ord('A')) * 10
    if debug:
        cv2.imshow('img', img)
        cv2.imshow('img2', imgPreFilter)
        cv2.imshow('img3', imgPostFilter)
        cv2.imshow('charMap', charMap)
        cv2.imshow('charMapScaled', charMapScaled)
        cv2.waitKey(0)

    return charMap


'''
Convert a cropped img into pytesseract char+bounding rect dictionary
In case no image is passed it reads a default one
Returns dictionary and image
'''


def ImgToPytesseractDict(img):
    if img is None:
        img_path = '../../Sample pictures/sample8.jpg'
        img = cv2.imread(img_path)
        # img = frameLib.processImage(img).img_cropped
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('img', img)

    custom_oem_psm_config = r'--psm 6'

    d = pytesseract.image_to_boxes(img, output_type=pytesseract.Output.DICT, config=custom_oem_psm_config)
    return d, img


'''
Receive a pytesseract dictionary (chars+bounding rect coordinates), opencv image, opencv image blank copy (optional)
Returns a dictionary with a list of Y coordinate for each row, a list of rows, in which there are char and X coordinate for each char
If the blank copy is passed it is filled with recognized characters
'''
def ReorderDictIntoMatrix(d, img, imgPreFilter = None, debug = False):
    n_boxes = len(d['char'])
    matrix = {'rows': [],
              'rowVPos': []}
    for i in range(n_boxes):
        (c, left, top, right, bottom) = (
            d['char'][i], d['left'][i], img.shape[0] - d['top'][i], d['right'][i], img.shape[0] - d['bottom'][i])
        color = (0, 0, 255)
        y = (top + bottom) / 2
        x = (left + right) / 2
        if debug:
            print(c, left, top, right, bottom)
        if c == '!':
            c = 'I'
        if not c.isalnum():
            continue
        if c.islower():
            c = c.upper()
        if c == '1':
            c = 'I'
        if c == 'T' and right - left < (bottom - top) / 2:
            c = 'I'
        if c == 'L' and right - left < (bottom - top) / 2:
            c = 'I'
        if c == '8':
            c = 'S'
        if not c.isalpha():
            continue

        if debug:
            cv2.rectangle(img, (left, bottom), (right, top), color, 1)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
        if imgPreFilter is not None:
            cv2.putText(imgPreFilter, c, (left, bottom), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 0, 255))
            # cv2.imshow('img2', imgPreFilter)
            # cv2.waitKey(0)

            # print(c)
            # print(img.shape[0])

        inserted = False
        for j in range(0, len(matrix['rowVPos'])):
            # print('y ' + y.__str__())
            # print('top ' + top.__str__())
            # print('bottom ' + bottom.__str__())
            # print('matrix ' + matrix['rowVPos'][j].__str__() + ' j ' + j.__str__())
            if top < matrix['rowVPos'][j] < bottom:
                inserted = True
                # print('inserted')
                matrix['rows'][j]['chars'].append(c)
                matrix['rows'][j]['charHPos'].append(x)
                # print(matrix['rows'])
            else:
                # print('not inserted')
                c = c
        if not inserted:
            # print('y ' + y.__str__())
            # print('top ' + top.__str__())
            # print('bottom ' + bottom.__str__())
            # print('matrix ' + len(matrix['rowVPos']).__str__())
            matrix['rowVPos'].append(y)
            matrix['rows'].append({'chars': [c], 'charHPos': [x]})
            # print(matrix['rows'])

    if debug:
        cv2.imshow('img', img)
        cv2.waitKey(0)
        print(len(matrix['rows']))

    charCount = 0
    for i in range(0, len(matrix['rows'])):
        sorted_chars = [x for _, x in sorted(zip(matrix['rows'][i]['charHPos'], matrix['rows'][i]['chars']))]
        sorted_pos = [x for x, _ in sorted(zip(matrix['rows'][i]['charHPos'], matrix['rows'][i]['chars']))]
        # print(sorted_chars, len(sorted_chars))
        # print(sorted_pos)
        matrix['rows'][i]['chars'] = sorted_chars
        matrix['rows'][i]['charHPos'] = sorted_pos
        charCount += len(sorted_chars)
        if debug:
            print(matrix['rowVPos'][i])
            print(matrix['rows'][i])
    charPerRow = math.floor(charCount / len(matrix['rows']))
    return matrix, charPerRow

'''
Receives a matrix in the format returned in ReorderDictIntoMatrix
Removes from each row that has more characters than other rows the character (or characters) that is too close to other characters
Returns polished matrix
'''
def RemoveOutliers(matrix, charPerRow, img, debug = False):
    for i in range(0, len(matrix['rows'])):
        while len(matrix['rows'][i]['chars']) > charPerRow:
            minDistSides = img.shape[1] * 2
            outlier = charPerRow
            for j in range(1, len(matrix['rows'][i]['chars']) - 1):
                distSides = matrix['rows'][i]['charHPos'][j + 1] - matrix['rows'][i]['charHPos'][j - 1]
                if distSides < minDistSides:
                    minDistSides = distSides
                    outlier = j
            del matrix['rows'][i]['chars'][outlier]
            del matrix['rows'][i]['charHPos'][outlier]
        if debug:
            print(matrix['rows'][i]['chars'])
    return matrix

getCharMap(None, True)
