import math
import cv2
import pytesseract
import numpy as np
from pytesseract import Output
from matplotlib import pyplot as plt
try:
    from PIL import Image
except ImportError:
    import Image

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
#print(pytesseract.image_to_string('../Sample pictures/sample.jpg'))
#print(pytesseract.image_to_boxes('../Sample pictures/sample.jpg'))

img_path = '../Sample pictures/sample2.jpg'
img = cv2.imread(img_path)
imgPreFilter = np.ones((img.shape[0], img.shape[1], 3), np.uint8)
imgPreFilter[:] = 255
imgPostFilter = imgPreFilter.copy()

# Example of adding any additional options.
custom_oem_psm_config = r'--psm 6'

d = pytesseract.image_to_boxes(img_path, output_type=Output.DICT, config=custom_oem_psm_config)
# print(d)
n_boxes = len(d['char'])
matrix = {'rows':
             [],
         'rowVPos': []}
for i in range(n_boxes):
    (c, left, top, right, bottom) = (d['char'][i], d['left'][i], img.shape[0]-d['top'][i], d['right'][i], img.shape[0]-d['bottom'][i])
    color = (0, 0, 255)
    y = (top + bottom) / 2
    x = (left + right) / 2
    if not c.isalnum():
        continue
    if c.islower():
        c = c.upper()
    if c == '1':
        c = 'I'
    if c == 'T' and right-left < 4:
        c = 'I'
    if c == '8':
        c = 'S'
    # if c == 'I':
    #     print(c, left, right, top, bottom)

    # cv2.rectangle(img, (left, bottom), (right, top), color, 1)
    cv2.putText(imgPreFilter, c, (left, bottom), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 0, 255))
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
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

print(len(matrix['rows']))
charCount = 0
for i in range(0, len(matrix['rows'])):
    sorted_chars = [x for _, x in sorted(zip(matrix['rows'][i]['charHPos'],matrix['rows'][i]['chars']))]
    sorted_pos = [x for x, _ in sorted(zip(matrix['rows'][i]['charHPos'],matrix['rows'][i]['chars']))]
    # print(sorted_chars, len(sorted_chars))
    # print(sorted_pos)
    matrix['rows'][i]['chars'] = sorted_chars
    matrix['rows'][i]['charHPos'] = sorted_pos
    charCount += len(sorted_chars)
    # print(matrix['rowVPos'][i])
    # print(matrix['rows'][i])
charPerRow = math.floor(charCount / len(matrix['rows']))

scale = 50

charMap = np.ones((len(matrix['rows']), charPerRow, 1), np.uint8)
charMapScaled = np.ones((len(matrix['rows'])*scale, charPerRow*scale, 1), np.uint8)

for i in range(0, len(matrix['rows'])):
    while(len(matrix['rows'][i]['chars']) > charPerRow):
        minDistSides = img.shape[1] * 2
        outlier = charPerRow
        for j in range(1, len(matrix['rows'][i]['chars'])-1):
            distSides = matrix['rows'][i]['charHPos'][j + 1] - matrix['rows'][i]['charHPos'][j - 1]
            if(distSides < minDistSides):
                minDistSides = distSides
                outlier = j
        del matrix['rows'][i]['chars'][outlier]
        del matrix['rows'][i]['charHPos'][outlier]
    print(matrix['rows'][i]['chars'])

for i in range(0, len(matrix['rows'])):
    for j in range(0, len(matrix['rows'][i]['chars'])):
        cv2.putText(imgPostFilter, matrix['rows'][i]['chars'][j], #Image just for developer visualization
                    (math.floor(matrix['rows'][0]['charHPos'][j]),
                        math.floor(matrix['rowVPos'][i])), cv2.FONT_HERSHEY_SIMPLEX, .4, 0)
        charMap[i, j] = (ord(matrix['rows'][i]['chars'][j]) - ord('A')) * 10
        charMapScaled[(i*scale):((i+1)*scale), (j*scale):((j+1)*scale)] = (ord(matrix['rows'][i]['chars'][j]) - ord('A')) * 10



cv2.imshow('img', img)
# cv2.imshow('img2', imgPreFilter)
cv2.imshow('img3', imgPostFilter)
cv2.imshow('charMap', charMap)
cv2.imshow('charMapScaled', charMapScaled)
cv2.waitKey(0)
