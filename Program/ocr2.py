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

img = cv2.imread('../Sample pictures/sample.jpg')
img2 = np.ones((img.shape[0], img.shape[1], 3), np.uint8)
img2[:] = 255

d = pytesseract.image_to_boxes('../Sample pictures/sample.jpg', output_type=Output.DICT)
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
    cv2.putText(img2, c, (left, bottom), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 0, 255))
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
for i in range(0, len(matrix['rows'])):
    sorted_chars = [x for _, x in sorted(zip(matrix['rows'][i]['charHPos'],matrix['rows'][i]['chars']))]
    sorted_pos = [x for x, _ in sorted(zip(matrix['rows'][i]['charHPos'],matrix['rows'][i]['chars']))]
    print(sorted_chars, len(sorted_chars))
    # print(sorted_pos)
    # print(matrix['rowVPos'][i])
    # print(matrix['rows'][i])

cv2.imshow('img', img)
cv2.imshow('img2', img2)
cv2.waitKey(0)