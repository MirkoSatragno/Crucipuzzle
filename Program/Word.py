import cv2
import numpy as np
import matplotlib.pyplot as plot
from Libraries import wordLibrary


def printPicture(printPicture_img, printPicture_mode="default"):
    code[0] += 1
    plot.subplot(printPicture_rows, printPicture_cols, code[0])
    if (printPicture_mode != "default"):
        plot.imshow(printPicture_img, cmap=printPicture_mode)
    else:
        plot.imshow(printPicture_img)
    # this is to hide "rulers"
    plot.xticks([]), plot.yticks([])


# 0 INPUT
lettersDictionary = wordLibrary.getLettersDictionary()

word = "dog"
img_BW = np.array([[50, 70, 120, 30, 240, 150],
                   [20, 70, 250, 80, 90, 100],
                   [30, 140, 50, 130, 70, 150],
                   [30, 140, 60, 100, 160, 50],
                   [210, 60, 40 , 200, 250, 20]])


# 1 FIND WORD
img_final = cv2.merge((img_BW, img_BW, img_BW))
try:
    pointA, pointB = wordLibrary.findWord(img_BW, word, lettersDictionary)
    cv2.line(img_final, pointA, pointB, (0, 255, 255), 1)
except Exception:
    print("Oh no!")


############# PRINT ################
# qui è il mio pigro-originale modo per stampare tante immagini assieme
# questi vengono usati come parametri dal metodo printPicture
printPicture_rows = 1
printPicture_cols = 2
code = [0]

printPicture(img_BW, "gray")
printPicture(img_final)

plot.show()
# cv2.waitKey(0)
cv2.destroyAllWindows()
