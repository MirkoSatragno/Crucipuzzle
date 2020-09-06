import cv2
import numpy as np
import matplotlib.pyplot as plot
from Libraries import frameLibrary as frameLib


def printPicture(printPicture_img, printPicture_mode="default"):
    code[0] += 1
    plot.subplot(printPicture_rows, printPicture_cols, code[0])
    if (printPicture_mode != "default"):
        plot.imshow(printPicture_img, cmap=printPicture_mode)
    else:
        plot.imshow(printPicture_img)
    # this is to hide "rulers"
    plot.xticks([]), plot.yticks([])


# 1 IMPORT
img_BGR = cv2.imread("../Sample pictures/10 - Copy.jpg")


# 2 COLORS
img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
img_BW = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)


# 3 VERTEXES
# questa è la posizione dei 4 vertici del quadrilatero originale
sortedVertexes = frameLib.getFrameVertexes(img_BW)
# questa è la nuova posizione dei vertici che mi server per fare il wrap
newSortedVertexes = frameLib.getNewFrameVertexes(sortedVertexes)


# 4 AFFINE IMAGE
# stiracchio l'immagine, per raddrizzare un po' il quadrilatero centrale
img_perspective = frameLib.getWarpedImage(img_RGB, sortedVertexes, newSortedVertexes)

# Qui se voglio posso fare un crop dell'area interessata.
# La cosa migliore sarebbe farla prendendo il bounding box di quella figura composta da punti usata nella getFrameVertexes, per evitare lati bombati all'infuori tagliati
# ma ce la facciamo andar bene così. Se ci sono bordi bombati dopo il warp allora l'immagine faceva proprio schifo
'''x0, y0, width, height = cv2.boundingRect(originalPoints)
img_cropped = img_RGB[y0: y0 + height, x0: x0 + width]'''


# 5 LINEE
# provo a disegnare linee a caso
img_perspective_lines = img_perspective.copy()
cv2.line(img_perspective_lines, (1000, 500), (2500, 500), (0, 255, 0), 30)
cv2.line(img_perspective_lines, (900, 2000), (2800, 2000), (0, 255, 0), 30)
cv2.line(img_perspective_lines, (1050, 700), (1050, 2200), (0, 255, 0), 30)


# 6 WARP INVERSO
img_lines = frameLib.getInverseWarpedImage(img_RGB, img_perspective_lines, newSortedVertexes, sortedVertexes)

############# PRINT ################
# qui è il mio pigro-originale modo per stampare tante immagini assieme
# questi vengono usati come parametri dal metodo printPicture
printPicture_rows = 2
printPicture_cols = 2
code = [0]

printPicture(img_RGB)
# printPicture(img_BW, "gray")
printPicture(img_perspective)
printPicture(img_perspective_lines)
printPicture(img_lines)

plot.show()
# cv2.waitKey(0)
cv2.destroyAllWindows()
