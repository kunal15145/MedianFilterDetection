import glob as glob
import cv2


def readdata(filtersize):
    path = "dataset/jpeg45/"
    path1 = "dataset/jpeg90/"
    path2 = ""
    path3 = ""
    if filtersize == 3:
        path2 = "dataset/medr3_45/"
        path3 = "dataset/medr3_90/"

    elif filtersize == 5:
        path2 = "dataset/medr5_45/"
        path3 = "dataset/medr5_90/"

    # 0 class non medianfilterd 1 class median filtered
    data_dict = {0: [], 1: []}

    for imagepath in glob.glob(path + "*"):
        image = cv2.imread(imagepath, 0)
        m, n = image.shape
        imagen = image.reshape((1, m * n))
        data_dict[0].append(imagen)

    for imagepath in glob.glob(path1 + "*"):
        image = cv2.imread(imagepath, 0)
        m, n = image.shape
        imagen = image.reshape((1, m * n))
        data_dict[0].append(imagen)

    for imagepath in glob.glob(path2 + "*"):
        image = cv2.imread(imagepath, 0)
        m, n = image.shape
        imagen = image.reshape((1, m * n))
        data_dict[1].append(imagen)

    for imagepath in glob.glob(path3 + "*"):
        image = cv2.imread(imagepath, 0)
        m, n = image.shape
        imagen = image.reshape((1, m * n))
        data_dict[1].append(imagen)

    return data_dict