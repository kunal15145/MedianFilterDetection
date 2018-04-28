import glob as glob
import cv2
import scipy.ndimage as scn
from sklearn.model_selection import train_test_split


def readdata(filtersize):
    global n, m
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

    return [data_dict, m, n]


def data_split(data):

    images = data[0]
    fulldata = []
    fulllabels = []
    for classvalue,im in images.items():
        for i in range(len(im)):
            fulldata.append(im[i])
            fulllabels.append(classvalue)

    X_train, X_test, y_train, y_test = train_test_split(fulldata, fulllabels, test_size=0.30, random_state=42)
    return [X_train, X_test, y_train, y_test]


def preprecessing_MFR(data):
    size_m = data[1]
    size_n = data[2]
    images = data[0]

    for classvalue, imagesdata in images.items():
        for im in range(len(imagesdata)):
            finalimage = imagesdata[im].reshape((size_m, size_n))
            denoised = scn.median_filter(finalimage, 3)
            mfrimage = denoised - finalimage
            mfrimage = mfrimage.reshape((1, size_m * size_n))
            imagesdata[im] = mfrimage

    return [images, size_m, size_n]
