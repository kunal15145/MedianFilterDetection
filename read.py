import glob as glob


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
