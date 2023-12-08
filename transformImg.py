def transformImg(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w = img.shape[1]
    h = img.shape[0]
    min = 255
    for i in range(0, h):
        for j in range(0, w):
            if img_gray[i, j] < min:
                min = img_gray[i, j]
    for i in range(0, h):
        for j in range(0, w):
            if img_gray[i, j] <= (min + 20):
                img[i, j] = 0, 0, 0
            else:
                img[i, j] = 255, 255, 255