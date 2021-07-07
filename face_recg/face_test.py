import cv2


def face_test(impath):
    image = cv2.imread(impath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')
    fa = face_cade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5)

    if not fa:
        return False
    else:
        for (x, y, w, h) in fa:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255.0), 2)
        cv2.imwrite('./result/cv_final.png', image)
        return True


if __name__ == '__main__':
    impath = "../op_boundary/result/face.png"
    face_test(impath)
