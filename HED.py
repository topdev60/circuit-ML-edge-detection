import os
import cv2


if __name__ == '__main__':
    print('Current Directory is {}'.format(os.getcwd()))

    print('*************** [INFO] ***************\n')
    print('Loading HED...')
    protoPath = os.getcwd() + '\\HED\\' + 'deploy.prototxt'
    modelPath = os.getcwd() + '\\HED\\' + 'hed_pretrained_bsds.caffemodel'
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    print('Loading HED finished...\n')

    print('*************** [INFO] ***************\n')
    print('Loading image...')
    # img = cv2.imread("larry.jpg")
    img = cv2.imread("tmp_2_2.bmp")

    (H, W) = img.shape[:2]
    # blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(W, H), swapRB=False, crop=False)
    print(img.shape)
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(W, H), swapRB=False, crop=False)

    print('Loading image finished...')

    print('*************** [INFO] ***************\n')
    print('Detecting edge...')
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype("uint8")
    print('Detecting edge finished...')

    cv2.imshow("Input", img)
    cv2.imshow("HED", hed)
    cv2.imwrite('first.bmp', hed)
    cv2.waitKey(0)