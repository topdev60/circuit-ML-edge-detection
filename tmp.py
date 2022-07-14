import os
import cv2
import numpy as np


def reconstruct():
    print("*********** Rebuilding ***********")
    img = np.zeros((16000, 14529, 3), dtype=np.float32)
    print('Empty Image created...')

    tmp_h = tmp_w = 0

    for iw in range(15):
        tmp_w = iw * 1000
        if iw == 14:
            cv2.imwrite('dump.bmp', img)
        for ih in range(16):
            tmp_h = ih * 1000

            name = 'tmp_{}_{}.bmp'.format(iw, ih)
            print('{} is about to be loaded...'.format(name))
            tmp = cv2.imread(name)
            print('{} is loaded...'.format(name))
            (W, H) = tmp.shape[:2]

            print("cur_H {}\t cur_W {}".format(W, H))
            for i in range(3):
                for w in range(W):
                    for h in range(H):
                        # print('tmp_h {} tmp_w {}'.format(tmp_h + h, tmp_w + w))
                        img[tmp_h + h][tmp_w + w][i] = tmp[h][w][i]
    
    cv2.imwrite("reconstruction_1.bmp", img)


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
    img = cv2.imread("3A_LED600_FFC_ON_001.bmp")
    (H, W) = img.shape[:2]

    for iw in range(int(W/1000)+1):
    # for iw in range(14,15):
        print('Default height and weight is 1000...')
        dft_w = dft_h = 1000

        if (iw+1) * 1000 > W:
            dft_w = W- (iw * 1000)
        for ih in range(int(H/1000)):
            print('cur_w: {}\t cur_h: {}'.format(dft_w, dft_h))
            tmp_img = np.zeros((dft_h, dft_w, 3), dtype=np.float32)
            
            for i in range(3):
                if iw == 14:
                    stop_w = iw * 1000 + dft_w
                else:
                    stop_w = (iw + 1) * 1000
                    for w in range(1000*iw, stop_w):
                        for h in range(1000*ih, 1000*(ih+1)):
                            tmp_w = 1000*iw
                            tmp_h = 1000*ih
                            tmp_img[h-tmp_h][w-tmp_w][i] = img[h][w][i]

            
            name = 'tmp_{}_{}.bmp'.format(iw, ih)

            print(tmp_img.shape)
            blob = cv2.dnn.blobFromImage(tmp_img, scalefactor=1.0, size=(dft_h, dft_w), swapRB=False, crop=True)
            print('Loading {} finished...'.format(name))

            print('*************** [INFO] ***************\n')
            print('Detecting edge of {}...'.format(name))
            net.setInput(blob)
            hed = net.forward()
            hed = cv2.resize(hed[0, 0], (dft_h, dft_w))
            hed = (255 * hed).astype("uint8")
            print('Detecting edge of {} finished...'.format(name))        

            print('Writing image...')
            cv2.imwrite(name, hed)
            print('Edge detection of {} saved...\n'.format(name))

    # reconstruct()

