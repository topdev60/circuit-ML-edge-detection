import os
import cv2
import numpy as np


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
    img = cv2.imread("3B_LED600_FFC_ON_001.bmp")
    (H, W) = img.shape[:2]

    for iw in range(14,15):
        print('Default height and weight is 1000...')
        dft_w = dft_h = 1000

        if (iw+1) * 1000 > W:
            dft_w = W- (iw * 1000)
        for ih in range(int(H/1000)):
            if (ih+1) * 1000 > H:
                dft_h = H-(ih * 1000)

            print('cur_w: {}\t cur_h: {}'.format(dft_w, dft_h))
            tmp_img = np.zeros((dft_h, dft_w, 3), dtype=np.float32)
            
            for i in range(3):
                if dft_w == 1000:
                    stop_w =1000 * (iw+1)
                else: 
                    stop_w = 1000*iw+dft_w

                for w in range(1000*iw, stop_w):
                    if dft_h == 1000:
                        stop_h =1000 * (ih+1)
                    else: 
                        stop_h = 1000*ih+dft_h

                    for h in range(1000*ih, stop_h):
                        tmp_w = 1000*iw
                        tmp_h = 1000*ih
                        tmp_img[h-tmp_h][w-tmp_w][i] = img[h][w][i]
            
            name = 'tmp_{}_{}.bmp'.format(iw, ih)

            print(tmp_img.shape)
            blob = cv2.dnn.blobFromImage(tmp_img, scalefactor=1.0, size=(dft_w, dft_h), swapRB=False, crop=True)
            print('Loading {} finished...'.format(name))

            print('*************** [INFO] ***************\n')
            print('Detecting edge of {}...'.format(name))
            net.setInput(blob)
            hed = net.forward()
            hed = cv2.resize(hed[0, 0], (dft_w, dft_h))
            hed = (255 * hed).astype("uint8")
            print('Detecting edge of {} finished...'.format(name))        

            print('Writing image...')
            cv2.imwrite(name, hed)
            print('Edge detection of {} saved...\n'.format(name))