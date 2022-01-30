from matplotlib import pyplot as plt
import cv2
import prnu


def main():
    source_image = r'/scratch/p288722/datasets/vision/all_frames/D02_Apple_iPhone4s/' \
                   r'D02_V_indoor_still_0001/D02_V_indoor_still_0001-00011.png'
    im = cv2.imread(source_image)
    im_prnu = prnu.extract_single(im)

    plt.figure()
    plt.imshow(im)
    plt.show()
    plt.close()

    plt.figure()
    plt.hist(im_prnu.ravel(), bins=1000, log=True)
    plt.ylabel('Count (log scale)')
    plt.xlabel('PRNU noise values')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
