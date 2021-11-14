import time

import numpy as np
from PIL import Image
from bm3d import bm3d, BM3DStages, bm3d_rgb
from matplotlib import pyplot as plt


def main():
    # source_image = r'/scratch/p288722/datasets/vision/all_frames/D02_Apple_iPhone4s/' \
    #                r'D02_V_indoor_still_0001/D02_V_indoor_still_0001-00011.png'
    source_image = r'/scratch/p288722/datasets/vision/all_frames/D02_Apple_iPhone4s/' \
                   r'D02_V_indoor_move_0001/D02_V_indoor_move_0001-00021.png'
    # im = cv2.imread(source_image)
    im = Image.open(source_image)
    im = im.resize((800, 480))
    im = np.asarray(im) / 255

    start = time.perf_counter()
    im_denoised = bm3d(im, sigma_psd=0.02)
    end = time.perf_counter()
    print(f'Total time for 1 img {end - start} sec')

    noise = im - im_denoised
    noise = (noise - np.mean(noise)) * 255.0 + 127.5
    noise = np.minimum(np.maximum(noise, 0), 255)
    noise = noise.astype(np.uint8)

    plt.figure(dpi=600)
    plt.imshow(im)
    plt.title('Original Image')
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(dpi=600)
    plt.imshow(np.minimum(np.maximum(im_denoised, 0), 1))
    plt.title('Denoised Image')
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(dpi=600)
    plt.hist(np.ravel(noise), bins=1000, log=True)
    plt.title('Image Noise distribution')
    plt.xlabel('Noise values')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(dpi=600)
    plt.imshow(noise)
    plt.title('Noise')
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(dpi=600)
    plt.imshow(noise[:, :, 0])
    plt.colorbar()
    plt.title('Noise - Channel 0')
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(dpi=600)
    plt.imshow(noise[:, :, 1])
    plt.colorbar()
    plt.title('Noise - Channel 1')
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(dpi=600)
    plt.imshow(noise[:, :, 2])
    plt.colorbar()
    plt.title('Noise - Channel 2')
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
