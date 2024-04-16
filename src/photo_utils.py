import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

d_path = "RGBD_ModelNet40/airplane/test/airplane_0627_r_000_depth0001.png"
rgb_path = "RGBD_ModelNet40/airplane/test/airplane_0627_r_000.png"

if __name__ == "__main__":
    d_im_arr = np.array(Image.open(d_path))
    rgb_im_arr = np.array(Image.open(rgb_path))
    print(d_im_arr.dtype)
    # has 4 channels - all uint8
    # 4th channel - alpha values; 0 corresponds to full transparency
    print(rgb_im_arr.shape)

    print(np.unique(rgb_im_arr[:, :, 3], return_counts=True))
    alpha_img = rgb_im_arr[:, :, 3]
    print(alpha_img.shape)
    plt.imshow(alpha_img)
    plt.colorbar()
    plt.show()
