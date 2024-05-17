import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

d_path = "RGBD_ModelNet40/airplane/test/airplane_0627_r_000_depth0001.png"
rgb_path = "RGBD_ModelNet40/airplane/test/airplane_0627_r_000.png"


def load_rgbd_image(rgb_png_image_path: str) -> np.ndarray[np.uint8]:
    """Given a path to an rgb .png combines it and a corresponding depth
    image into a sigle np.ndarray. The first three channels store RGB data,
    and the last one - depth information.
    """
    d_image_path = rgb_png_image_path.rstrip(".png")+"_depth0001.png"
    rgb_im_arr = np.array(Image.open(rgb_png_image_path))
    d_im_arr = np.array(Image.open(d_image_path))
    rgb_im_arr[:, :, 3] = d_im_arr  # replace alpha channel with depth data
    return rgb_im_arr

def load_d_image(rgb_png_image_path: str) -> np.ndarray[np.uint8]:
    """Given a path to an rgb .png returns a corresponding depth
    image as an np.ndarray
    """
    d_image_path = rgb_png_image_path.rstrip(".png")+"_depth0001.png"
    d_im_arr = np.array(Image.open(d_image_path))
    return d_im_arr


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
