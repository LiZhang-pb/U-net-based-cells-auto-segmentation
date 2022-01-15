import numpy as np
import cv2
from skimage import exposure
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import rotate
from PIL import Image


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """
    Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


# Define function to draw a grid
def draw_grid(img, grid_size):
    # Draw grid lines
    for x in range(0, img.shape[1], grid_size):
        cv2.line(img, (x, 0), (x, im.shape[0]), color=(255,))
    for y in range(0, img.shape[0], grid_size):
        cv2.line(img, (0, y), (im.shape[1], y), color=(255,))


def augmentation(image, imageB, org_width=256, org_height=256, width=270, height=270):

    max_angle = 360
    image = cv2.resize(image, (height, width))
    imageB = cv2.resize(imageB, (height, width))

    angle = np.random.randint(max_angle)
    if np.random.randint(2):
        angle = -angle
    image = rotate(image, angle, resize=True)
    imageB = rotate(imageB, angle, resize=True)

    xstart = np.random.randint(width - org_width)
    ystart = np.random.randint(height - org_height)
    image = image[xstart:xstart + org_width, ystart:ystart + org_height]
    imageB = imageB[xstart:xstart + org_width, ystart:ystart + org_height]

    if np.random.randint(2):
        image = cv2.flip(image, 1)
        imageB = cv2.flip(imageB, 1)

    if np.random.randint(2):
        image = cv2.flip(image, 0)
        imageB = cv2.flip(imageB, 0)

    image = cv2.resize(image, (org_height, org_width))
    imageB = cv2.resize(imageB, (org_height, org_width))

    return image, imageB


for i in range(0, 30):
    # Load images
    print(i)
    im = cv2.imread('../data/train/%d.png' % i, 0)  # 原始训练集的原始图像的位置
    im_mask = cv2.imread('../data/label/%d_mask.png' % i, 0)  # 原始训练集的标记图像的位置
    # 以下两行每次选一行注释掉
    im = exposure.adjust_gamma(im, random.uniform(1.2, 1.8))  # 调暗
    #im = exposure.adjust_gamma(im, random.uniform(0.6, 0.9))  # 调亮
    a, b = augmentation(im, im_mask)
	# Merge images into separete channels (shape will be (cols, rols, 2))
    im_merge = np.concatenate((im[..., None], im_mask[..., None]), axis=2)
	# Apply transformation on image
    im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.1, im_merge.shape[1] * 0.1)
	# Split image and mask
    im_t = im_merge_t[..., 0]
    im_mask_t = im_merge_t[..., 1]
    origin = Image.fromarray(im_merge_t[..., 0])
    # 下面的120在每次运行时要进行更改，表示保存图像时第一张图的标号
    # 如果把原始数据也算作最后训练集的一组，第一次从30开始，之后依次改为60、90、120、150、180、210、240
    origin.save('../deform/train/%d.png' % (i + 240))  # 形变后的原始图像存放位置
    label = Image.fromarray(im_merge_t[..., 1])
    label.save('../deform/label/%d_mask.png' % (i + 240))  # 形变后的标记图像存放位置
