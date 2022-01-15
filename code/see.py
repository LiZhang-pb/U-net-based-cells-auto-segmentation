import numpy as np
import cv2
'''将训练结果从npy格式转换成png格式的图片'''
def see(img_path, mask_path, save_path):
    image = np.load(img_path)
    mask = np.load(mask_path)
    for i in range(0, image.shape[0]):
        cv2.imwrite(save_path + str(i) + ".png", image[i, :, :])
        cv2.imwrite(save_path + str(i) + "_mask.png", mask[i, :, :])
        #测试imshow
        #cv2.imshow(save_path + str(i) + ".png", image[i, :, :])
        #cv2.imshow(save_path + str(i) + "_mask.png", mask[i, :, :])
        print(str(i), "is saved.")
    print("All works are finished.")


if __name__ == '__main__':
    img_path = "../npydata/imgs_test.npy"
    mask_path = "../npydata/imgs_mask_test.npy"
    save_path = "../results/"
    see(img_path, mask_path, save_path)
