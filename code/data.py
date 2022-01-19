from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2
OUT_ROWS = 256  # 确定模型要识别的图像的高和宽,此处设置为256像素*256像素
OUT_COLS = 256


class dataProcess(object):
	def __init__(self, out_rows, out_cols, data_path="../deform/train/", label_path="../deform/label/",
				 test_path="../test/", npy_path="../npydata/", img_type="png"): # ALL addresses you need to set
		"""
		A class used to process data.
		"""
		self.out_rows = out_rows
		self.out_cols = out_cols
		self.data_path = data_path
		self.label_path = label_path
		self.img_type = img_type
		self.test_path = test_path
		self.npy_path = npy_path

	# 创建训练集的npy文件
	def create_train_data(self):
		print('-'*30)
		print('Creating training images...')
		print('-'*30)

		imgs = os.listdir(self.data_path)
		total = len(imgs)
		print(total)
		imgdatas = np.ndarray((total, self.out_rows, self.out_cols, 1), dtype=np.uint8)
		imglabels = np.ndarray((total, self.out_rows, self.out_cols, 1), dtype=np.uint8)

		i = 0
		for i in range(total):
			imgname = str(i) + "." + self.img_type
			labelname = str(i) + "_mask." + self.img_type
			img = cv2.imread(os.path.join(self.data_path, imgname), cv2.IMREAD_GRAYSCALE)
			img = cv2.resize(img, (OUT_ROWS, OUT_COLS))
			label = cv2.imread(os.path.join(self.label_path, labelname), cv2.IMREAD_GRAYSCALE)
			label = cv2.resize(label, (OUT_ROWS, OUT_COLS))
			imgdatas[i, :, :, 0] = img
			imglabels[i, :, :, 0] = label
			if i % 10 == 0:
				print('Done: {0}/{1} images'.format(i, total))
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_train.npy', imgdatas)
		np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
		print('Saving to .npy files done.')
		return i

	# 创建测试集的npy文件
	def create_test_data(self):
		print('-'*360)
		print('Creating test images...')
		print('-'*360)

		imgs = os.listdir(self.test_path)
		total = len(imgs)
		print(total)
		imgdatas = np.ndarray((total, self.out_rows, self.out_cols, 1), dtype=np.uint8)

		i = 0
		for i in range(total):
			imgname = str(i) + "." + self.img_type
			img = cv2.imread(os.path.join(self.test_path, imgname), cv2.IMREAD_GRAYSCALE)
			img = cv2.resize(img, (OUT_ROWS, OUT_COLS))
			imgdatas[i, :, :, 0] = img
			if i % 10 == 0:
				print('Done: {0}/{1} images'.format(i, total))
			i += 1
		print('loading done')

		np.save(self.npy_path + '/imgs_test.npy', imgdatas)
		print('Saving to imgs_test.npy files done.')
		return i

	# 读取训练集的npy文件
	def load_train_data(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)
		imgs_train = np.load(self.npy_path+"/imgs_train.npy")
		imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train.npy")
		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		imgs_train /= 255
		mean = imgs_train.mean(axis=0)
		imgs_train -= mean
		imgs_mask_train /= 255
		imgs_mask_train[imgs_mask_train > 0.5] = 1
		imgs_mask_train[imgs_mask_train <= 0.5] = 0
		return imgs_train, imgs_mask_train

	# 读取测试集的npy文件
	def load_test_data(self):
		print('-'*30)
		print('load test images...')
		print('-'*30)
		imgs_test = np.load(self.npy_path+"/imgs_test.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255
		mean = imgs_test.mean(axis=0)
		imgs_test -= mean
		return imgs_test


if __name__ == "__main__":
	mydata = dataProcess(OUT_ROWS, OUT_COLS)
	mydata.create_train_data()
	mydata.create_test_data()
