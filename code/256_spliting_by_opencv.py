import cv2
import glob
'''Cut images into 256*256 将目标图片裁剪为256*256 像素'''
i = 0


def crop(img, outdir):
    img = cv2.imread(img)  # 读入图片
    #Row1
    cropped = img[0:256, 0:256]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite("../corpped_PNG/{}.png".format(i+11000), cropped)  # 裁剪并存储在指定文件夹中
    # 图片名110xx,第xx张照片的第1行第1列
    # 例如11035，文件夹中第35张照片的剪裁出的第1行第1列
    cropped = img[0:256, 256:512]
    cv2.imwrite("../corpped_PNG/{}.png".format(i+12000), cropped)
    cropped = img[0:256, 512:768]
    cv2.imwrite("../corpped_PNG/{}.png".format(i + 13000), cropped)
    cropped = img[0:256, 768:1024]
    cv2.imwrite("../corpped_PNG/{}.png".format(i + 14000), cropped)
    cropped = img[0:256, 1024:1280] # 是否有这一行要看照片尺寸是否够大
    cv2.imwrite("../corpped_PNG/{}.png".format(i + 15000), cropped)
    # Row2
    cropped = img[256:512, 0:256]
    cv2.imwrite("../corpped_PNG/{}.png".format(i+21000), cropped)
    cropped = img[256:512, 256:512]
    cv2.imwrite("../corpped_PNG/{}.png".format(i+22000), cropped)
    cropped = img[256:512, 512:768]
    cv2.imwrite("../corpped_PNG/{}.png".format(i + 23000), cropped)
    cropped = img[256:512, 768:1024]
    cv2.imwrite("../corpped_PNG/{}.png".format(i + 24000), cropped)
    cropped = img[256:512, 1024:1280]# 是否有这一行要看照片尺寸是否够大
    cv2.imwrite("../corpped_PNG/{}.png".format(i + 25000), cropped)
    # Row3
    cropped = img[512:768, 0:256]
    cv2.imwrite("../corpped_PNG/{}.png".format(i+31000), cropped)
    cropped = img[512:768, 256:512]
    cv2.imwrite("../corpped_PNG/{}.png".format(i+32000), cropped)
    cropped = img[512:768, 512:768]
    cv2.imwrite("../corpped_PNG/{}.png".format(i + 33000), cropped)
    cropped = img[512:768, 768:1024]
    cv2.imwrite("../corpped_PNG/{}.png".format(i + 34000), cropped)
    cropped = img[512:768, 1024:1280]# 是否有这一行要看照片尺寸是否够大
    cv2.imwrite("../corpped_PNG/{}.png".format(i + 35000), cropped)
    # Row4  是否有这一行要看照片尺寸是否够大
    cropped = img[768:1024, 0:256]
    cv2.imwrite("../corpped_PNG/{}.png".format(i+41000), cropped)
    cropped = img[768:1024, 256:512]
    cv2.imwrite("../corpped_PNG/{}.png".format(i+42000), cropped)
    cropped = img[768:1024, 512:768]
    cv2.imwrite("../corpped_PNG/{}.png".format(i + 43000), cropped)
    cropped = img[768:1024, 768:1024]
    cv2.imwrite("../corpped_PNG/{}.png".format(i + 44000), cropped)
    cropped = img[768:1024, 1024:1280]
    cv2.imwrite("../corpped_PNG/{}.png".format(i + 45000), cropped)
for img in glob.glob("Your Folder address/*.png"):  # 对需要裁剪的图片的文件夹循环读取
    crop(img, "Your Folder address/")
    i = i + 1
