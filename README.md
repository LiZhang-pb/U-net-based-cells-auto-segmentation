<br />
<br />
#IN UPDATING
<br />
<br />
<br />

# U-net model for _Ginkgo biloba_ epidermal cells auto-segmentation
Python implementation of U-net.	<br />
Original code:	<br />
This segmentation code was used here for Epidermal Cells auto-segmentation from Cuticles micrographs of **_Ginkgo biloba_** and **fossil ginkgoaleans**, **followed** the open-source code of U-Net and FCN (**Long et al. 2015； Ronneberger et al. 2015**), and also works on Github [Community Codes](https://paperswithcode.com/paper/u-net-convolutional-networks-for-biomedical) and Chinese Software Developer Network [CSDN Projects](https://blog.csdn.net/ly_980311/article/details/105095888). <br />
***Main References:*** <br />
[Long, J., et al. (2015)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf). Fully convolutional networks for semantic segmentation. Proceedings of the IEEE conference on computer vision and pattern recognition.	<br />
[Ronneberger, O., et al. (2015).](https://paperswithcode.com/paper/u-net-convolutional-networks-for-biomedical) U-net: Convolutional networks for biomedical image segmentation. International Conference on Medical image computing and computer-assisted intervention, Springer.	<br />

![Image text](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/blob/main/read_me_Pics/Fig.2_166.png)

Figure. The U-net Architecture	<br />

Requirements
--
Please see the file named [_Requirements_](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/blob/main/Requirements.txt) in the repository. It shows the running environments and hardware needed.

Folders
--
Several folders are listed for particular aims:	<br />
[_Folder code_](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/tree/main/code) for placing all the codes used in the whole process of segmentation, and also for the training model named [_unet.hdf5_](https://drive.google.com/file/d/1bCE4AYBkh6kYh1HrBPlmyZzUm8ZGGb1Y/view?usp=sharing) (on Google Drive), which would be generated by the code after training. <br />
[_Folder data_](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/tree/main/data) for storing original images of cuticles and their labels for further deformation.	<br />
[_Folder deform_](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/tree/main/deform) for storing images and label after elastic-deformation, which comprise the training set.	<br />
[_Folder npydata_](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/tree/main/npydata) for storing npy files generated during the model-training.	<br />
[_Folder test_](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/tree/main/test) for placing images waited for segmentation, names of images should be started as 0.	<br />
[_Folder results_](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/tree/main/results) for storing predictions results.	<br />

Process
--
### Before starting this project
To guarantee all running environments are set in your computer and your hardware meets the [_Requirements_](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/blob/main/Requirements.txt).<br />
Download our model file named [_unet.hdf5_](https://drive.google.com/file/d/1bCE4AYBkh6kYh1HrBPlmyZzUm8ZGGb1Y/view?usp=sharing) (on Google Drive) in Supplementary files of our study, or generate your own model file by your own training set. The model file should be placed in the [_Folder code_](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/tree/main/code).
### Image pre-processing
The images should be cropped into 256x256 pi, transformed into 8-bit grayscale images, and renamed starting from 0.<br />
One epidermal micrograph of 1388x1040 pi, for example, a cuticle graph of _Ginkgo biloba_ in 100 magnification under an optical microscope, is supposed to be cropped into 20 pieces(4 rows x 5 columns) of 256x256 pi images. The code of [***256_spliting_by_opencv.py***](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/blob/main/code/256_spliting_by_opencv.py) in [_Folder code_](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/tree/main/code) were used in our case, and you can use the code of [***rename.py***](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/blob/main/code/rename.py) to rename the image.
Note: The images for segmentation should be placed in the [_Folder test_](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/tree/main/test).
### Run the code
in this order: <br />
[***256_spliting_by_opencv.py***](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/blob/main/code/256_spliting_by_opencv.py) to crop the images;<br />
[***rename.py***](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/blob/main/code/rename.py) to name the images, starting from 0, for model importing and training; <br />
[***elastic_deform.py***](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/blob/main/code/elastic_deform.py) to expand the training set (only in the first round); <br />
[***data.py***](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/blob/main/code/data.py) to generate data file (.npy format) that the model can read; <br />
[***unet.py***](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/blob/main/code/unet.py) to train the model based on the training set and make the prediction of the cell boundaries based on the trained model; <br />
[***see.py***](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/blob/main/code/see.py) to read the prediction data files and generate png files accordingly (in the [_Folder code_](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/tree/main/code));<br />
[***see.m***](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/blob/main/results/see2.m) to transform generated png files into the right format (in the [_Folder results_](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/tree/main/results));<br />
You would see the results of the prediction map shown in the [_Folder results_](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/tree/main/results), if the code ran smoothly;<br />
[***assessment.py.***](https://github.com/LiZhang-pb/U-net-based-cells-auto-segmentation/blob/main/code/assessment.py) to generate the performance indicator of your model based on prediction results.<br />
