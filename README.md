<br />
<br />
#IN UPDATING
<br />
<br />
<br />
<br />

### U-net model for cells auto-segmentation
Python implementation of U-net.	<br />
Original code:	<br />
This segmentation code was used here for Epidermal Cells auto-segmentation from Cuticles micrographs of **_Ginkgo biloba_** and **fossil ginkgoaleans**, **followed** the open-source code of U-Net and FCN (**Long et al. 2015； Ronneberger et al. 2015**), and a lot of other works on Github [Community Codes](https://paperswithcode.com/paper/u-net-convolutional-networks-for-biomedical).	<br />
Long, J., et al. (2015). Fully convolutional networks for semantic segmentation. Proceedings of the IEEE conference on computer vision and pattern recognition.	<br />
Ronneberger, O., et al. (2015). U-net: Convolutional networks for biomedical image segmentation. International Conference on Medical image computing and computer-assisted intervention, Springer.	<br />

Requirements
--
Please see the file named _Requirements_ in the repository. It shows the running environments and hardware needed.

Folders
--
Several folders are listed for particular aims:	<br />
_Folder code_ for placing all the codes used in the whole process of segmentation, and also for the training model named _unet.hdf5_, which would be generated by the code after training. <br />
_Folder data_ for storing original images of cuticles and their labels for further deformation.	<br />
_Folder deform_ for storing images and label after elastic-deformation, which comprise the training set.	<br />
_Folder npydata_ for storing npy files generated during the model-training.	<br />
_Folder test_ for placing images waited for segmentation, names of images should be started as 0.	<br />
_Folder results_ for storing predictions results.	<br />

Process
--
### Before starting this project
To guarantee all running environments are set in your computer and your hardware meets the requirement.<br />
Download our model file named _unet.hdf5_ in Supplementary files of our study, or generate your own model file by your own training set. The model file should be placed in the _Folder code_.
### Image pre-processing
The images should be cropped into 256x256 pi, transformed into 8-bit grayscale images, and renamed starting with 0.<br />
One epidermal micrograph of 1388x1040 pi, for example, a cuticle graph of _Ginkgo biloba_ in 100 magnification under an optical microscope, is supposed to be cropped into 20 pieces(4 rows x 5 columns) of 256x256 pi images. The code of 256_spliting_by_opencv.py in _Folder code_ were used in our case, and you can use the code of rename.py to rename the image.
Note: The images for segmentation should be placed in the _Folder test_.
### Run the code
in this order: <br />
***256_spliting_by_opencv.py*** to crop the images.<br />
***rename.py*** <br />
***elastic_deform.py*** to expand the training set (only in the first round) <br />
***data.py*** <br />
***unet.py***<br />
***see.py*** (in the _Folder code_)<br />
***see.m*** (in the _Folder results_).<br />
You would see the results of the prediction map shown in the _Folder results_.
assessment.py. to generate the performance indicator of your model based on prediction results.<br />
