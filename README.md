# U-net model for cells auto-segmentation
Original code:
This segmentation code was used here for Epidermal Cells auto-segmentation from Cuticles micrographs of _Ginkgo biloba_ and fossil ginkgoaleans, **followed** the open-source code of U-Net and FCN (**Long et al. 2015； Ronneberger et al. 2015**). 

Requirements
--
Please see the file named _Requirements_ in the repository. It shows the running environments and hardware needed.

Folders
--
There are several folders for particular aims:	
_Folder code_ for all the codes used in the whole process of segmentation, and also for the training model named _unet.hdf5_.	

_Folder data_ for storing original images of cuticles and their labels for further deformation.	

_Folder deform_ for storing images and label after elastic-deformation, which comprise the training set.	

_Folder npydata_ for storing npy files generated during the model-training.	

_Folder test_ for images waited for segmentation, files' name should be started as 0.	

_Folder results_ for storing predictions results.	

