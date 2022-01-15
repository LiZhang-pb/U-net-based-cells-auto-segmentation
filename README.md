# U-net-based-cells-auto-segmentation
This segmentation code was used here for Epidermal Cells auto-segmentation from Cuticles micrographs of _Ginkgo biloba_, followed the open-source code of U-Net and FCN (Long et al. 2015, Ronneberger et al. 2015). 
There are several folders for particular aims:
_code_ for all the codes used in the whole process of segmentation, and also for the training model named _unet.hdf5_.
_data_ for storing original images of cuticles and their labels for further deformation.
_deform_ for storing images and label after elastic-deformation, which comprise the training set.
_npydata_ for storing npy files generated during the model-training.
_test_ for images waited for segmentation, files' name should be started as 0.
_results_ for storing predictions results.
