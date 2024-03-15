# Brain Tumor Segmentation (BraTS) Challenge with Computer Vision

This aim of this project is to train a Computer Vision model (CNN, RNN etc.) on MRI scans to identify brain tumor segmentation. The data is provided by the RSNA-ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge 2021 and similar datasets can be downloaded from one of the below:

- [www.kaggle.com/datasets/dschettler8845/brats-2021-task1](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)
- [www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/)
- [www.kaggle.com/datasets/shakilrana/brats-2023-adult-glioma/](https://www.kaggle.com/datasets/shakilrana/brats-2023-adult-glioma/)
- [www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## Imaging Data Description

All BraTS mpMRI scans are available as NIfTI files (.nii.gz) and describe a) native (T1) and b) post-contrast T1-weighted (T1Gd), c) T2-weighted (T2), and d) T2 Fluid Attenuated Inversion Recovery (T2-FLAIR) volumes, and were acquired with different clinical protocols and various scanners from multiple data contributing institutions.

All the imaging datasets have been annotated manually, by one to four raters, following the same annotation protocol, and their annotations were approved by experienced neuro-radiologists. Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edematous/invaded tissue (ED — label 2), and the necrotic tumor core (NCR — label 1), as described both in the BraTS 2012-2013 TMI paper and in the latest BraTS summarizing paper. The ground truth data were created after their pre-processing, i.e., co-registered to the same anatomical template, interpolated to the same resolution (1 mm3) and skull-stripped.