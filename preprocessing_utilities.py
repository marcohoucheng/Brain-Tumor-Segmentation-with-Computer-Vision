import os, cv2, nibabel as nib, numpy as np

from utilities import boundary

def CropAndResize(image):

    top_row, bottom_row, left_col, right_col = boundary(image)
    top_row, bottom_row, left_col, right_col = int(top_row), int(bottom_row), int(left_col), int(right_col)

    # Crop the image
    cropped_image = image[top_row:bottom_row + 1, left_col:right_col + 1]
    
    # Resize the image
    dim = [64,64]
    resized_image = cv2.resize(cropped_image, dim)
    return resized_image

def CropAndResizeWithRef(ref, image):

    top_row, bottom_row, left_col, right_col = boundary(ref)

    # Crop the image
    resized_ref = ref[top_row:bottom_row + 1, left_col:right_col + 1]
    cropped_image = image[top_row:bottom_row + 1, left_col:right_col + 1]

    # Resize the image
    dim = [64,64]
    resized_image = cv2.resize(cropped_image, dim, interpolation = cv2.INTER_NEAREST)
    resized_ref = cv2.resize(resized_ref, dim)
    return resized_ref, resized_image.astype(np.uint8)

def Standardise(image) :
        if np.abs(image).sum() == 0:
            return image
        with np.errstate(divide='ignore',invalid='ignore'):
            image_nan = np.where(image == 0, np.nan, image)
            new_image = (image_nan - np.nanmean(image_nan)) / np.nanstd(image_nan)
            new_image = np.nan_to_num(new_image)
        return new_image

def convert(folder, master_path = './BraTS'):
    org_folder = 'BraTS2021_Training_Data'
    img_path = os.path.join(master_path, org_folder, folder, folder + '_flair.nii.gz')
    lab_path = os.path.join(master_path, org_folder, folder, folder + '_seg.nii.gz')
    img = nib.load(img_path).get_fdata()
    lab = nib.load(lab_path).get_fdata()
    for i in range(img.shape[-1]):
        img_slice = img[:,:,i]
        lab_slice = lab[:,:,i]
        img_final, lab_final = CropAndResizeWithRef(img_slice, lab_slice)
        img_final = Standardise(img_final)
        np.save(os.path.join(master_path, 'BraTS2021_Training_Data_Slice', folder, 'flair', folder + '_flair_' + str(i+1)), img_final)
        np.save(os.path.join(master_path, 'BraTS2021_Training_Data_Slice', folder, 'seg', folder + '_seg_' + str(i+1)), lab_final)
    for img_type in ['t1', 't1ce', 't2']:
        img_path = os.path.join(master_path, org_folder, folder, folder + '_' + img_type + '.nii.gz')
        img = nib.load(img_path).get_fdata()
        for i in range(img.shape[-1]):
            img_slice = img[:,:,i]
            img_final = Standardise(CropAndResize(img_slice))
            np.save(os.path.join(master_path, 'BraTS2021_Training_Data_Slice', folder, img_type, folder + '_' + img_type + '_' + str(i+1)), img_final)

def convert_Unet_train(folder, master_path = './BraTS'):
    save_folder = 'train'
    org_folder = 'BraTS2021_Training_Data_Slice'
    for idx in range(155):
        dim = [64,64]
        lab_path = os.path.join(master_path, org_folder, folder, 'seg', folder + '_seg_' + str(idx+1) + '.npy')
        label = np.load(lab_path)

        # Find boundary box for the segmentation
        top_row, bottom_row, left_col, right_col = boundary(label)
        
        if not(top_row == 0 and bottom_row == 63 and left_col == 0 and right_col == 63):
            # Crop the label
            label = label[top_row:bottom_row + 1, left_col:right_col + 1]
            # Resize the label
            resized_label = cv2.resize(label, dim, interpolation = cv2.INTER_NEAREST)

            # Save the resized label
            np.save(os.path.join(master_path, 'BraTS2021_Training_Data_Slice_Cropped', save_folder, 'seg', folder + '_seg_' + str(idx+1)), resized_label)
            np.save(os.path.join(master_path, 'BraTS2021_Training_Data_Slice_Cropped', save_folder, 'cropped_area', folder + '_area_' + str(idx+1)), np.array([top_row, bottom_row, left_col, right_col]))
            # Resize each scan type...
            for img_type in ['flair', 't1', 't1ce', 't2']:
                img_path = os.path.join(master_path, org_folder, folder, img_type, folder + '_' + img_type + '_' + str(idx+1) + '.npy')
                img = np.load(img_path)
                img = img[top_row:bottom_row + 1, left_col:right_col + 1]
                # Resize the image
                resized_image = cv2.resize(img, dim)
                np.save(os.path.join(master_path, 'BraTS2021_Training_Data_Slice_Cropped', save_folder, img_type, folder + '_' + img_type + '_' + str(idx+1)), resized_image)

def convert_Unet_valid(folder, master_path = './BraTS'):
    save_folder = 'valid'
    org_folder = 'BraTS2021_Training_Data_Slice'
    for idx in range(155):
        dim = [64,64]
        lab_path = os.path.join(master_path, org_folder, folder, 'seg', folder + '_seg_' + str(idx+1) + '.npy')
        label = np.load(lab_path)
        
        # Find boundary box for the segmentation
        top_row, bottom_row, left_col, right_col = boundary(label)
        
        if not(top_row == 0 and bottom_row == 63 and left_col == 0 and right_col == 63):
                        
            # Crop the label
            label = label[top_row:bottom_row + 1, left_col:right_col + 1]
            # Resize the label
            resized_label = cv2.resize(label, dim, interpolation = cv2.INTER_NEAREST)

            # Save the resized label
            np.save(os.path.join(master_path, 'BraTS2021_Training_Data_Slice_Cropped', save_folder, 'seg', folder + '_seg_' + str(idx+1)), resized_label)
            np.save(os.path.join(master_path, 'BraTS2021_Training_Data_Slice_Cropped', save_folder, 'cropped_area', folder + '_area_' + str(idx+1)), np.array([top_row, bottom_row, left_col, right_col]))
            # Resize each scan type...
            for img_type in ['flair', 't1', 't1ce', 't2']:
                img_path = os.path.join(master_path, org_folder, folder, img_type, folder + '_' + img_type + '_' + str(idx+1) + '.npy')
                img = np.load(img_path)
                img = img[top_row:bottom_row + 1, left_col:right_col + 1]
                # Resize the image
                resized_image = cv2.resize(img, dim)
                np.save(os.path.join(master_path, 'BraTS2021_Training_Data_Slice_Cropped', save_folder, img_type, folder + '_' + img_type + '_' + str(idx+1)), resized_image)

def convert_Unet_test(folder, master_path = './BraTS'):
    save_folder = 'test'
    org_folder = 'BraTS2021_Training_Data_Slice'
    for idx in range(155):
        dim = [64,64]
        lab_path = os.path.join(master_path, org_folder, folder, 'seg', folder + '_seg_' + str(idx+1) + '.npy')
        label = np.load(lab_path)
        
        # Find boundary box for the segmentation
        top_row, bottom_row, left_col, right_col = boundary(label)
        
        if not(top_row == 0 and bottom_row == 63 and left_col == 0 and right_col == 63):
            
            # Crop the label
            label = label[top_row:bottom_row + 1, left_col:right_col + 1]
            # Resize the label
            resized_label = cv2.resize(label, dim, interpolation = cv2.INTER_NEAREST)

            # Save the resized label
            np.save(os.path.join(master_path, 'BraTS2021_Training_Data_Slice_Cropped', save_folder, 'seg', folder + '_seg_' + str(idx+1)), resized_label)
            np.save(os.path.join(master_path, 'BraTS2021_Training_Data_Slice_Cropped', save_folder, 'cropped_area', folder + '_area_' + str(idx+1)), np.array([top_row, bottom_row, left_col, right_col]))
            # Resize each scan type...
            for img_type in ['flair', 't1', 't1ce', 't2']:
                img_path = os.path.join(master_path, org_folder, folder, img_type, folder + '_' + img_type + '_' + str(idx+1) + '.npy')
                img = np.load(img_path)
                img = img[top_row:bottom_row + 1, left_col:right_col + 1]
                # Resize the image
                resized_image = cv2.resize(img, dim)
                np.save(os.path.join(master_path, 'BraTS2021_Training_Data_Slice_Cropped', save_folder, img_type, folder + '_' + img_type + '_' + str(idx+1)), resized_image)

def convert_Unet_pred(file, master_path = './BraTS'):
    org_folder = 'BraTS2021_Training_Data_Slice'
    ref_folder = 'CA_Flair_Area'
    save_folder = 'UNet_Test_Input'
    org_image = file[0] + '_flair_' + file[1] + '.npy'
    ref_dim = file[0] + '_ROI_pred_' + file[1] + '.npy'

    org_image = np.load(os.path.join(master_path, org_folder, file[0], 'flair', org_image))
    ref_dim = np.load(os.path.join(master_path, ref_folder, ref_dim))

    dim = [64,64]

    top_row, bottom_row, left_col, right_col = ref_dim
    
    # Crop image
    image = org_image[top_row:bottom_row + 1, left_col:right_col + 1]

    # Resize the label
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_NEAREST)
    
    # Save the resized image
    np.save(os.path.join(master_path, save_folder, file[0] + '_ROI_cropped_' + file[1]), resized_image)