import os, cv2, nibabel as nib, numpy as np

def CropAndResize(image):
    # Find the non-zero regions
    rows = np.any(image, axis=1)
    cols = np.any(image, axis=0)

    # Find the bounding box of the non-zero regions
    rows_indices = np.where(rows)[0]
    cols_indices = np.where(cols)[0]
    if len(rows_indices) != 0 or len(cols_indices) != 0:
        top_row = np.min(rows_indices)
        bottom_row = np.max(rows_indices)
        left_col = np.min(cols_indices)
        right_col = np.max(cols_indices)

        width = right_col - left_col
        height = bottom_row - top_row

        if width > height:
            top_row = top_row - (width - height) // 2
            bottom_row = bottom_row + (width - height) // 2
        else:
            left_col = left_col - (height - width) // 2
            right_col = right_col + (height - width) // 2

        # Crop the image
        cropped_image = image[top_row:bottom_row + 1, left_col:right_col + 1]
    else:
        cropped_image = image
    # Resize the image
    dim = [64,64]
    resized_image = cv2.resize(cropped_image, dim)
    return resized_image

def CropAndResizeWithRef(ref, image):
    # Find the non-zero regions
    rows = np.any(ref, axis=1)
    cols = np.any(ref, axis=0)

    # Find the bounding box of the non-zero regions
    rows_indices = np.where(rows)[0]
    cols_indices = np.where(cols)[0]
    if len(rows_indices) != 0 or len(cols_indices) != 0:
        top_row = np.min(rows_indices)
        bottom_row = np.max(rows_indices)
        left_col = np.min(cols_indices)
        right_col = np.max(cols_indices)

        width = right_col - left_col
        height = bottom_row - top_row

        if width > height:
            top_row = top_row - (width - height) // 2
            bottom_row = bottom_row + (width - height) // 2
        else:
            left_col = left_col - (height - width) // 2
            right_col = right_col + (height - width) // 2

        # Crop the image
        resized_ref = ref[top_row:bottom_row + 1, left_col:right_col + 1]
        cropped_image = image[top_row:bottom_row + 1, left_col:right_col + 1]
    else:
        resized_ref = ref
        cropped_image = image
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
        # plt.imsave(os.path.join(master_path, 'BraTS2021_Training_Data_2D', folder, 'flair', folder + '_flair_' + str(i+1) + '.png'), img_final)
        np.save(os.path.join(master_path, 'BraTS2021_Training_Data_2D', folder, 'flair', folder + '_flair_' + str(i+1)), img_final)
        np.save(os.path.join(master_path, 'BraTS2021_Training_Data_2D', folder, 'seg', folder + '_seg_' + str(i+1)), lab_final)
        # np.save(os.path.join(master_path, 'BraTS2021_Training_Data_2D', folder, 'seg_bin', folder + '_seg_bin_' + str(i+1) + '.png'), (lab_final > 0).astype(np.uint8))
    for img_type in ['t1', 't1ce', 't2']:
        img_path = os.path.join(master_path, org_folder, folder, folder + '_' + img_type + '.nii.gz')
        img = nib.load(img_path).get_fdata()
        for i in range(img.shape[-1]):
            img_slice = img[:,:,i]
            img_final = Standardise(CropAndResize(img_slice))
            np.save(os.path.join(master_path, 'BraTS2021_Training_Data_2D', folder, img_type, folder + '_' + img_type + '_' + str(i+1)), img_final)
            # plt.imsave(os.path.join(master_path, 'BraTS2021_Training_Data_2D', folder, img_type, folder + '_' + img_type + '_' + str(i+1) + '.png'), img_final)

def convert_Unet_train(folder, master_path = './BraTS'):
    save_folder = 'train'
    org_folder = 'BraTS2021_Training_Data_2D'
    for idx in range(155):
        dim = [64,64]
        lab_path = os.path.join(master_path, org_folder, folder, 'seg', folder + '_seg_' + str(idx+1) + '.npy')
        label = np.load(lab_path)
        ## Find boundary box for the segmentation
        # Find the non-zero regions
        rows = np.any(label, axis=1)
        cols = np.any(label, axis=0)
        # Find the bounding box of the non-zero regions
        rows_indices = np.where(rows)[0]
        cols_indices = np.where(cols)[0]
        if len(rows_indices) != 0 or len(cols_indices) != 0:
            top_row = np.min(rows_indices)
            bottom_row = np.max(rows_indices)
            left_col = np.min(cols_indices)
            right_col = np.max(cols_indices)

            width = right_col - left_col
            height = bottom_row - top_row

            if width > height:
                top_row = top_row - (width - height) // 2
                bottom_row = bottom_row + (width - height) // 2
                if top_row < 0:
                    bottom_row = bottom_row - top_row
                    top_row = 0
                if bottom_row > 63:
                    top_row = top_row - (bottom_row - 63)
                    bottom_row = 63
            else:
                left_col = left_col - (height - width) // 2
                right_col = right_col + (height - width) // 2
                if left_col < 0:
                    right_col = right_col - left_col
                    left_col = 0
                if right_col > 63:
                    left_col = left_col - (right_col - 63)
                    right_col = 63
            
            # Crop the label
            label = label[top_row:bottom_row + 1, left_col:right_col + 1]
            # Resize the label
            resized_label = cv2.resize(label, dim, interpolation = cv2.INTER_NEAREST)

            # Save the resized label
            np.save(os.path.join(master_path, 'BraTS2021_Training_Data_2D_Unet', save_folder, 'seg', folder + '_seg_' + str(idx+1)), resized_label)
            np.save(os.path.join(master_path, 'BraTS2021_Training_Data_2D_Unet', save_folder, 'cropped_area', folder + '_area_' + str(idx+1)), np.array([top_row, bottom_row, left_col, right_col]))
            # Resize each scan type...
            for img_type in ['flair', 't1', 't1ce', 't2']:
                img_path = os.path.join(master_path, org_folder, folder, img_type, folder + '_' + img_type + '_' + str(idx+1) + '.npy')
                img = np.load(img_path)
                img = img[top_row:bottom_row + 1, left_col:right_col + 1]
                # Resize the image
                resized_image = cv2.resize(img, dim)
                np.save(os.path.join(master_path, 'BraTS2021_Training_Data_2D_Unet', save_folder, img_type, folder + '_' + img_type + '_' + str(idx+1)), resized_image)

def convert_Unet_valid(folder, master_path = './BraTS'):
    save_folder = 'valid'
    org_folder = 'BraTS2021_Training_Data_2D'
    for idx in range(155):
        dim = [64,64]
        lab_path = os.path.join(master_path, org_folder, folder, 'seg', folder + '_seg_' + str(idx+1) + '.npy')
        label = np.load(lab_path)
        ## Find boundary box for the segmentation
        # Find the non-zero regions
        rows = np.any(label, axis=1)
        cols = np.any(label, axis=0)
        # Find the bounding box of the non-zero regions
        rows_indices = np.where(rows)[0]
        cols_indices = np.where(cols)[0]
        if len(rows_indices) != 0 or len(cols_indices) != 0:
            top_row = np.min(rows_indices)
            bottom_row = np.max(rows_indices)
            left_col = np.min(cols_indices)
            right_col = np.max(cols_indices)

            width = right_col - left_col
            height = bottom_row - top_row

            if width > height:
                top_row = top_row - (width - height) // 2
                bottom_row = bottom_row + (width - height) // 2
                if top_row < 0:
                    bottom_row = bottom_row - top_row
                    top_row = 0
                if bottom_row > 63:
                    top_row = top_row - (bottom_row - 63)
                    bottom_row = 63
            else:
                left_col = left_col - (height - width) // 2
                right_col = right_col + (height - width) // 2
                if left_col < 0:
                    right_col = right_col - left_col
                    left_col = 0
                if right_col > 63:
                    left_col = left_col - (right_col - 63)
                    right_col = 63
            
            # Crop the label
            label = label[top_row:bottom_row + 1, left_col:right_col + 1]
            # Resize the label
            resized_label = cv2.resize(label, dim, interpolation = cv2.INTER_NEAREST)

            # Save the resized label
            np.save(os.path.join(master_path, 'BraTS2021_Training_Data_2D_Unet', save_folder, 'seg', folder + '_seg_' + str(idx+1)), resized_label)
            np.save(os.path.join(master_path, 'BraTS2021_Training_Data_2D_Unet', save_folder, 'cropped_area', folder + '_area_' + str(idx+1)), np.array([top_row, bottom_row, left_col, right_col]))
            # Resize each scan type...
            for img_type in ['flair', 't1', 't1ce', 't2']:
                img_path = os.path.join(master_path, org_folder, folder, img_type, folder + '_' + img_type + '_' + str(idx+1) + '.npy')
                img = np.load(img_path)
                img = img[top_row:bottom_row + 1, left_col:right_col + 1]
                # Resize the image
                resized_image = cv2.resize(img, dim)
                np.save(os.path.join(master_path, 'BraTS2021_Training_Data_2D_Unet', save_folder, img_type, folder + '_' + img_type + '_' + str(idx+1)), resized_image)

def convert_Unet_test(folder, master_path = './BraTS'):
    save_folder = 'test'
    org_folder = 'BraTS2021_Training_Data_2D'
    for idx in range(155):
        dim = [64,64]
        lab_path = os.path.join(master_path, org_folder, folder, 'seg', folder + '_seg_' + str(idx+1) + '.npy')
        label = np.load(lab_path)
        ## Find boundary box for the segmentation
        # Find the non-zero regions
        rows = np.any(label, axis=1)
        cols = np.any(label, axis=0)
        # Find the bounding box of the non-zero regions
        rows_indices = np.where(rows)[0]
        cols_indices = np.where(cols)[0]
        if len(rows_indices) != 0 or len(cols_indices) != 0:
            top_row = np.min(rows_indices)
            bottom_row = np.max(rows_indices)
            left_col = np.min(cols_indices)
            right_col = np.max(cols_indices)

            width = right_col - left_col
            height = bottom_row - top_row

            if width > height:
                top_row = top_row - (width - height) // 2
                bottom_row = bottom_row + (width - height) // 2
                if top_row < 0:
                    bottom_row = bottom_row - top_row
                    top_row = 0
                if bottom_row > 63:
                    top_row = top_row - (bottom_row - 63)
                    bottom_row = 63
            else:
                left_col = left_col - (height - width) // 2
                right_col = right_col + (height - width) // 2
                if left_col < 0:
                    right_col = right_col - left_col
                    left_col = 0
                if right_col > 63:
                    left_col = left_col - (right_col - 63)
                    right_col = 63
            
            # Crop the label
            label = label[top_row:bottom_row + 1, left_col:right_col + 1]
            # Resize the label
            resized_label = cv2.resize(label, dim, interpolation = cv2.INTER_NEAREST)

            # Save the resized label
            np.save(os.path.join(master_path, 'BraTS2021_Training_Data_2D_Unet', save_folder, 'seg', folder + '_seg_' + str(idx+1)), resized_label)
            np.save(os.path.join(master_path, 'BraTS2021_Training_Data_2D_Unet', save_folder, 'cropped_area', folder + '_area_' + str(idx+1)), np.array([top_row, bottom_row, left_col, right_col]))
            # Resize each scan type...
            for img_type in ['flair', 't1', 't1ce', 't2']:
                img_path = os.path.join(master_path, org_folder, folder, img_type, folder + '_' + img_type + '_' + str(idx+1) + '.npy')
                img = np.load(img_path)
                img = img[top_row:bottom_row + 1, left_col:right_col + 1]
                # Resize the image
                resized_image = cv2.resize(img, dim)
                np.save(os.path.join(master_path, 'BraTS2021_Training_Data_2D_Unet', save_folder, img_type, folder + '_' + img_type + '_' + str(idx+1)), resized_image)