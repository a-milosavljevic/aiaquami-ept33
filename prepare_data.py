"""
prepare_data.py script is used to process original images and create training, validation, and test subsets.
The output depends from image_size and stretch parameters, so the script should be executed initially and
after changing those two parameters.
"""
from settings import *
import cv2 as cv


def square_resize_image(img_input):
    img_out = img_input
    if not stretch:
        w, h = img_out.shape[1], img_out.shape[0]
        top, bottom, left, right = 0, 0, 0, 0
        if w >= h:
            top = (w - h) // 2
            bottom = (w - h) - top
        else:
            left = (h - w) // 2
            right = (h - w) - left
        #mean_color = tuple(np.average(img_input, axis=(0, 1)))
        img_out = cv.copyMakeBorder(img_out, top, bottom, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0)) #mean_color
    img_out = cv.resize(img_out, (image_size, image_size), interpolation=cv.INTER_AREA)
    return img_out


classes = [fname for fname in os.listdir(original_data_folder) if len(fname) == 3]
print(len(classes), classes)

for cls in classes:
    cls_folder = os.path.join(original_data_folder, cls)
    files = [fname for fname in os.listdir(cls_folder)]
    train_images = []
    val_images = []
    test_images = []
    curr_subset_distribution = subset_distribution[-1]['distribution']
    for obj in subset_distribution:
        if cls in obj['filter']:
            curr_subset_distribution = obj['distribution']
    for fname in files:
        if len(fname) < 14 or len(fname) > 15:
            print('BAD FILENAME:', fname)
        if fname[-4:].lower() == '.jpg' or fname[-4:].lower() == '.tif':
            #print(fname, fname[6:9])
            unit = int(fname[6:9])
            fold = unit % len(curr_subset_distribution)
            if curr_subset_distribution[fold].upper() == 'T':
                train_images.append(fname)
            elif curr_subset_distribution[fold].upper() == 'V':
                val_images.append(fname)
            elif curr_subset_distribution[fold].upper() == 'S':
                test_images.append(fname)

    print(len(train_images), len(val_images), len(test_images), cls)

    # NORMAL PREPROCESSING FOR TRAINING PURPOSES
    train_cls_folder = os.path.join(train_folder, cls)
    if not os.path.exists(train_cls_folder):
        os.makedirs(train_cls_folder)
    val_cls_folder = os.path.join(val_folder, cls)
    if not os.path.exists(val_cls_folder):
        os.makedirs(val_cls_folder)
    test_cls_folder = os.path.join(test_folder, cls)
    if not os.path.exists(test_cls_folder):
        os.makedirs(test_cls_folder)

    for fname in train_images:
        fpath = os.path.join(cls_folder, fname)
        img = cv.imread(fpath)
        img = square_resize_image(img)
        cv.imwrite(os.path.join(train_cls_folder, fname[:-4] + '.jpg'), img)

    for fname in val_images:
        fpath = os.path.join(cls_folder, fname)
        img = cv.imread(fpath)
        img = square_resize_image(img)
        cv.imwrite(os.path.join(val_cls_folder, fname[:-4] + '.jpg'), img)

    for fname in test_images:
        fpath = os.path.join(cls_folder, fname)
        img = cv.imread(fpath)
        img = square_resize_image(img)
        cv.imwrite(os.path.join(test_cls_folder, fname[:-4] + '.jpg'), img)
