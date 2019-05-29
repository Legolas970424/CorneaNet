import os
import glob
import random
import numpy as np
import skimage.io as io
import imgaug as ia
import imgaug.augmenters as iaa


# 定义一组变换方法，使用下面的0个到4个之间的方法去增强图像
ia.seed(1)
seq = iaa.Sequential([
 
    iaa.Fliplr(0.5),
    
    iaa.Affine(                          
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        rotate=(-15, 15),
        shear=(-15, 15),
        order=[0, 1],
        cval=0,
        mode="constant"
    ),
           
    iaa.Sometimes(0.5,
        iaa.CropAndPad(
            percent=0.05,
            pad_mode='constant',
            pad_cval=0,
            keep_size=True,
            sample_independently=True
        ),
    ),
                           
    iaa.Sometimes(0.4,iaa.PiecewiseAffine(scale=(0.01, 0.05))),
    
    iaa.SomeOf((0, 3),[
        iaa.Multiply((0.9, 1.1)),
        iaa.ContrastNormalization((0.9, 1.1)),
        iaa.Dropout((0, 0.03))       
    ], random_order=True) 

], random_order=True)

def aug(image,label,num_class):   
    mask = ia.SegmentationMapOnImage(label, shape = image.shape, nb_classes = num_class)
    image_aug, label_aug = seq(image = image, segmentation_maps = mask)
    label_aug = label_aug.get_arr_int().astype(np.uint8)
    
    return image_aug, label_aug


def adjustData(image, mask, flag_multi_class, num_class):
    if(flag_multi_class):
        image = image / 255        
        mask = mask[:, :, :, 0] if(len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i, i] = 1
        mask = new_mask
    elif(np.max(image) > 1):
        image = image / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        
    return (image, mask)
 

def trainGenerator(image_path = './data/train/image',
                   mask_path = './data/train/label',
                   batch_size = 4,
                   flag_multi_class = True,
                   num_class = 4,
                   image_as_gray = True,
                   mask_as_gray = True):
    image_name_list = glob.glob(image_path + '/*.jpeg')
    image_list = []
    label_list = []
    num=0
    
    while True:
        random.shuffle(image_name_list)
        for item in image_name_list:
#            print(item)
            num=num + 1
            image = io.imread(item, as_gray = image_as_gray)
            mask = io.imread((item.replace(image_path,mask_path)).replace('jpeg','png'),
                             as_gray = mask_as_gray)       
            mask[mask == 255]=1
            mask[mask == 190]=2
            mask[mask == 105]=3                     
            image_aug, label_aug = aug(image,mask,num_class)
            
            image_aug = np.reshape(image_aug,image_aug.shape + (1,))
            label_aug = np.reshape(label_aug,label_aug.shape + (1,))
            image_aug, label_aug = adjustData(image_aug,
                                              label_aug,
                                              flag_multi_class,
                                              num_class)
            image_list.append(image_aug)
            label_list.append(label_aug)
            
            if num >= batch_size:
                image_array = np.array(image_list).astype('float32')
                label_array = np.array(label_list)                
                yield (image_array, label_array)
                image_list = []
                label_list = []
                num = 0
                
                
def validGenerator(image_path = './data/valid/image',
                   mask_path = './data/valid/label',
                   batch_size = 4,
                   flag_multi_class = True,
                   num_class = 4,
                   image_as_gray = True,
                   mask_as_gray = True):
    image_name_list = glob.glob(image_path + '/*.jpeg')
    image_list = []
    label_list = []
    num=0
    
    while True:
        random.shuffle(image_name_list)
        for item in image_name_list:
#            print(item)
            num=num + 1
            image = io.imread(item, as_gray = image_as_gray)
            mask = io.imread((item.replace(image_path,mask_path)).replace('jpeg','png'),
                             as_gray = mask_as_gray)       
            mask[mask == 255]=1
            mask[mask == 190]=2
            mask[mask == 105]=3                     
            image_aug, label_aug = aug(image,mask,num_class)
            
            image_aug = np.reshape(image_aug,image_aug.shape + (1,))
            label_aug = np.reshape(label_aug,label_aug.shape + (1,))
            image_aug, label_aug = adjustData(image_aug,
                                              label_aug,
                                              flag_multi_class,
                                              num_class)
            image_list.append(image_aug)
            label_list.append(label_aug)
            
            if num >= batch_size:
                image_array = np.array(image_list).astype('float32')
                label_array = np.array(label_list)                
                yield (image_array, label_array)
                image_list = []
                label_list = []
                num = 0


def testGenerator(test_path = './data/test/aug', as_gray = True):
    test_name_list = sorted(glob.glob(test_path + '/*.png'))
    for i in test_name_list:
        image = io.imread(i, as_gray = as_gray)
        image = image / 255
        image = np.reshape(image, image.shape + (1,))
        image = np.reshape(image, (1,) + image.shape)
        yield image
        
                    
def labelVisualize_gray(image, num_class):
    image = image[:,:,0] if len(image.shape) == 3 else image
    image[image == 1] = 255
    image[image == 2] = 190
    image[image == 3] = 105
    return image


def saveResult(save_path,npyfile,flag_multi_class = True,num_class = 4):
    save_name_list = sorted(glob.glob(save_path + '/*.png'))
    
    for i,item in enumerate(npyfile):
        if flag_multi_class:
            item_mask = np.argmax(item,axis=-1)
#            print(np.max(item_mask),np.min(item_mask))
            label = labelVisualize_gray(item_mask, num_class)           
        else:
            label = item[:,:,0]
#            print(np.max(label), np.min(label))
            label[label > 0.5] = 1
            label[label <= 0.5] = 0
            
        tempfilename = os.path.basename(save_name_list[i])
        (filename,extension) = os.path.splitext(tempfilename)
        save_dir =  os.path.join(save_path, 'predict')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        io.imsave(os.path.join(save_dir,"{}_predict.png".format(filename)),label)

                
            
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    