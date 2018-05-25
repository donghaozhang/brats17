import nibabel
import numpy as np
import matplotlib.pyplot as plt
import random
import colorsys
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

def load_nifty_volume_as_array(filename, with_header = False):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
        with_header: return affine and hearder infomation
    outputs:
        data: a numpy data array
    """
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2,1,0])
    if(with_header):
        return data, img.affine, img.header
    else:
        return data

def convert_label(in_volume, label_convert_source, label_convert_target):
    """
    convert the label value in a volume
    inputs:
        in_volume: input nd volume with label set label_convert_source
        label_convert_source: a list of integers denoting input labels, e.g., [0, 1, 2, 4]
        label_convert_target: a list of integers denoting output labels, e.g.,[0, 1, 2, 3]
    outputs:
        out_volume: the output nd volume with label set label_convert_target
    """
    mask_volume = np.zeros_like(in_volume)
    convert_volume = np.zeros_like(in_volume)
    for i in range(len(label_convert_source)):
        source_lab = label_convert_source[i]
        target_lab = label_convert_target[i]
        if(source_lab != target_lab):
            temp_source = np.asarray(in_volume == source_lab)
            temp_target = target_lab * temp_source
            mask_volume = mask_volume + temp_source
            convert_volume = convert_volume + temp_target
    out_volume = in_volume * 1
    out_volume[mask_volume>0] = convert_volume[mask_volume>0]
    return out_volume

def apply_mask(image, mask, alpha=0.5):
    """Apply the given mask to the image.
    """
    newimage = np.zeros([image.shape[0], image.shape[1], 3])
    color_rand = random.random()
    for c in range(3):
        newimage[:, :, c] = np.where(mask == 1, alpha * color_rand * 255, image)
    return newimage

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    # print('the value of colors: ', colors)
    return colors

prefix = '/home/donghao/Desktop/donghao/brain_sgementation/MICCAI_BraTS17_Data_Training/HGG/Brats17_CBICA_AAB_1/'
t1_img = load_nifty_volume_as_array(prefix + 'Brats17_CBICA_AAB_1_t1.nii.gz')
t1ce_img = load_nifty_volume_as_array(prefix + 'Brats17_CBICA_AAB_1_t1ce.nii.gz')
t2_img = load_nifty_volume_as_array(prefix + 'Brats17_CBICA_AAB_1_t2.nii.gz')
flair_img = load_nifty_volume_as_array(prefix + 'Brats17_CBICA_AAB_1_flair.nii.gz')
label_img = load_nifty_volume_as_array(prefix + 'Brats17_CBICA_AAB_1_seg.nii.gz')

# 1. non-enhancing tumor core (NETC) 2. edema 4. enhancing tumor core (ETC)
# enhancing tumor core -> t1ce
# edema -> t2
#
NETC_vol = convert_label(label_img, [0, 1, 2, 4], [0, 1, 0, 0])
edema_vol = convert_label(label_img, [0, 1, 2, 4], [0, 0, 1, 0])
ETC_vol = convert_label(label_img, [0, 1, 2, 4], [0, 0, 0, 1])

# print('the size of t1_img is ', t1_img.shape)
slice_id = 70
t1_slice = t1_img[slice_id, :, :]
t1ce_slice = t1ce_img[slice_id, :, :]
t2_slice = t2_img[slice_id, :, :]
flair_slice = flair_img[slice_id, :, :]

edema_slice = edema_vol[slice_id, :, :]
NETC_slice = NETC_vol[slice_id, :, :]
ETC_slice = ETC_vol[slice_id, :, :]

color = random_colors(1)
edema_t2_slice = apply_mask(t2_slice, edema_slice)

# fig1 = plt.figure()
# plt.imshow(t1_slice, cmap='gray')
# plt.title('t1_slice')
# fig2 = plt.figure()
# plt.imshow(t1ce_slice, cmap='gray')
# plt.title('t1ce_slice')
fig3 = plt.figure()
plt.imshow(t2_slice, cmap='gray')
plt.title('t2_slice')
# fig4 = plt.figure()
# plt.imshow(flair_slice, cmap='gray')
# plt.title('flair_slice')
# label_slice_copy = label_slice.copy()
# label_slice_copy[label_slice>1] = 0
# fig5 = plt.figure()
# plt.imshow(label_slice, cmap='gray')
# plt.title('label_slice')

# fig1 = plt.figure()
# plt.imshow(NETC_slice)
# plt.title('NETC_slice')
fig2 = plt.figure()
plt.imshow(edema_slice)
plt.title('edma_slice')
# fig4 = plt.figure()
# plt.imshow(ETC_slice)
# plt.title('ETC_slice')
fig7 = plt.figure()
plt.imshow(edema_t2_slice)
plt.title('edema_t2_slice')
plt.show()
print('random number is ', random.random())
