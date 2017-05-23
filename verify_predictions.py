import os 
import numpy as np 
from skimage import io
from skimage import color
from skimage.transform import resize

def show(img):
	io.imshow(img)
	io.show()
    
def overlay_on_gray(oimg,overlayGREEN=None,overlayRED=None,overlayBLUE=None,alpha = 0.6):
    rows, cols = oimg.shape
    color_mask = np.zeros((rows, cols, 3))
    if overlayGREEN is not None:
        color_mask[:,:,1] = overlayGREEN
    if overlayRED  is not None:
        color_mask[:,:,2] = overlayRED
    if overlayBLUE is not None:
        color_mask[:,:,0] = overlayBLUE
    
    img_color = np.dstack((oimg, oimg, oimg))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)
    
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
    img_masked = color.hsv2rgb(img_hsv)

    return img_masked,color_mask

   
testimgs = []
for f2 in os.listdir('../splitdata/testX/a')[:]:
	print f2 
	test_img = io.imread('../splitdata/testX/a/'+ f2 ,  True)
	test_img = resize( test_img , (160,160), preserve_range=True)
	# show(test_img)
	testimgs.append(test_img)


i =0 
for f in os.listdir('./preds')[:]:
	print f
	pred_img = io.imread('./preds/'+ f)
	
	print pred_img.shape 
	print pred_img.max()

	# show(pred_img)

	# Half in Range
	final_mask = pred_img >= ((pred_img.max() - pred_img.min()) / 2 ) + pred_img.min()
	a , b = overlay_on_gray(testimgs[i], overlayGREEN = final_mask)
	show(a)
	i+=1