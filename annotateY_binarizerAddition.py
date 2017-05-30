import os
from skimage import io , color, measure
import numpy as np 


def binarizer_nav_multiple(navimg):
	navimg[-25:,:65] = 0
	th = 0.92
	th2 = 0.99
	navimg = (navimg >= th) & (navimg <=th2)
	
	regions = measure.label(navimg)
	nregs = np.unique(regions)

	imgs = np.zeros(navimg.shape)
	boundaries = []
	for i in range(len(nregs)):
		if i == 0:
			continue
		ri = (regions == i)
		if np.sum(ri.ravel()) > 2000:    
			imgs+=ri

	return imgs

def show(img):
	io.imshow(img)
	io.show()

def annotate_and_backup(DATAPATH , ANOTPATH):

	for filename in os.listdir(DATAPATH)[:]:
		print DATAPATH , filename

		if not os.path.exists(DATAPATH + '_multi'):
			os.mkdir(DATAPATH+'_multi')

		nav = io.imread(ANOTPATH + '/'+ filename, True)
		gt_one = io.imread(DATAPATH +'/'+filename, True)
		navmask = binarizer_nav_multiple(nav)

		gt_all = navmask.astype(bool) + gt_one.astype(bool)

		io.imsave(DATAPATH+'_multi/'+filename, gt_all.astype(int)*255)
		# show(gt_all


if __name__ == '__main__':
	
	DATAPATH1 = '../splitdata/trainY'
	DATAPATH2 = '../splitdata/testY'
	ANOTPATH =  '../massivedata/full_static/'

	annotate_and_backup(DATAPATH1 , ANOTPATH)
	annotate_and_backup(DATAPATH2 , ANOTPATH)
