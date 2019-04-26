import numpy
import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
ResultFolder = "1"
OriginImage = '../demo_data/demo_im.jpg'
IUVImage = '/demo_im_IUV.png'
INDSImage = '/demo_im_INDS.png'
'''
'''
ResultFolder = "2"
OriginImage = ResultFolder+'/000000000057.jpg'
IUVImage = '/000000000057_IUV.png'
INDSImage = '/000000000057_INDS.png'
'''
'''
ResultFolder = "3"
OriginImage = ResultFolder+'/000000000063.jpg'
IUVImage = '/000000000063_IUV.png'
INDSImage = '/000000000063_INDS.png'
'''
'''
ResultFolder = "4"
OriginImage = ResultFolder+'/000000000069.jpg'
IUVImage = '/000000000069_IUV.png'
INDSImage = '/000000000069_INDS.png'
'''
ResultFolder = "5"
OriginImage = ResultFolder+'/000000000171.jpg'
IUVImage = '/000000000171_IUV.png'
INDSImage = '/000000000171_INDS.png'

im  = cv2.imread(OriginImage)
IUV = cv2.imread(ResultFolder+IUVImage)
INDS = cv2.imread(ResultFolder+INDSImage,  0)

fig, ax = plt.subplots(3, 1, figsize=(10, 15))

#fig = plt.figure(figsize=[15,15])
ax[0].imshow(   np.hstack((IUV[:,:,0]/24. ,IUV[:,:,1]/256. ,IUV[:,:,2]/256.))  )
ax[0].set_title('I, U and V images.')
ax[0].axis('off')
#plt.show()

#fig = plt.figure(figsize=[12,12])
ax[1].imshow( im[:,:,::-1] )
ax[1].contour( IUV[:,:,1]/256.,10, linewidths = 1 )
ax[1].contour( IUV[:,:,2]/256.,10, linewidths = 1 )
ax[1].axis('off')
#plt.show()

#fig = plt.figure(figsize=[12,12])
ax[2].imshow( im[:,:,::-1] )
ax[2].contour( INDS, linewidths = 4 )
ax[2].axis('off')

#plt.show()
plt.savefig(ResultFolder+'/merged_result.pdf', bbox_inches='tight')