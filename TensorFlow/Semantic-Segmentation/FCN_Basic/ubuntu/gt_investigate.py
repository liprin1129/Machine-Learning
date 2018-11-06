#import scipy.misc
import matplotlib.pyplot as plt
import sys
from PIL import Image
import numpy as np
import tensorflow as tf

#print(sys.argv[1])

gt_name = '2007_000170.png'

img = Image.open(sys.argv[1])
gt = Image.open(gt_name)
pallete = gt.getpalette()

gt = np.array(gt, dtype=np.uint8)

#print(np.max(gt))
print(set(gt.reshape(-1)))

a = np.where(gt == 255, 21, gt)

print(a)

with tf.Session() as sess:
    hot_tf = sess.run(tf.one_hot(indices=a,depth=22, on_value=1.0, off_value=0.0, dtype=tf.float32))
    print(hot_tf.shape)
    #print(hot)

hot_np = np.eye(22)[a]
print(hot_np.shape)

print(np.array_equal(hot_tf, hot_np))

'''
#pil_image = Image.fromarray(gt.astype(dtype=np.uint8))

with tf.gfile.Open('gt_no_colormap.png', mode='w') as f:
    pil_image.save(f, 'PNG')

_, ax = plt.subplots(1, 2, figsize=(5,5))

ax[0].imshow(img)
ax[1].imshow(pil_image)
plt.show()
'''