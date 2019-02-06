import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
#import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

tf.reset_default_graph()

img_placeholder = tf.placeholder(tf.string)

filename = 'faces.jpg'
#filename = '/home/Downloads/lord_of_the_ring.jpg'
#filename = '/home/Downloads/gollum.jpg'
#filename = '/home/Downloads/Explorer_HD2K_SN19671_06-32-16.png'

with tf.gfile.GFile(filename, 'rb') as fid:
    image_data_binary = fid.read()

#rgb_image = tf.image.decode_jpeg(img_placeholder, channels=3)
rgb_image = tf.image.decode_png(img_placeholder, channels=3)
rgb_image = tf.image.convert_image_dtype(rgb_image, tf.float32)
resized_rgb_image = tf.image.resize_images(rgb_image, [128, 128])
resized_rgb_image = tf.expand_dims(resized_rgb_image, 0)
##########################################

# Apply image detector on a single image.
module = hub.Module("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1")
detector_output = module(resized_rgb_image, as_dict=True)
class_names = detector_output["detection_class_entities"]
boxes = detector_output['detection_boxes']
scores = detector_output["detection_scores"]

##########################################
with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    #image = sess.run(rgb_image, feed_dict={img_placeholder: image_data_binary})

    #print(np.max(image))
    origin_img, detected_class_names, detected_image_boxes, detected_scores = sess.run([rgb_image, class_names, boxes, scores],
        feed_dict={img_placeholder: image_data_binary})
    
    shape = origin_img.shape
    for ce, b, s in zip (detected_class_names, detected_image_boxes, detected_scores):
        ce = ce.decode('ascii')
        #if (ce == "Human face" or ce == "Person") and s>0.5:
        if ce == "Human face":
            s1, s2, s3, s4 = int(b[1] * shape[1]), int(b[0] * shape[0]), int(b[3] * shape[1]), int(b[2] * shape[0])
            cv2.rectangle(origin_img, (s1, s2), (s3, s4), (0, 255, 0), 5)
            cv2.putText(origin_img, ce, (s1, s2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #print(s)
            cv2.putText(origin_img, str(s), (s1, s2+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #print("< {0}: {1} >".format(ce, b))
    
    mpimg.imsave("detected_2.png", origin_img)