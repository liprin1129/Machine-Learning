from facenet import FacenetModel
import os
import tensorflow as tf
import sys
import numpy as np
from tqdm import tqdm
import tensorboard_helper as tbh
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('image_path', '/home/user170/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/', "root directory including images and tfrecords")

tf.app.flags.DEFINE_string('mode', 'test', "Select train or pred mode")


# Set tensorboard folder name
summary_name_str = 'facenet2'

current_epoch_return = 0;

def training():
    """ #####################################
    ####### Hyper parameters starts ######"""

    epoch = 60
    batch = 20
    learning_rate = 0.001

    num_outputs = len(list(
        filter(lambda y: os.path.isdir(y), list(
            #map(lambda x: os.path.join(FLAGS.image_path+"images", x), os.listdir(FLAGS.image_path+"images"))))))
            map(lambda x: os.path.join('/home/user170/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/'+"images", x), os.listdir('/home/user170/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/'+"images"))))))

    model = FacenetModel(epoch, batch, learning_rate, num_outputs) # Initialize FacenetModel class to set shapes of layers

    model.block1["unit1"] = [64, 64, 32]
    model.block1["unit2"] = [64, 64, 32]
    model.block1["unit3"] = [64, 64, 32]

    model.block2["unit1"] = [128, 128, 64]
    model.block2["unit2"] = [128, 128, 64]
    model.block2["unit3"] = [128, 128, 64]
    model.block2["unit4"] = [128, 128, 64]

    model.block3["unit1"] = [256, 256, 128]
    model.block3["unit2"] = [256, 256, 128]
    model.block3["unit3"] = [256, 256, 128]
    model.block3["unit4"] = [256, 256, 128]
    model.block3["unit5"] = [256, 256, 128]
    model.block3["unit6"] = [256, 256, 128]

    model.block4["unit1"] = [512, 512, 256]
    model.block4["unit2"] = [512, 512, 256]
    model.block4["unit3"] = [512, 512, 256]

    """ ###### Hyper parameters ends #########
    ###################################### """
    
    model('/home/user170/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/', 224, 224)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    config_proto = tf.ConfigProto()
    #config_proto.gpu_options.allow_growth = True
    config_proto.gpu_options.per_process_gpu_memory_fraction = 0.6
    with tf.Session(config=config_proto) as sess:

        #summary_writer = tf.summary.FileWriter(os.path.join('summaries','facenet'), sess.graph) # Write tensorboard
        summary_writer = tbh.summary_writer_fn(summary_name_str, sess)

        sess.run(tf.global_variables_initializer())

        training_handle = sess.run(model.train_iterator.string_handle())
        validation_handle = sess.run(model.validation_iterator.string_handle())
        for current_epoch_int in tqdm(range(epoch)):
            try:
                print("=== Training ==="); sys.stdout.flush()
                sess.run(model.train_iterator.initializer)
                total_accuracy = 0
                count = 0
                #print("Processing: ", end=""); sys.stdout.flush()
                while True:
                    count += 1
                    #print(".", end=""); sys.stdout.flush()
                    
                    extracted_data = sess.run(model.get_next_in_interators, feed_dict={model.handle_placeholder: training_handle})
                    
                    if len(extracted_data[1]) > 1:
                        #extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        loss, _ = sess.run([model.loss, model.loss_minimizer], feed_dict={model.input_dataset_placeholder:extracted_data[0], model.labels_dataset_placeholder:extracted_data[1], model.train_valid_placeholder:True})
                        pred = sess.run(model.predictions, feed_dict={model.input_dataset_placeholder:extracted_data[0], model.train_valid_placeholder:True})
                        total_accuracy += np.sum(np.equal(pred, extracted_data[1])) / batch
                    
                    current_epoch_return = current_epoch_int # Set current itoration as current epoch state
                    #print("{0} accuracy: \t{1}".format(np.sum(np.equal(pred, extracted_data[1])), np.equal(pred, extracted_data[1]))); sys.stdout.flush()
                    #print("expected:\t{0} \npredicted:\t{1}".format(extracted_data[1], pred)); sys.stdout.flush()
                    
            except tf.errors.OutOfRangeError:
                # Execute the summaries defined above
                tbh.write_loss_and_accuracy_summary_fn(summary_writer, sess, loss, total_accuracy, count, epoch)

                print("\n{0} %".format((total_accuracy/count)*100)); sys.stdout.flush()
                pass

            try:
                print("=== Validation ==="); sys.stdout.flush()
                sess.run(model.validation_iterator.initializer)
                total_accuracy = 0
                count = 0
                #print("Processing: ", end=""); sys.stdout.flush()
                while True:
                    count += 1
                    #print(".", end=""); sys.stdout.flush()

                    extracted_data = sess.run(model.get_next_in_interators, feed_dict={model.handle_placeholder: validation_handle})

                    if len(extracted_data[1]) > 1:
                        pred = sess.run(model.predictions, feed_dict={model.input_dataset_placeholder:extracted_data[0], model.train_valid_placeholder:False})
                        total_accuracy += np.sum(np.equal(pred, extracted_data[1])) / batch

                    current_epoch = count # Set current itoration as current epoch state
                    
            except tf.errors.OutOfRangeError:
                print("\n{0} %".format((total_accuracy/count)*100)); sys.stdout.flush()

                # Execute the summaries defined above
                tbh.write_accuracy(summary_writer, sess, total_accuracy, count, epoch)

                # Save the variables to disk.
                save_file_name = 'trained_tf_data/{0}_{1:05d}-of-{2:05d}'.format("facenet", current_epoch_int+1, epoch)
                save_path = saver.save(sess, os.path.join('/home/user170/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/', save_file_name))
                #save_path = saver.save(sess, "/tmp/model.ckpt")
                #print("Model saved in path: %s" % save_path)

                pass

def testing():
    # Read stred data
    batch = 50

    trained_tf_data_folder = os.path.join('/home/user170/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/', "trained_tf_data")
    save_file_name = 'facenet_00060-of-00060.meta'
    saver = tf.train.import_meta_graph(os.path.join(trained_tf_data_folder, save_file_name))
    
    sess = tf.Session()
    sess.__enter__()
    #with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint(trained_tf_data_folder))
    #summary_writer = tf.summary.FileWriter(os.path.join('summaries','facenet'), sess.graph) # Write tensorboard
    summary_writer = tbh.summary_writer_fn('loaded_session', sess)

    graph = tf.get_default_graph()
    input_dataset_placeholder = graph.get_tensor_by_name("placeholders/input:0")
    train_valid_placeholder = graph.get_tensor_by_name("placeholders/tv_mode_selector_placeholder:0")

    prediction = graph.get_tensor_by_name("predictions/ArgMax:0")


    def _read_image(filepath):
        # Convert filepath string to string tensor
        tf_filepath = tf.convert_to_tensor(filepath, dtype=tf.string)

        # Read .JPEG image
        tf_image_string = tf.read_file(tf_filepath)
        image_tf = tf.image.decode_jpeg(tf_image_string, channels=3)

        # Rescale image and convert to float
        image_tf = tf.to_float(image_tf)
        #image_tf = tf.image.resize_images(image_tf, [224, 224], method=tf.image.ResizeMethod.AREA, align_corners=True)
        image_tf = tf.image.resize_images(image_tf, [224, 224])
        image_tf = image_tf * (1./255) # Normalization
        image_tf = tf.expand_dims(image_tf, 0)

        return image_tf

    image_tf = _read_image("/home/user170/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/images/166/36.jpg")
    
    #sess.run(tf.global_variables_initializer())
    image_tf = sess.run(image_tf)

    for i in range(batch-1):
        if i < 1:
            stacked_image = np.vstack([image_tf, image_tf])
        else:
            stacked_image = np.vstack([stacked_image, image_tf])

    prev_time = time.time()
    result = sess.run(prediction, feed_dict={input_dataset_placeholder: stacked_image, train_valid_placeholder:False})
    
    print("Execution time: ", time.time() - prev_time)
    
    print(result)

    return result[0]

if __name__=="__main__":
    testing()
