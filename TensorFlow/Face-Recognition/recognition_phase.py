from facenet import FacenetModel
import os
import tensorflow as tf
import sys
import numpy as np
from tqdm import tqdm
import tensorboard_helper as tbh

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('image_path', '/home/shared-data/SJC_Dev/Projects/SJC_Git/Face-Detector/SJC-Face-Data/', "root directory including images and tfrecords")


# Set tensorboard folder name
summary_name_str = 'facenet2'


""" ################
# Hyper parameters #
################ """
epoch = 60
batch = 50
learning_rate = 0.001

num_outputs = len(list(
    filter(lambda y: os.path.isdir(y), list(
        map(lambda x: os.path.join(FLAGS.image_path+"images", x), os.listdir(FLAGS.image_path+"images"))))))

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


def training():
    model(FLAGS.image_path, 224, 224)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    config_proto = tf.ConfigProto()
    #config_proto.gpu_options.allow_growth = True
    config_proto.gpu_options.per_process_gpu_memory_fraction = 0.8
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
                print("Processing: ", end=""); sys.stdout.flush()
                while True:
                    count += 1
                    print(".", end=""); sys.stdout.flush()
                    
                    extracted_data = sess.run(model.get_next_in_interators, feed_dict={model.handle_placeholder: training_handle})
                    
                    if len(extracted_data[1]) > 1:
                        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        loss, _, _ = sess.run([model.loss, model.loss_minimizer, extra_update_ops], feed_dict={model.input_dataset_placeholder:extracted_data[0], model.labels_dataset_placeholder:extracted_data[1], model.train_valid_placeholder:True})
                        pred = sess.run(model.predictions, feed_dict={model.input_dataset_placeholder:extracted_data[0], model.train_valid_placeholder:True})
                        total_accuracy += np.sum(np.equal(pred, extracted_data[1])) / batch

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
                print("Processing: ", end=""); sys.stdout.flush()
                while True:
                    count += 1
                    print(".", end=""); sys.stdout.flush()

                    extracted_data = sess.run(model.get_next_in_interators, feed_dict={model.handle_placeholder: validation_handle})

                    if len(extracted_data[1]) > 1:
                        pred = sess.run(model.predictions, feed_dict={model.input_dataset_placeholder:extracted_data[0], model.train_valid_placeholder:False})
                        total_accuracy += np.sum(np.equal(pred, extracted_data[1])) / batch

            except tf.errors.OutOfRangeError:
                print("\n{0} %".format((total_accuracy/count)*100)); sys.stdout.flush()

                # Execute the summaries defined above
                tbh.write_accuracy(summary_writer, sess, total_accuracy, count, epoch)

                # Save the variables to disk.
                save_path = saver.save(sess, '/tmp/%s_%03d-of-%03d.ckpt' % ("facenet", current_epoch_int, epoch))
                #save_path = saver.save(sess, "/tmp/model.ckpt")
                print("Model saved in path: %s" % save_path)

                pass

def testing():
     model(FLAGS.image_path, 224, 224)

     
    
def main(argv):
    training()

if __name__=="__main__":
    tf.app.run(main)
