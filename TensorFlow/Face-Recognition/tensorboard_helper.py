import tensorflow as tf
import os


# Create a folder to save tensorboard data
if not os.path.exists('summaries'):
    os.mkdir('summaries')
#if not os.path.exists(os.path.join('summaries','facenet')):
#    os.mkdir(os.path.join('summaries','facenet'))

with tf.name_scope('training_performance'):
    # Summaries need to be displayed
    # Whenever you need to record the loss, feed the mean loss to this placeholder
    train_loss_placeholder = tf.placeholder(tf.float32,shape=None,name='loss_summary')
    # Create a scalar summary object for the loss so it can be displayed
    train_loss_summary = tf.summary.scalar('loss', train_loss_placeholder)

    # Whenever you need to record the loss, feed the mean test accuracy to this placeholder
    train_accuracy_placeholder = tf.placeholder(tf.float32,shape=None, name='accuracy_summary')
    # Create a scalar summary object for the accuracy so it can be displayed
    train_accuracy_summary = tf.summary.scalar('accuracy', train_accuracy_placeholder)

    # Merge all summaries together
    train_performance_summaries = tf.summary.merge([train_loss_summary, train_accuracy_summary])

with tf.name_scope('validation_performance'):
    # Summaries need to be displayed

    # Whenever you need to record the loss, feed the mean test accuracy to this placeholder
    valid_accuracy_placeholder = tf.placeholder(tf.float32,shape=None, name='accuracy_summary')
    # Create a scalar summary object for the accuracy so it can be displayed
    valid_accuracy_summary = tf.summary.scalar('accuracy', valid_accuracy_placeholder)

    # Merge all summaries together
    valid_performance_summaries = tf.summary.merge([valid_accuracy_summary])

def summary_writer_fn(_summary_name_str, _sess_tf):
    summary_writer = tf.summary.FileWriter(os.path.join('summaries', _summary_name_str), _sess_tf.graph) # Write tensorboard
    
    return summary_writer

def write_accuracy(_summary_writer, _sess_tf, _total_accuracy_int, _count_int, _epoch_int):
    valid_summary = _sess_tf.run(valid_performance_summaries, feed_dict={valid_accuracy_placeholder:(_total_accuracy_int/_count_int)*100})
    _summary_writer.add_summary(valid_summary, _epoch_int)
    _summary_writer.flush()

def write_loss_and_accuracy_summary_fn(_summary_writer, _sess_tf, _loss_int, _total_accuracy_int, _count_int, _epoch_int):
    train_summary = _sess_tf.run(train_performance_summaries, feed_dict={train_loss_placeholder:_loss_int, train_accuracy_placeholder:(_total_accuracy_int/_count_int)*100})
    _summary_writer.add_summary(train_summary, _epoch_int)
    _summary_writer.flush()