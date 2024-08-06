from distutils.version import LooseVersion
import numpy as np
import os
from six.moves import xrange
import tensorflow as tf
import time
import warnings
import math
import sys

from tensorflow_estimator.python.estimator.canned.timeseries import model

sys.path.append('../../')



def model_loss(y, model, mean=True):
    op = model.op
    if op.type == "Softmax":
        logits, = op.inputs
    else:
        logits = model

    out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

    if mean:
        out = tf.reduce_mean(out)
    return out

def initialize_uninitialized_global_variables(sess):
    # List all global variables
    global_vars = tf.global_variables()

    # Find initialized status for all variables
    is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
    is_initialized = sess.run(is_var_init)

    # List all variables that were not initialized previously
    not_initialized_vars = [var for (var, init) in
                            zip(global_vars, is_initialized) if not init]

    # Initialize all uninitialized variables found, if any
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def _ArgsWrapper(param):
    pass


def batch_indices(batch, param, batch_size):
    pass


def model_train(sess, x, y, predictions, X_train, Y_train, save=False,
                predictions_adv=None, init_all=True, evaluate=None,
                feed=None, args=None, rng=None):

    args = _ArgsWrapper(args or {})

    # Check that necessary arguments were given (see doc above)
    assert args.nb_epochs, "Number of epochs was not given in args dict"
    assert args.learning_rate, "Learning rate was not given in args dict"
    assert args.batch_size, "Batch size was not given in args dict"

    if save:
        assert args.train_dir, "Directory for save was not given in args dict"
        assert args.filename, "Filename for save was not given in args dict"

    if rng is None:
        rng = np.random.RandomState()

    # Define loss
    loss = model_loss(y, predictions)
    if predictions_adv is not None:
        loss = (loss + model_loss(y, predictions_adv)) / 2

    train_step = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_op):
        train_step = train_step.minimize(loss)

    with sess.as_default():
        if hasattr(tf, "global_variables_initializer"):
            if init_all:
                tf.global_variables_initializer().run()
            else:
                initialize_uninitialized_global_variables(sess)
        else:
            warnings.warn("Update your copy of tensorflow; future versions of "
                          "guardai_util may drop support for this version.")
            sess.run(tf.initialize_all_variables())

        for epoch in xrange(args.nb_epochs):
            # Compute number of batches
            nb_batches = int(math.ceil(float(len(X_train)) / args.batch_size))
            assert nb_batches * args.batch_size >= len(X_train)

            # Indices to shuffle training set
            index_shuf = list(range(len(X_train)))
            rng.shuffle(index_shuf)

            prev = time.time()
            for batch in range(nb_batches):

                # Compute batch start and end indices
                start, end = batch_indices(
                    batch, len(X_train), args.batch_size)

                # Perform one training step
                feed_dict = {x: X_train[index_shuf[start:end]],
                             y: Y_train[index_shuf[start:end]]}
                if feed is not None:
                    feed_dict.update(feed)
                train_step.run(feed_dict=feed_dict)
            assert end >= len(X_train)  # Check that all examples were used
            cur = time.time()
            print("Epoch " + str(epoch) + " took " +
                         str(cur - prev) + " seconds")

            if evaluate is not None:
                evaluate()

        if save:
            train_dir = os.path.join(args.train_dir, str(args.nb_epochs - 1))
            if not os._exists(train_dir):
                os.makedirs(train_dir)
            save_path = os.path.join(train_dir, args.filename)
            saver = tf.train.Saver()
            saver.save(sess, save_path)
            print("Completed model training and saved at: " +
                         str(save_path))
        else:
            print("Completed model training.")

    return True

def model_eval(sess, x, y, predictions, X_test=None, Y_test=None,feed=None, args=None):

    args = _ArgsWrapper(args or {})

    assert args.batch_size, "Batch size was not given in args dict"
    if X_test is None or Y_test is None:
        raise ValueError("X_test argument and Y_test argument "
                         "must be supplied.")

    # Define accuracy symbolically
    if LooseVersion(tf.__version__) >= LooseVersion('1.0.0'):
        correct_preds = tf.equal(tf.argmax(y, axis=-1),
                                 tf.argmax(predictions, axis=-1))
    else:
        correct_preds = tf.equal(tf.argmax(y, axis=tf.rank(y) - 1),
                                 tf.argmax(predictions,
                                           axis=tf.rank(predictions) - 1))

    # Init result var
    accuracy = 0.0

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))
        assert nb_batches * args.batch_size >= len(X_test)

        X_cur = np.zeros((args.batch_size,) + X_test.shape[1:],
                         dtype=X_test.dtype)
        Y_cur = np.zeros((args.batch_size,) + Y_test.shape[1:],
                         dtype=Y_test.dtype)
        for batch in range(nb_batches):
            if batch % 100 == 0 and batch > 0:
                print("Batch " + str(batch))


            # It's acceptable to repeat during training, but not eval.
            start = batch * args.batch_size
            end = min(len(X_test), start + args.batch_size)

            # The last batch may be smaller than all others. This should not
            # affect the accuarcy disproportionately.
            cur_batch_size = end - start
            X_cur[:cur_batch_size] = X_test[start:end]
            Y_cur[:cur_batch_size] = Y_test[start:end]
            feed_dict = {x: X_cur, y: Y_cur}
            if feed is not None:
                feed_dict.update(feed)
            cur_corr_preds = correct_preds.eval(feed_dict=feed_dict)

            accuracy += cur_corr_preds[:cur_batch_size].sum()

        assert end >= len(X_test)

        # Divide by number of examples to get final value
        accuracy /= len(X_test)

    return accuracy

def model_argmax(sess, x, predictions, samples, feed=None):
    """
    Helper function that computes the current class prediction
    :param sess: TF session
    :param x: the input placeholder
    :param predictions: the model's symbolic output
    :param samples: numpy array with input samples (dims must match x)
    :param feed: An optional dictionary that is appended to the feeding
             dictionary before the session runs. Can be used to feed
             the learning phase of a Keras model for instance
    :return: the argmax output of predictions, i.e. the current predicted class
    """
    feed_dict = {x: samples}
    if feed is not None:
        feed_dict.update(feed)
    probabilities = sess.run(predictions, feed_dict)

    if samples.shape[0] == 1:
        return np.argmax(probabilities)
    else:
        return np.argmax(probabilities, axis=1)

def model_prediction(sess, x, predictions, samples, feed=None, batch_size= model.batchsize):

    nb_classes = 2
    nb_batches = int(math.ceil(float(samples.shape[0]) / batch_size))

    pros_all=np.zeros(shape=(samples.shape[0],nb_classes), dtype='float32')

    for batch in range(nb_batches):
        start=batch*batch_size
        end=(batch+1)*batch_size
        if end>samples.shape[0]:
            end=samples.shape[0]

        feed_dict = {x: samples[start:end]}
        if feed is not None:
            feed_dict.update(feed)
        pros=sess.run(predictions, feed_dict)

        for i in range(start,end):
            pros_all[i]=pros[i-start]
    return pros_all
