import sys

from tensorflow.python.estimator import keras

import generation_utilities
from model.utils_tf import model_argmax

sys.path.append("../../")

import numpy as np
import tensorflow as tf
import copy

def compute_grad(x, model, loss_func=keras.losses.binary_crossentropy):
    # compute the gradient of loss w.r.t input attributes

    x = tf.constant([x], dtype=tf.float32)
    y_pred = tf.cast(model(x) > 0.5, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = loss_func(y_pred, model(x))
    gradient = tape.gradient(loss, x)
    return gradient[0].numpy()
def global_generation(X, seeds, num_attribs, protected_attribs, constraint, model, max_iter, s_g):
    # global generation phase of ADF

    g_id = np.empty(shape=(0, num_attribs))
    all_gen_g = np.empty(shape=(0, num_attribs))
    try_times = 0
    g_num = len(seeds)
    for i in range(g_num):
        x1 = seeds[i].copy()
        for _ in range(max_iter):
            try_times += 1
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            if generation_utilities.is_discriminatory(x1, similar_x1, model):
                g_id = np.append(g_id, [x1], axis=0)
                break
            x2 = generation_utilities.max_diff(x1, similar_x1, model)
            grad1 = compute_grad(x1, model)
            grad2 = compute_grad(x2, model)
            direction = np.zeros_like(X[0])
            sign_grad1 = np.sign(grad1)
            sign_grad2 = np.sign(grad2)
            for attrib in range(num_attribs):
                if attrib not in protected_attribs and sign_grad1[attrib] == sign_grad2[attrib]:
                    direction[attrib] = sign_grad1[attrib]
            x1 = x1 + s_g * direction
            x1 = generation_utilities.clip(x1, constraint)
            all_gen_g = np.append(all_gen_g, [x1], axis=0)
    g_id = np.array(list(set([tuple(id) for id in g_id])))
    return g_id, all_gen_g, try_times


def local_generation(num_attribs, l_num, g_id, protected_attribs, constraint, model, s_l, epsilon):
    # local generation phase of ADF

    direction = [-1, 1]
    l_id = np.empty(shape=(0, num_attribs))
    all_gen_l = np.empty(shape=(0, num_attribs))
    try_times = 0
    for x1 in g_id:
        x0 = x1.copy()
        for _ in range(l_num):
            try_times += 1
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            x2 = generation_utilities.find_pair(x1, similar_x1, model)
            grad1 = compute_grad(x1, model)
            grad2 = compute_grad(x2, model)
            p = generation_utilities.normalization(grad1, grad2, protected_attribs, epsilon)
            a = generation_utilities.random_pick(p)
            s = generation_utilities.random_pick([0.5, 0.5])
            x1[a] = x1[a] + direction[s] * s_l
            x1 = generation_utilities.clip(x1, constraint)
            all_gen_l = np.append(all_gen_l, [x1], axis=0)
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            if generation_utilities.is_discriminatory(x1, similar_x1, model):
                l_id = np.append(l_id, [x1], axis=0)
            else:
                x1 = x0.copy()
    l_id = np.array(list(set([tuple(id) for id in l_id])))
    return l_id, all_gen_l, try_times



def individual_discrimination_generation(X, seeds, protected_attribs, constraint, model, l_num, max_iter=10, s_g=1.0,
                                         s_l=1.0, epsilon=1e-6):
    # return non-duplicated individual discriminatory instances generated, non-duplicate instances generated and total number of search iterations

    num_attribs = len(X[0])
    g_id, gen_g, g_gen_num = global_generation(X, seeds, num_attribs, protected_attribs, constraint, model, max_iter,
                                               s_g)
    l_id, gen_l, l_gen_num = local_generation(num_attribs, l_num, g_id, protected_attribs, constraint, model, s_l,
                                              epsilon)
    all_id = np.append(g_id, l_id, axis=0)
    all_gen = np.append(gen_g, gen_l, axis=0)
    all_id_nondup = np.array(list(set([tuple(id) for id in all_id])))
    all_gen_nondup = np.array(list(set([tuple(gen) for gen in all_gen])))
    return all_id_nondup, all_gen_nondup, g_gen_num + l_gen_num

def check_for_error_condition(conf, sess, x, preds, t, sens):
    """
    Check whether the test case is an individual discriminatory instance
    """
    t = np.array(t).astype("int")
    label = model_argmax(sess, x, preds, np.array([t]))

    # check for all the possible values of sensitive feature
    for val in range(conf.input_bounds[sens-1][0], conf.input_bounds[sens-1][1]+1):
        if val != int(t[sens-1]):
            tnew = copy.deepcopy(t)
            tnew[sens-1] = val
            label_new = model_argmax(sess, x, preds, np.array([tnew]))
            if label_new != label:
                return val
    return t[sens - 1]




