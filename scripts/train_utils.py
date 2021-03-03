import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.math import reduce_sum,argmax,equal

def msk_loss(y_true,y_probs,tgt_pad_index = 0):
    """Calculates cross entropy by ignoring loss items corresponding to pad tokens

    Args:
        y_true (Tensor): Ground truth tensor of shape (batch_size, time_steps)
        y_probs (Tensor): Probabilities output by decoder (batch_size,time_steps,tgt_vocab_size)
        tgt_pad_index (int, optional): Pad token id. Defaults to 0.

    Returns:
        Tensor: Masked loss of shape of (1,)
    """
    #If wts are not supplied, then the first column of y_true has just start_id i.e 1. So, multiplication with
    # 1 doesnt affect the loss. If the first col is wts then losses are accordingly scaled
    wts = y_true[:,0]
    y_true = tf.cast(y_true[:,1:], tf.int64)#y_true[:,1:]
    mask = tf.cast((y_true != tgt_pad_index),tf.float32)
    num_words = reduce_sum(mask)
    unmasked_loss_matrix = sparse_categorical_crossentropy(y_true, y_probs)
    masked_loss_matrix = unmasked_loss_matrix*mask
    masked_loss_matrix *= tf.reshape(wts,(-1,1))
    masked_loss = reduce_sum(masked_loss_matrix)/num_words
    return masked_loss

def msk_acc(y_true,y_probs,tgt_pad_index = 0):
    """Calculates accuracy by ignoring accuracy scores corresponding to pad tokens

    Args:
        y_true (Tensor): Ground truth tensor of shape (batch_size, time_steps)
        y_probs (Tensor): Probabilities output by decoder (batch_size,time_steps,tgt_vocab_size)
        tgt_pad_index (int, optional): Pad token id. Defaults to 0.

    Returns:
        Tensor: Masked accuracy of shape of (1,)
    """
    #If wts are not supplied, then the first column of y_true has just start_id i.e 1. So, multiplication with
    # 1 doesnt affect the loss. If the first col is wts then losses are accordingly scaled
    y_true = tf.cast(y_true[:,1:], tf.int64)
    mask = tf.cast((y_true != tgt_pad_index),tf.int64)
    num_words = reduce_sum(mask)
    y_preds = argmax(y_probs, axis = -1)
    unmasked_accuracy_matrix = tf.cast(equal(y_preds,y_true), tf.int64)#equal(y_preds,y_true)#
    masked_accuracy_matrix = unmasked_accuracy_matrix*mask
    masked_accuracy = reduce_sum(masked_accuracy_matrix)/num_words
    return masked_accuracy