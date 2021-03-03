import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.backend import batch_dot
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    Layer,
)
from tensorflow.keras.regularizers import l2
from tensorflow.nn import softmax

from . import transliteration_tokenizers


class Luong_Attention(Layer):
    """Class implementing Luong style Attention

    Args:
        Inherits from Keras Layer
    """

    def __init__(self, dec_size, regularizer=None):
        """Initializes the layer

        Args:
            dec_size (int): Encoding size of sequences output by encoder network
            regularizer (regularizer, optional): L2 regularizer. Defaults to None.
        """
        super().__init__()
        self.dec_size = dec_size
        self.regularizer = regularizer
        self.enc_trnsfrmr = Dense(
            self.dec_size, kernel_regularizer=self.regularizer, use_bias=False
        )
        self.final_trnsfrmr = Dense(
            self.dec_size,
            kernel_regularizer=self.regularizer,
            use_bias=False,
            activation="tanh",
        )

    def call(self, enc_states, dec_states, src_mask):
        """Calculates attention states based on attention scores

        Args:
            enc_states (Tensor): Final states of encoder of shape (batch_size, src_len, enc_size)
            dec_states (Tensor): Output of recurrent networks of decoder (batch_size, tgt_len, dec_size)
            src_mask (Tensor): source mask needed to propagate padding information through the model of
            shape [batch_size, max_src_len]

        Returns:
            Tensor: Attention states of shape (batch_size, tgt_len, dec_size)
        """
        enc_trnsfms = tf.transpose(self.enc_trnsfrmr(enc_states), perm=[0, 2, 1])
        # (batch_size, dec_size, src_len) =  (batch_size, src_len, enc_size)
        atten_scores = batch_dot(dec_states, enc_trnsfms)
        # (batch_size, tgt_len, src_len)  =  (batch_size, tgt_len, dec_size), (batch_size, dec_size, src_len)
        atten_scores_masked = tf.where(
            tf.expand_dims(src_mask, axis=1), atten_scores, [float("-inf")]
        )
        # atten_scores_masked = (batch_size, 1 , src_len) ,(batch_size, tgt_len, src_len), (1,)
        del src_mask
        atten_wts = tf.nn.softmax(atten_scores_masked, axis=-1)
        # (batch_size, tgt_len, src_len)  =  (batch_size, tgt_len, src_len)
        del atten_scores_masked
        enc_states_reshaped = tf.expand_dims(enc_states, axis=1)
        # (batch_size  ,   1    ,src_len, enc_size)   = (batch_size, src_len, enc_size)
        del enc_states
        atten_wts_reshaped = tf.expand_dims(atten_wts, axis=-1)
        # (batch_size, tgt_len, src_len, 1) = (batch_size, tgt_len, src_len)
        del atten_wts
        enc_states_weighted = tf.math.multiply(enc_states_reshaped, atten_wts_reshaped)
        # (batch_size  ,tgt_len,src_len, enc_size)  = (batch_size,1,src_len, enc_size),(batch_size, tgt_len, src_len, 1)
        del enc_states_reshaped, atten_wts_reshaped
        enc_states_summed = tf.reduce_sum(enc_states_weighted, axis=2)
        # (batch_size,tgt_len, enc_size)  =  (batch_size  ,tgt_len,src_len, enc_size)
        del enc_states_weighted
        context_vetcor = tf.concat([enc_states_summed, dec_states], axis=-1)
        # (batch_size, tgt_len, enc_size + dec_size)  =   (batch_size,tgt_len, enc_size), (batch_size, tgt_len, dec_size)
        del enc_states_summed, dec_states
        final_atten_states = self.final_trnsfrmr(context_vetcor)
        # (batch_size, tgt_len, dec_size)  =  (batch_size, tgt_len, enc_size + dec_size)
        return final_atten_states

    # def get_config(self):
    #    return {"dec_size": self.dec_size}
