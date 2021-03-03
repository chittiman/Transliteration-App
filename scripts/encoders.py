import tensorflow as tf
from tensorflow.keras import Model
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

from . import transliteration_tokenizers


class Encoder_GRU_1(Model):
    """1 Layered GRU based Encoder Model.
    Inherits from Keras model
    """

    def __init__(
        self,
        embed_size,
        hidden_size,
        linear_dropout,
        recurrent_dropout,
        weight_decay,
        src_tokenizer,
    ):
        """Initializes the model

        Args:
            embed_size (int): Embeeding size
            hidden_size (int): hidden size of recuurent layer
            linear_dropout (float): linear dropout of recurrent layer
            recurrent_dropout (float): recurrent dropout of recurrent layer
            weight_decay (float): weight decay
            src_tokenizer (hugging_face tokenizer): source i.e. romanized words tokenizer
        """
        super().__init__()

        src_vocab_size = src_tokenizer.get_vocab_size()
        src_pad_id = src_tokenizer.token_to_id("<pad>")
        regularizer = l2(weight_decay)
        assert src_pad_id == 0, "Source pad token id is not 0."
        self.rnn_type = "GRU"
        self.num_layers = 1
        self.src_embedding = Embedding(src_vocab_size, embed_size, mask_zero=True)
        self.encoder = Bidirectional(
            GRU(
                hidden_size,
                kernel_regularizer=regularizer,
                recurrent_regularizer=regularizer,
                dropout=linear_dropout,
                recurrent_dropout=recurrent_dropout,
                return_sequences=True,
                return_state=True,
            )
        )

    def call(self, src_ids, training=False):
        """Takes in source ids and encodes them

        Args:
            src_ids (Tensor): Source ids tensor of shape (batch_size, max_src_len)
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.

        Returns:
            List: Output from encoder - [sequences,[enc_hid],src_mask]
                :sequences - encoder recurrent network output
                    :shape - [batch_size, max_src_len, 2*hidden_size]
                    :dtype - float
                :enc_hid -final hidden state of recurrent network. Used in creating initial states of recurrent networks in decoder
                    :shape - [batch_size, 2*hidden_size]
                    :dtype - float
                :src_mask - source mask needed to propagate padding information through the model
                    :shape - [batch_size, max_src_len]
                    :dtype - bool

        """
        # deleting to save space, useful on gpu's
        src_embeds, src_mask = (
            self.src_embedding(src_ids),
            self.src_embedding.compute_mask(src_ids),
        )
        del src_ids
        sequences, enc_fwd, enc_bwd = self.encoder(
            inputs=src_embeds, mask=src_mask, training=training
        )
        del src_embeds
        # Conactenating forward and backward states to create a complete state
        enc_hid = tf.concat([enc_fwd, enc_bwd], axis=-1)
        del enc_fwd, enc_bwd
        encoder_output = [sequences, [enc_hid], src_mask]
        return encoder_output


class Encoder_LSTM_1(Model):
    """1 Layered LSTM based Encoder Model.
    Inherits from Keras model
    """

    def __init__(
        self,
        embed_size,
        hidden_size,
        linear_dropout,
        recurrent_dropout,
        weight_decay,
        src_tokenizer,
    ):
        """Initializes the model

        Args:
            embed_size (int): Embeeding size
            hidden_size (int): hidden size of recuurent layer
            linear_dropout (float): linear dropout of recurrent layer
            recurrent_dropout (float): recurrent dropout of recurrent layer
            weight_decay (float): weight decay
            src_tokenizer (hugging_face tokenizer): source i.e. romanized words tokenizer
        """
        super().__init__()

        src_vocab_size = src_tokenizer.get_vocab_size()
        src_pad_id = src_tokenizer.token_to_id("<pad>")
        regularizer = l2(weight_decay)
        assert src_pad_id == 0, "Pad token id is not 0."
        self.rnn_type = "LSTM"
        self.num_layers = 1
        self.src_embedding = Embedding(src_vocab_size, embed_size, mask_zero=True)
        self.encoder = Bidirectional(
            LSTM(
                hidden_size,
                kernel_regularizer=regularizer,
                recurrent_regularizer=regularizer,
                dropout=linear_dropout,
                recurrent_dropout=recurrent_dropout,
                return_sequences=True,
                return_state=True,
            )
        )

    def call(self, src_ids, training=False):
        """Takes in source ids and encodes them

        Args:
            src_ids (Tensor): Source ids tensor of shape (batch_size, max_src_len)
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.

        Returns:
            List: Output from encoder - [sequences,[enc_hid,enc_cell],src_mask]
                :sequences - encoder recurrent network output
                    :shape - [batch_size, max_src_len, 2*hidden_size]
                    :dtype - float
                :enc_hid -final hidden state of recurrent network. Used in creating initial states of recurrent networks in decoder
                    :shape - [batch_size, 2*hidden_size]
                    :dtype - float
                :enc_cell -final cell state of recurrent network. Used in creating initial states of recurrent networks in decoder
                    :shape - [batch_size, 2*hidden_size]
                    :dtype - float
                :src_mask - source mask needed to propagate padding information through the model
                    :shape - [batch_size, max_src_len]
                    :dtype - bool

        """

        src_embeds, src_mask = (
            self.src_embedding(src_ids),
            self.src_embedding.compute_mask(src_ids),
        )
        del src_ids
        sequences, enc_hid_fwd, enc_cell_fwd, enc_hid_bwd, enc_cell_bwd = self.encoder(
            inputs=src_embeds, mask=src_mask, training=training
        )
        del src_embeds
        enc_hid = tf.concat([enc_hid_fwd, enc_hid_bwd], axis=1)
        del enc_hid_fwd, enc_hid_bwd
        enc_cell = tf.concat([enc_cell_fwd, enc_cell_bwd], axis=-1)
        del enc_cell_fwd, enc_cell_bwd
        encoder_output = [sequences, [enc_hid, enc_cell], src_mask]
        return encoder_output


class Encoder_GRU_2(Model):
    """2 Layered GRU based Encoder Model.
    Inherits from Keras model
    """

    def __init__(
        self,
        embed_size,
        hidden_size,
        linear_dropout,
        recurrent_dropout,
        weight_decay,
        src_tokenizer,
    ):
        """Initializes the model

        Args:
            embed_size (int): Embedding size
            hidden_size (int): hidden size of recurrent layer
            linear_dropout (float): linear dropout of recurrent layer
            recurrent_dropout (float): recurrent dropout of recurrent layer
            weight_decay (float): weight decay
            src_tokenizer (hugging_face tokenizer): source i.e. romanized words tokenizer
        """
        super().__init__()

        src_vocab_size = src_tokenizer.get_vocab_size()
        src_pad_id = src_tokenizer.token_to_id("<pad>")
        regularizer = l2(weight_decay)
        assert src_pad_id == 0, "Pad token id is not 0."
        self.rnn_type = "GRU"
        self.num_layers = 2
        self.src_embedding = Embedding(src_vocab_size, embed_size, mask_zero=True)
        self.encoder_1 = Bidirectional(
            GRU(
                hidden_size,
                kernel_regularizer=regularizer,
                recurrent_regularizer=regularizer,
                dropout=linear_dropout,
                recurrent_dropout=recurrent_dropout,
                return_sequences=True,
                return_state=True,
            )
        )

        self.dropout = Dropout(linear_dropout)

        self.encoder_2 = GRU(
            hidden_size,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
            dropout=linear_dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            return_state=True,
        )

    def call(self, src_ids, training=False):
        """Takes in source ids and encodes them

        Args:
            src_ids (Tensor): Source ids tensor of shape (batch_size, max_src_len)
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.

        Returns:
            List: Output from encoder - [sequences,[enc_hid],src_mask]
                :sequences - encoder recurrent network output
                    :shape - [batch_size, max_src_len, 3*hidden_size]
                    :dtype - float
                :enc_hid -final hidden state of recurrent network. Used in creating initial states of recurrent networks in decoder
                    :shape - [batch_size, 3*hidden_size]
                    :dtype - float
                :src_mask - source mask needed to propagate padding information through the model
                    :shape - [batch_size, max_src_len]
                    :dtype - bool

        """
        src_embeds, src_mask = (
            self.src_embedding(src_ids),
            self.src_embedding.compute_mask(src_ids),
        )
        del src_ids
        sequences_1, enc_fwd_1, enc_bwd_1 = self.encoder_1(
            inputs=src_embeds, mask=src_mask, training=training
        )
        del src_embeds
        sequences_1_drop = self.dropout(sequences_1, training=training)
        sequences_2, enc_fwd_2 = self.encoder_2(
            inputs=sequences_1_drop, mask=src_mask, training=training
        )
        del sequences_1_drop
        enc_hid = tf.concat([enc_fwd_1, enc_bwd_1, enc_fwd_2], axis=-1)
        del enc_fwd_1, enc_bwd_1, enc_fwd_2
        sequences = tf.concat([sequences_1, sequences_2], axis=-1)
        encoder_output = [sequences, [enc_hid], src_mask]
        return encoder_output


class Encoder_LSTM_2(Model):
    """2 Layered LSTM based Encoder Model.
    Inherits from Keras model
    """

    def __init__(
        self,
        embed_size,
        hidden_size,
        linear_dropout,
        recurrent_dropout,
        weight_decay,
        src_tokenizer,
    ):
        """Initializes the model

        Args:
            embed_size (int): Embedding size
            hidden_size (int): hidden size of recurrent layer
            linear_dropout (float): linear dropout of recurrent layer
            recurrent_dropout (float): recurrent dropout of recurrent layer
            weight_decay (float): weight decay
            src_tokenizer (hugging_face tokenizer): source i.e. romanized words tokenizer
        """
        super().__init__()

        src_vocab_size = src_tokenizer.get_vocab_size()
        src_pad_id = src_tokenizer.token_to_id("<pad>")
        regularizer = l2(weight_decay)
        assert src_pad_id == 0, "Pad token id is not 0."
        self.rnn_type = "LSTM"
        self.num_layers = 2
        self.src_embedding = Embedding(
            src_vocab_size, embed_size, mask_zero=True, input_shape=(None, None)
        )
        self.encoder_1 = Bidirectional(
            LSTM(
                hidden_size,
                kernel_regularizer=regularizer,
                recurrent_regularizer=regularizer,
                dropout=linear_dropout,
                recurrent_dropout=recurrent_dropout,
                return_sequences=True,
                return_state=True,
            )
        )

        self.dropout = Dropout(linear_dropout)

        self.encoder_2 = LSTM(
            hidden_size,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
            dropout=linear_dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            return_state=True,
        )

    def call(self, src_ids, training=False):
        """Takes in source ids and encodes them

        Args:
            src_ids (Tensor): Source ids tensor of shape (batch_size, max_src_len)
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.

        Returns:
            List: Output from encoder - [sequences,[enc_hid],src_mask]
                :sequences - encoder recurrent network output
                    :shape - [batch_size, max_src_len, 3*hidden_size]
                    :dtype - float
                :enc_hid -final hidden state of recurrent network. Used in creating initial states of recurrent networks in decoder
                    :shape - [batch_size, 3*hidden_size]
                    :dtype - float
                :enc_cell -final  cell state of recurrent network. Used in creating initial states of recurrent networks in decoder
                    :shape - [batch_size, 3*hidden_size]
                    :dtype - float
                :src_mask - source mask needed to propagate padding information through the model
                    :shape - [batch_size, max_src_len]
                    :dtype - bool

        """
        src_embeds, src_mask = (
            self.src_embedding(src_ids),
            self.src_embedding.compute_mask(src_ids),
        )
        del src_ids
        (
            sequences_1,
            enc_hid_fwd_1,
            enc_cell_fwd_1,
            enc_hid_bwd_1,
            enc_cell_bwd_1,
        ) = self.encoder_1(inputs=src_embeds, mask=src_mask, training=training)

        del src_embeds
        sequences_1_drop = self.dropout(sequences_1, training=training)
        sequences_2, enc_hid_fwd_2, enc_cell_fwd_2 = self.encoder_2(
            inputs=sequences_1_drop, mask=src_mask, training=training
        )
        del sequences_1_drop
        enc_hid = tf.concat([enc_hid_fwd_1, enc_hid_bwd_1, enc_hid_fwd_2], axis=-1)
        del enc_hid_fwd_1, enc_hid_bwd_1, enc_hid_fwd_2
        enc_cell = tf.concat([enc_cell_fwd_1, enc_cell_bwd_1, enc_cell_fwd_2], axis=-1)
        del enc_cell_fwd_1, enc_cell_bwd_1, enc_cell_fwd_2
        sequences = tf.concat([sequences_1, sequences_2], axis=-1)
        encoder_output = [sequences, [enc_hid, enc_cell], src_mask]
        return encoder_output
