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
from tensorflow.math import argmax

from . import transliteration_tokenizers
from .layers import Luong_Attention


class Decoder_Seq2seq_GRU_1(Model):
    """1 layered GRU based Seq2seq type decoder
    Inherits from Keras model
    """

    def __init__(
        self,
        embed_size,
        hidden_size,
        linear_dropout,
        recurrent_dropout,
        weight_decay,
        tgt_tokenizer,
    ):
        """Initializes the model

        Args:
            embed_size (int): Embedding size
            hidden_size (int): hidden size of recuurent layer
            linear_dropout (float): linear dropout of recurrent layer
            recurrent_dropout (float): recurrent dropout of recurrent layer
            weight_decay (float): weight decay
            tgt_tokenizer (hugging_face tokenizer): target i.e. native script words tokenizer
        """
        super().__init__()

        tgt_vocab_size = tgt_tokenizer.get_vocab_size()
        tgt_pad_id = tgt_tokenizer.token_to_id("<pad>")
        assert tgt_pad_id == 0, "Target pad token id is not 0."

        regularizer = l2(weight_decay)
        # self.tgt_start_id = #self.tgt_tokenizer.token_to_id('<s>')
        # self.tgt_end_id = #self.tgt_tokenizer.token_to_id('</s>')

        self.rnn_type = "GRU"
        self.decoder_type = "Seq2seq"
        self.num_layers = 1
        self.enc_hid2dec_hid_init_1 = Dense(
            hidden_size, kernel_regularizer=regularizer, use_bias=False
        )
        self.tgt_embedding = Embedding(tgt_vocab_size, embed_size, mask_zero=True)
        self.decoder_1 = GRU(
            hidden_size,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
            dropout=linear_dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            return_state=True,
        )
        self.decode2vocab = Dense(
            tgt_vocab_size, kernel_regularizer=regularizer, activation="softmax"
        )

    def call(self, tgt_ids, encoder_output, training=False):
        """Takes in Encoder output, target ids and produce probabilities of target vocab tokens
        for each sample and for each time step

        Args:
            tgt_ids (Tensor): Target ids tensor of shape [batch_size, max_tgt_len]
            encoder_output (Nested lists of tensors): Output of encoder
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.

        Returns:
            Tensor: Target token probabilities of shape [batch_size, max_tgt_len, tgt_vocab_size]
        """
        # tgt_ids = tgt_ids[:,:-1]
        sequences, enc_final_state, src_mask = encoder_output
        del sequences, src_mask
        initial_states = self.initialize_states(enc_final_state)
        tgt_embeds, tgt_mask = (
            self.tgt_embedding(tgt_ids),
            self.tgt_embedding.compute_mask(tgt_ids),
        )
        del tgt_ids
        dec_sequences_1, dec_hid_1 = self.decoder_1(
            inputs=tgt_embeds,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[0],
        )
        del tgt_embeds, dec_hid_1, initial_states
        dec_vocab_probs = self.decode2vocab(dec_sequences_1)
        return dec_vocab_probs

    def initialize_states(self, encoder_final_state):
        """Initializes states of recurrent networks of model from encoder final states

        Args:
            encoder_final_state (tensor): List of final states of encoder

        Returns:
            Nested list of tensors: Tensors to initialize states of model's recurrent networks
        """
        enc_hid = encoder_final_state[0]
        dec_hid_init_1 = self.enc_hid2dec_hid_init_1(enc_hid)
        del encoder_final_state, enc_hid
        initial_state_1 = [dec_hid_init_1]
        initial_states = [initial_state_1]
        return initial_states

    def decode_step(self, tgt_in, initial_states, encoder_output, training=False):
        """Performs a single decoding step

        Args:
            tgt_in (Tensor): tgt_ids for one time step of shape [batch_size,1]
            initial_states (Nested lists of Tensors): States from previous time step to be fed to recurrent networks
            encoder_output (Nested lists of tensor): Output of the encoder
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.. Defaults to False.

        Returns:
            tgt_out: tgt_ids for next time step obtained through greedy search of shape [batch_size, 1]
            final_states: (Nested lists of Tensors) : States to be fed to recurrent networks in next decoding time
            step
            dec_vob_probs: Target token probabilities tensor of shape [batch_size, 1, tgt_vocab_size]
        """
        tgt_embeds_in, tgt_mask = (
            self.tgt_embedding(tgt_in),
            self.tgt_embedding.compute_mask(tgt_in),
        )
        dec_sequences_1, hid_out_1 = self.decoder_1(
            inputs=tgt_embeds_in,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[0],
        )
        dec_vocab_probs = self.decode2vocab(dec_sequences_1)
        tgt_out = argmax(dec_vocab_probs, axis=-1)
        final_state_1 = [hid_out_1]
        final_states = [final_state_1]
        return tgt_out, final_states, dec_vocab_probs


class Decoder_Attention_GRU_1(Model):
    """1 layered GRU based Attention type decoder
    Inherits from Keras model
    """

    def __init__(
        self,
        embed_size,
        hidden_size,
        linear_dropout,
        recurrent_dropout,
        weight_decay,
        tgt_tokenizer,
    ):
        """Initializes the model

        Args:
            embed_size (int): Embedding size
            hidden_size (int): hidden size of recuurent layer
            linear_dropout (float): linear dropout of recurrent layer
            recurrent_dropout (float): recurrent dropout of recurrent layer
            weight_decay (float): weight decay
            tgt_tokenizer (hugging_face tokenizer): target i.e. native script words tokenizer
        """
        super().__init__()

        tgt_vocab_size = tgt_tokenizer.get_vocab_size()
        tgt_pad_id = tgt_tokenizer.token_to_id("<pad>")
        assert tgt_pad_id == 0, "Target pad token id is not 0."

        regularizer = l2(weight_decay)
        # self.tgt_start_id = #self.tgt_tokenizer.token_to_id('<s>')
        # self.tgt_end_id = #self.tgt_tokenizer.token_to_id('</s>')
        self.rnn_type = "GRU"
        self.decoder_type = "Attention"
        self.num_layers = 1
        self.enc_hid2dec_hid_init_1 = Dense(
            hidden_size, kernel_regularizer=regularizer, use_bias=False
        )

        self.tgt_embedding = Embedding(tgt_vocab_size, embed_size, mask_zero=True)
        self.decoder_1 = GRU(
            hidden_size,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
            dropout=linear_dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            return_state=True,
        )
        self.attention = Luong_Attention(hidden_size, regularizer)
        self.decode2vocab = Dense(
            tgt_vocab_size, kernel_regularizer=regularizer, activation="softmax"
        )

    def call(self, tgt_ids, encoder_output, training=False):
        """Takes in Encoder output, target ids and produce probabilities of target vocab tokens
        for each sample and for each time step

        Args:
            tgt_ids (Tensor): Target ids tensor of shape [batch_size, max_tgt_len]
            encoder_output (Nested lists of tensors): Output of encoder
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.

        Returns:
            Tensor: Target token probabilities of shape [batch_size, max_tgt_len, tgt_vocab_size]
        """
        # tgt_ids = tgt_ids[:,:-1]
        sequences, enc_final_state, src_mask = encoder_output
        initial_states = self.initialize_states(enc_final_state)
        tgt_embeds, tgt_mask = (
            self.tgt_embedding(tgt_ids),
            self.tgt_embedding.compute_mask(tgt_ids),
        )
        del tgt_ids
        dec_sequences_1, dec_hid_1 = self.decoder_1(
            inputs=tgt_embeds,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[0],
        )
        del tgt_embeds, dec_hid_1, initial_states
        attention_states = self.attention(sequences, dec_sequences_1, src_mask)
        del sequences, dec_sequences_1, src_mask
        dec_vocab_probs = self.decode2vocab(attention_states)
        return dec_vocab_probs

    def initialize_states(self, encoder_final_state):
        """Initializes states of recurrent networks of model from encoder final states

        Args:
            encoder_final_state (tensor): List of final states of encoder

        Returns:
            Nested list of tensors: Tensors to initialize states of model's recurrent networks
        """
        enc_hid = encoder_final_state[0]
        dec_hid_init_1 = self.enc_hid2dec_hid_init_1(enc_hid)
        del encoder_final_state, enc_hid
        initial_state_1 = [dec_hid_init_1]
        initial_states = [initial_state_1]
        return initial_states

    def decode_step(self, tgt_in, initial_states, encoder_output, training=False):
        """Performs a single decoding step

        Args:
            tgt_in (Tensor): tgt_ids for one time step of shape [batch_size,1]
            initial_states (Nested lists of Tensors): States from previous time step to be fed to recurrent networks
            encoder_output (Nested lists of tensor): Output of the encoder
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.. Defaults to False.

        Returns:
            tgt_out: tgt_ids for next time step obtained through greedy search of shape [batch_size, 1]
            final_states: (Nested lists of Tensors) : States to be fed to recurrent networks in next decoding time
            step
            dec_vob_probs: Target token probabilities tensor of shape [batch_size, 1, tgt_vocab_size]
        """
        sequences, src_mask = encoder_output[0], encoder_output[-1]
        tgt_embeds_in, tgt_mask = (
            self.tgt_embedding(tgt_in),
            self.tgt_embedding.compute_mask(tgt_in),
        )
        dec_sequences_1, hid_out_1 = self.decoder_1(
            inputs=tgt_embeds_in,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[0],
        )
        attention_states = self.attention(sequences, dec_sequences_1, src_mask)
        dec_vocab_probs = self.decode2vocab(attention_states)
        tgt_out = argmax(dec_vocab_probs, axis=-1)
        final_state_1 = [hid_out_1]
        final_states = [final_state_1]
        return tgt_out, final_states, dec_vocab_probs


class Decoder_Attention_GRU_2(Model):
    """2 layered GRU based Attention type decoder
    Inherits from Keras model
    """

    def __init__(
        self,
        embed_size,
        hidden_size,
        linear_dropout,
        recurrent_dropout,
        weight_decay,
        tgt_tokenizer,
    ):
        """Initializes the model

        Args:
            embed_size (int): Embedding size
            hidden_size (int): hidden size of recuurent layer
            linear_dropout (float): linear dropout of recurrent layer
            recurrent_dropout (float): recurrent dropout of recurrent layer
            weight_decay (float): weight decay
            tgt_tokenizer (hugging_face tokenizer): target i.e. native script words tokenizer
        """
        super().__init__()

        tgt_vocab_size = tgt_tokenizer.get_vocab_size()
        tgt_pad_id = tgt_tokenizer.token_to_id("<pad>")
        assert tgt_pad_id == 0, "Target pad token id is not 0."

        regularizer = l2(weight_decay)
        # self.tgt_start_id = #self.tgt_tokenizer.token_to_id('<s>')
        # self.tgt_end_id = #self.tgt_tokenizer.token_to_id('</s>')
        self.rnn_type = "GRU"
        self.decoder_type = "Attention"
        self.num_layers = 2
        self.enc_hid2dec_hid_init_1 = Dense(
            hidden_size, kernel_regularizer=regularizer, use_bias=False
        )
        self.enc_hid2dec_hid_init_2 = Dense(
            hidden_size, kernel_regularizer=regularizer, use_bias=False
        )

        self.tgt_embedding = Embedding(tgt_vocab_size, embed_size, mask_zero=True)
        self.decoder_1 = GRU(
            hidden_size,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
            dropout=linear_dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            return_state=True,
        )
        self.decoder_2 = GRU(
            hidden_size,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
            dropout=linear_dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            return_state=True,
        )
        self.attention = Luong_Attention(hidden_size, regularizer)
        self.decode2vocab = Dense(
            tgt_vocab_size, kernel_regularizer=regularizer, activation="softmax"
        )

    def call(self, tgt_ids, encoder_output, training=False):
        """Takes in Encoder output, target ids and produce probabilities of target vocab tokens
        for each sample and for each time step

        Args:
            tgt_ids (Tensor): Target ids tensor of shape [batch_size, max_tgt_len]
            encoder_output (Nested lists of tensors): Output of encoder
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.

        Returns:
            Tensor: Target token probabilities of shape [batch_size, max_tgt_len, tgt_vocab_size]
        """
        # tgt_ids = tgt_ids[:,:-1]
        sequences, enc_final_state, src_mask = encoder_output
        initial_states = self.initialize_states(enc_final_state)
        tgt_embeds, tgt_mask = (
            self.tgt_embedding(tgt_ids),
            self.tgt_embedding.compute_mask(tgt_ids),
        )
        del tgt_ids
        dec_sequences_1, dec_hid_1 = self.decoder_1(
            inputs=tgt_embeds,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[0],
        )
        dec_sequences_2, dec_hid_2 = self.decoder_2(
            inputs=dec_sequences_1,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[1],
        )
        del tgt_embeds, dec_hid_1, dec_hid_2, dec_sequences_1, initial_states
        attention_states = self.attention(sequences, dec_sequences_2, src_mask)
        del sequences, dec_sequences_2, src_mask
        dec_vocab_probs = self.decode2vocab(attention_states)
        return dec_vocab_probs

    def initialize_states(self, encoder_final_state):
        """Initializes states of recurrent networks of model from encoder final states

        Args:
            encoder_final_state (tensor): List of final states of encoder

        Returns:
            Nested list of tensors: Tensors to initialize states of model's recurrent networks
        """
        enc_hid = encoder_final_state[0]
        dec_hid_init_1 = self.enc_hid2dec_hid_init_1(enc_hid)
        dec_hid_init_2 = self.enc_hid2dec_hid_init_2(enc_hid)
        del encoder_final_state, enc_hid
        initial_state_1 = [dec_hid_init_1]
        initial_state_2 = [dec_hid_init_2]
        initial_states = [initial_state_1, initial_state_2]
        return initial_states

    def decode_step(self, tgt_in, initial_states, encoder_output, training=False):
        """Performs a single decoding step

        Args:
            tgt_in (Tensor): tgt_ids for one time step of shape [batch_size,1]
            initial_states (Nested lists of Tensors): States from previous time step to be fed to recurrent networks
            encoder_output (Nested lists of tensor): Output of the encoder
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.. Defaults to False.

        Returns:
            tgt_out: tgt_ids for next time step obtained through greedy search of shape [batch_size, 1]
            final_states: (Nested lists of Tensors) : States to be fed to recurrent networks in next decoding time
            step
            dec_vob_probs: Target token probabilities tensor of shape [batch_size, 1, tgt_vocab_size]
        """
        sequences, src_mask = encoder_output[0], encoder_output[-1]
        tgt_embeds_in, tgt_mask = (
            self.tgt_embedding(tgt_in),
            self.tgt_embedding.compute_mask(tgt_in),
        )
        dec_sequences_1, hid_out_1 = self.decoder_1(
            inputs=tgt_embeds_in,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[0],
        )
        dec_sequences_2, hid_out_2 = self.decoder_2(
            inputs=dec_sequences_1,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[1],
        )
        attention_states = self.attention(sequences, dec_sequences_2, src_mask)
        dec_vocab_probs = self.decode2vocab(attention_states)
        tgt_out = argmax(dec_vocab_probs, axis=-1)
        final_state_1 = [hid_out_1]
        final_state_2 = [hid_out_2]
        final_states = [final_state_1, final_state_2]
        return tgt_out, final_states, dec_vocab_probs


class Decoder_Seq2seq_GRU_2(Model):
    """2 layered GRU based Seq2seq type decoder
    Inherits from Keras model
    """

    def __init__(
        self,
        embed_size,
        hidden_size,
        linear_dropout,
        recurrent_dropout,
        weight_decay,
        tgt_tokenizer,
    ):
        """Initializes the model

        Args:
            embed_size (int): Embedding size
            hidden_size (int): hidden size of recuurent layer
            linear_dropout (float): linear dropout of recurrent layer
            recurrent_dropout (float): recurrent dropout of recurrent layer
            weight_decay (float): weight decay
            tgt_tokenizer (hugging_face tokenizer): target i.e. native script words tokenizer
        """
        super().__init__()

        tgt_vocab_size = tgt_tokenizer.get_vocab_size()
        tgt_pad_id = tgt_tokenizer.token_to_id("<pad>")
        assert tgt_pad_id == 0, "Target pad token id is not 0."

        regularizer = l2(weight_decay)
        # self.tgt_start_id = #self.tgt_tokenizer.token_to_id('<s>')
        # self.tgt_end_id = #self.tgt_tokenizer.token_to_id('</s>')
        self.rnn_type = "GRU"
        self.decoder_type = "Seq2seq"
        self.num_layers = 2
        self.enc_hid2dec_hid_init_1 = Dense(
            hidden_size, kernel_regularizer=regularizer, use_bias=False
        )
        self.enc_hid2dec_hid_init_2 = Dense(
            hidden_size, kernel_regularizer=regularizer, use_bias=False
        )

        self.tgt_embedding = Embedding(tgt_vocab_size, embed_size, mask_zero=True)
        self.decoder_1 = GRU(
            hidden_size,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
            dropout=linear_dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            return_state=True,
        )

        self.decoder_2 = GRU(
            hidden_size,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
            dropout=linear_dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            return_state=True,
        )

        self.decode2vocab = Dense(
            tgt_vocab_size, kernel_regularizer=regularizer, activation="softmax"
        )

    def call(self, tgt_ids, encoder_output, training=False):
        """Takes in Encoder output, target ids and produce probabilities of target vocab tokens
        for each sample and for each time step

        Args:
            tgt_ids (Tensor): Target ids tensor of shape [batch_size, max_tgt_len]
            encoder_output (Nested lists of tensors): Output of encoder
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.

        Returns:
            Tensor: Target token probabilities of shape [batch_size, max_tgt_len, tgt_vocab_size]
        """
        # tgt_ids = tgt_ids[:,:-1]
        sequences, enc_final_state, src_mask = encoder_output
        del sequences, src_mask
        initial_states = self.initialize_states(enc_final_state)
        tgt_embeds, tgt_mask = (
            self.tgt_embedding(tgt_ids),
            self.tgt_embedding.compute_mask(tgt_ids),
        )
        del tgt_ids
        dec_sequences_1, dec_hid_1 = self.decoder_1(
            inputs=tgt_embeds,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[0],
        )
        dec_sequences_2, dec_hid_2 = self.decoder_2(
            inputs=dec_sequences_1,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[1],
        )

        del tgt_embeds, dec_hid_1, dec_hid_2, dec_sequences_1, initial_states
        dec_vocab_probs = self.decode2vocab(dec_sequences_2)
        return dec_vocab_probs

    # encoder_output = [sequences,[enc_hid],src_mask]  GRU

    def initialize_states(self, encoder_final_state):
        """Initializes states of recurrent networks of model from encoder final states

        Args:
            encoder_final_state (tensor): List of final states of encoder

        Returns:
            Nested list of tensors: Tensors to initialize states of model's recurrent networks
        """
        enc_hid = encoder_final_state[0]
        dec_hid_init_1 = self.enc_hid2dec_hid_init_1(enc_hid)
        dec_hid_init_2 = self.enc_hid2dec_hid_init_2(enc_hid)
        del encoder_final_state, enc_hid
        initial_state_1 = [dec_hid_init_1]
        initial_state_2 = [dec_hid_init_2]
        initial_states = [initial_state_1, initial_state_2]
        return initial_states

    def decode_step(self, tgt_in, initial_states, encoder_output, training=False):
        """Performs a single decoding step

        Args:
            tgt_in (Tensor): tgt_ids for one time step of shape [batch_size,1]
            initial_states (Nested lists of Tensors): States from previous time step to be fed to recurrent networks
            encoder_output (Nested lists of tensor): Output of the encoder
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.. Defaults to False.

        Returns:
            tgt_out: tgt_ids for next time step obtained through greedy search of shape [batch_size, 1]
            final_states: (Nested lists of Tensors) : States to be fed to recurrent networks in next decoding time
            step
            dec_vob_probs: Target token probabilities tensor of shape [batch_size, 1, tgt_vocab_size]
        """
        tgt_embeds_in, tgt_mask = (
            self.tgt_embedding(tgt_in),
            self.tgt_embedding.compute_mask(tgt_in),
        )
        dec_sequences_1, hid_out_1 = self.decoder_1(
            inputs=tgt_embeds_in,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[0],
        )
        dec_sequences_2, hid_out_2 = self.decoder_2(
            inputs=dec_sequences_1,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[1],
        )

        dec_vocab_probs = self.decode2vocab(dec_sequences_2)
        tgt_out = argmax(dec_vocab_probs, axis=-1)
        final_state_1 = [hid_out_1]
        final_state_2 = [hid_out_2]
        final_states = [final_state_1, final_state_2]
        return tgt_out, final_states, dec_vocab_probs


class Decoder_Seq2seq_LSTM_1(Model):
    """1 layered LSTM based Seq2seq type decoder
    Inherits from Keras model
    """

    def __init__(
        self,
        embed_size,
        hidden_size,
        linear_dropout,
        recurrent_dropout,
        weight_decay,
        tgt_tokenizer,
    ):
        """Initializes the model

        Args:
            embed_size (int): Embedding size
            hidden_size (int): hidden size of recuurent layer
            linear_dropout (float): linear dropout of recurrent layer
            recurrent_dropout (float): recurrent dropout of recurrent layer
            weight_decay (float): weight decay
            tgt_tokenizer (hugging_face tokenizer): target i.e. native script words tokenizer
        """
        super().__init__()

        tgt_vocab_size = tgt_tokenizer.get_vocab_size()
        tgt_pad_id = tgt_tokenizer.token_to_id("<pad>")
        assert tgt_pad_id == 0, "Target pad token id is not 0."

        regularizer = l2(weight_decay)
        # self.tgt_start_id = #self.tgt_tokenizer.token_to_id('<s>')
        # self.tgt_end_id = #self.tgt_tokenizer.token_to_id('</s>')
        self.rnn_type = "LSTM"
        self.decoder_type = "Seq2seq"
        self.num_layers = 1
        self.enc_hid2dec_hid_init_1 = Dense(
            hidden_size, kernel_regularizer=regularizer, use_bias=False
        )

        self.enc_cell2dec_cell_init_1 = Dense(
            hidden_size, kernel_regularizer=regularizer, use_bias=False
        )

        self.tgt_embedding = Embedding(tgt_vocab_size, embed_size, mask_zero=True)
        self.decoder_1 = LSTM(
            hidden_size,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
            dropout=linear_dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            return_state=True,
        )

        self.decode2vocab = Dense(
            tgt_vocab_size, kernel_regularizer=regularizer, activation="softmax"
        )

    def call(self, tgt_ids, encoder_output, training=False):
        """Takes in Encoder output, target ids and produce probabilities of target vocab tokens
        for each sample and for each time step

        Args:
            tgt_ids (Tensor): Target ids tensor of shape [batch_size, max_tgt_len]
            encoder_output (Nested lists of tensors): Output of encoder
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.

        Returns:
            Tensor: Target token probabilities of shape [batch_size, max_tgt_len, tgt_vocab_size]
        """

        sequences, enc_final_state, src_mask = encoder_output
        del sequences, src_mask
        initial_states = self.initialize_states(enc_final_state)

        tgt_embeds, tgt_mask = (
            self.tgt_embedding(tgt_ids),
            self.tgt_embedding.compute_mask(tgt_ids),
        )
        del tgt_ids
        dec_sequences_1, dec_hid_1, dec_cell_1 = self.decoder_1(
            inputs=tgt_embeds,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[0],
        )
        del tgt_embeds, dec_hid_1, dec_cell_1, initial_states
        dec_vocab_probs = self.decode2vocab(dec_sequences_1)
        return dec_vocab_probs

    def initialize_states(self, encoder_final_state):
        """Initializes states of recurrent networks of model from encoder final states

        Args:
            encoder_final_state (tensor): List of final states of encoder

        Returns:
            Nested list of tensors: Tensors to initialize states of model's recurrent networks
        """
        enc_hid, enc_cell = encoder_final_state[0], encoder_final_state[1]
        dec_hid_init_1 = self.enc_hid2dec_hid_init_1(enc_hid)
        dec_cell_init_1 = self.enc_cell2dec_cell_init_1(enc_cell)
        del encoder_final_state, enc_hid, enc_cell
        initial_state_1 = [dec_hid_init_1, dec_cell_init_1]
        initial_states = [initial_state_1]
        return initial_states

    def decode_step(self, tgt_in, initial_states, encoder_output, training=False):
        """Performs a single decoding step

        Args:
            tgt_in (Tensor): tgt_ids for one time step of shape [batch_size,1]
            initial_states (Nested lists of Tensors): States from previous time step to be fed to recurrent networks
            encoder_output (Nested lists of tensor): Output of the encoder
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.. Defaults to False.

        Returns:
            tgt_out: tgt_ids for next time step obtained through greedy search of shape [batch_size, 1]
            final_states: (Nested lists of Tensors) : States to be fed to recurrent networks in next decoding time
            step
            dec_vob_probs: Target token probabilities tensor of shape [batch_size, 1, tgt_vocab_size]
        """
        tgt_embeds_in, tgt_mask = (
            self.tgt_embedding(tgt_in),
            self.tgt_embedding.compute_mask(tgt_in),
        )
        dec_sequences_1, hid_out_1, cell_out_1 = self.decoder_1(
            inputs=tgt_embeds_in,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[0],
        )
        dec_vocab_probs = self.decode2vocab(dec_sequences_1)
        tgt_out = argmax(dec_vocab_probs, axis=-1)
        final_state_1 = [hid_out_1, cell_out_1]
        final_states = [final_state_1]
        return tgt_out, final_states, dec_vocab_probs


class Decoder_Seq2seq_LSTM_2(Model):
    """2 layered LSTM based Seq2seq type decoder
    Inherits from Keras model
    """

    def __init__(
        self,
        embed_size,
        hidden_size,
        linear_dropout,
        recurrent_dropout,
        weight_decay,
        tgt_tokenizer,
    ):
        """Initializes the model

        Args:
            embed_size (int): Embedding size
            hidden_size (int): hidden size of recuurent layer
            linear_dropout (float): linear dropout of recurrent layer
            recurrent_dropout (float): recurrent dropout of recurrent layer
            weight_decay (float): weight decay
            tgt_tokenizer (hugging_face tokenizer): target i.e. native script words tokenizer
        """
        super().__init__()

        tgt_vocab_size = tgt_tokenizer.get_vocab_size()
        tgt_pad_id = tgt_tokenizer.token_to_id("<pad>")
        assert tgt_pad_id == 0, "Target pad token id is not 0."

        regularizer = l2(weight_decay)
        # self.tgt_start_id = #self.tgt_tokenizer.token_to_id('<s>')
        # self.tgt_end_id = #self.tgt_tokenizer.token_to_id('</s>')
        self.rnn_type = "LSTM"
        self.decoder_type = "Seq2seq"
        self.num_layers = 2

        self.enc_hid2dec_hid_init_1 = Dense(
            hidden_size, kernel_regularizer=regularizer, use_bias=False
        )

        self.enc_cell2dec_cell_init_1 = Dense(
            hidden_size, kernel_regularizer=regularizer, use_bias=False
        )

        self.enc_hid2dec_hid_init_2 = Dense(
            hidden_size, kernel_regularizer=regularizer, use_bias=False
        )

        self.enc_cell2dec_cell_init_2 = Dense(
            hidden_size, kernel_regularizer=regularizer, use_bias=False
        )

        self.tgt_embedding = Embedding(tgt_vocab_size, embed_size, mask_zero=True)
        self.decoder_1 = LSTM(
            hidden_size,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
            dropout=linear_dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            return_state=True,
        )

        self.decoder_2 = LSTM(
            hidden_size,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
            dropout=linear_dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            return_state=True,
        )

        self.decode2vocab = Dense(
            tgt_vocab_size, kernel_regularizer=regularizer, activation="softmax"
        )

    def call(self, tgt_ids, encoder_output, training=False):
        """Takes in Encoder output, target ids and produce probabilities of target vocab tokens
        for each sample and for each time step

        Args:
            tgt_ids (Tensor): Target ids tensor of shape [batch_size, max_tgt_len]
            encoder_output (Nested lists of tensors): Output of encoder
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.

        Returns:
            Tensor: Target token probabilities of shape [batch_size, max_tgt_len, tgt_vocab_size]
        """

        sequences, enc_final_state, src_mask = encoder_output
        del sequences, src_mask
        initial_states = self.initialize_states(enc_final_state)

        tgt_embeds, tgt_mask = (
            self.tgt_embedding(tgt_ids),
            self.tgt_embedding.compute_mask(tgt_ids),
        )
        del tgt_ids
        dec_sequences_1, dec_hid_1, dec_cell_1 = self.decoder_1(
            inputs=tgt_embeds,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[0],
        )
        dec_sequences_2, dec_hid_2, dec_cell_2 = self.decoder_2(
            inputs=dec_sequences_1,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[1],
        )
        del (
            tgt_embeds,
            dec_hid_1,
            dec_cell_1,
            dec_sequences_1,
            dec_hid_2,
            dec_cell_2,
            initial_states,
        )
        dec_vocab_probs = self.decode2vocab(dec_sequences_2)
        return dec_vocab_probs

    def initialize_states(self, encoder_final_state):
        """Initializes states of recurrent networks of model from encoder final states

        Args:
            encoder_final_state (tensor): List of final states of encoder

        Returns:
            Nested list of tensors: Tensors to initialize states of model's recurrent networks
        """
        enc_hid, enc_cell = encoder_final_state[0], encoder_final_state[1]
        dec_hid_init_1 = self.enc_hid2dec_hid_init_1(enc_hid)
        dec_cell_init_1 = self.enc_cell2dec_cell_init_1(enc_cell)
        dec_hid_init_2 = self.enc_hid2dec_hid_init_2(enc_hid)
        dec_cell_init_2 = self.enc_cell2dec_cell_init_2(enc_cell)
        del encoder_final_state, enc_hid, enc_cell
        initial_state_1 = [dec_hid_init_1, dec_cell_init_1]
        initial_state_2 = [dec_hid_init_2, dec_cell_init_2]
        initial_states = [initial_state_1, initial_state_2]
        return initial_states

    def decode_step(self, tgt_in, initial_states, encoder_output, training=False):
        """Performs a single decoding step

        Args:
            tgt_in (Tensor): tgt_ids for one time step of shape [batch_size,1]
            initial_states (Nested lists of Tensors): States from previous time step to be fed to recurrent networks
            encoder_output (Nested lists of tensor): Output of the encoder
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.. Defaults to False.

        Returns:
            tgt_out: tgt_ids for next time step obtained through greedy search of shape [batch_size, 1]
            final_states: (Nested lists of Tensors) : States to be fed to recurrent networks in next decoding time
            step
            dec_vob_probs: Target token probabilities tensor of shape [batch_size, 1, tgt_vocab_size]
        """
        tgt_embeds_in, tgt_mask = (
            self.tgt_embedding(tgt_in),
            self.tgt_embedding.compute_mask(tgt_in),
        )
        dec_sequences_1, hid_out_1, cell_out_1 = self.decoder_1(
            inputs=tgt_embeds_in,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[0],
        )
        dec_sequences_2, hid_out_2, cell_out_2 = self.decoder_2(
            inputs=dec_sequences_1,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[1],
        )
        dec_vocab_probs = self.decode2vocab(dec_sequences_2)
        tgt_out = argmax(dec_vocab_probs, axis=-1)
        final_state_1 = [hid_out_1, cell_out_1]
        final_state_2 = [hid_out_2, cell_out_2]
        final_states = [final_state_1, final_state_2]
        return tgt_out, final_states, dec_vocab_probs


class Decoder_Attention_LSTM_1(Model):
    """1 layered LSTM based Attention type decoder
    Inherits from Keras model
    """

    def __init__(
        self,
        embed_size,
        hidden_size,
        linear_dropout,
        recurrent_dropout,
        weight_decay,
        tgt_tokenizer,
    ):
        """Initializes the model

        Args:
            embed_size (int): Embedding size
            hidden_size (int): hidden size of recuurent layer
            linear_dropout (float): linear dropout of recurrent layer
            recurrent_dropout (float): recurrent dropout of recurrent layer
            weight_decay (float): weight decay
            tgt_tokenizer (hugging_face tokenizer): target i.e. native script words tokenizer
        """
        super().__init__()

        tgt_vocab_size = tgt_tokenizer.get_vocab_size()
        tgt_pad_id = tgt_tokenizer.token_to_id("<pad>")
        assert tgt_pad_id == 0, "Target pad token id is not 0."

        regularizer = l2(weight_decay)
        self.rnn_type = "LSTM"
        self.decoder_type = "Attention"
        self.num_layers = 2

        # self.tgt_start_id = #self.tgt_tokenizer.token_to_id('<s>')
        # self.tgt_end_id = #self.tgt_tokenizer.token_to_id('</s>')
        self.enc_hid2dec_hid_init_1 = Dense(
            hidden_size, kernel_regularizer=regularizer, use_bias=False
        )

        self.enc_cell2dec_cell_init_1 = Dense(
            hidden_size, kernel_regularizer=regularizer, use_bias=False
        )

        self.tgt_embedding = Embedding(tgt_vocab_size, embed_size, mask_zero=True)
        self.decoder_1 = LSTM(
            hidden_size,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
            dropout=linear_dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            return_state=True,
        )
        self.attention = Luong_Attention(hidden_size, regularizer)
        self.decode2vocab = Dense(
            tgt_vocab_size, kernel_regularizer=regularizer, activation="softmax"
        )

    def call(self, tgt_ids, encoder_output, training=False):
        """Takes in Encoder output, target ids and produce probabilities of target vocab tokens
        for each sample and for each time step

        Args:
            tgt_ids (Tensor): Target ids tensor of shape [batch_size, max_tgt_len]
            encoder_output (Nested lists of tensors): Output of encoder
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.

        Returns:
            Tensor: Target token probabilities of shape [batch_size, max_tgt_len, tgt_vocab_size]
        """

        sequences, enc_final_state, src_mask = encoder_output
        initial_states = self.initialize_states(enc_final_state)
        del enc_final_state
        tgt_embeds, tgt_mask = (
            self.tgt_embedding(tgt_ids),
            self.tgt_embedding.compute_mask(tgt_ids),
        )
        del tgt_ids
        dec_sequences_1, dec_hid_1, dec_cell_1 = self.decoder_1(
            inputs=tgt_embeds,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[0],
        )
        del tgt_embeds, dec_hid_1, dec_cell_1, initial_states
        attention_states = self.attention(sequences, dec_sequences_1, src_mask)
        del sequences, dec_sequences_1, src_mask
        dec_vocab_probs = self.decode2vocab(attention_states)
        return dec_vocab_probs

    def initialize_states(self, encoder_final_state):
        """Initializes states of recurrent networks of model from encoder final states

        Args:
            encoder_final_state (tensor): List of final states of encoder

        Returns:
            Nested list of tensors: Tensors to initialize states of model's recurrent networks
        """
        enc_hid, enc_cell = encoder_final_state[0], encoder_final_state[1]
        dec_hid_init_1 = self.enc_hid2dec_hid_init_1(enc_hid)
        dec_cell_init_1 = self.enc_cell2dec_cell_init_1(enc_cell)
        del encoder_final_state, enc_hid, enc_cell
        initial_state_1 = [dec_hid_init_1, dec_cell_init_1]
        initial_states = [initial_state_1]
        return initial_states

    def decode_step(self, tgt_in, initial_states, encoder_output, training=False):
        """Performs a single decoding step

        Args:
            tgt_in (Tensor): tgt_ids for one time step of shape [batch_size,1]
            initial_states (Nested lists of Tensors): States from previous time step to be fed to recurrent networks
            encoder_output (Nested lists of tensor): Output of the encoder
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.. Defaults to False.

        Returns:
            tgt_out: tgt_ids for next time step obtained through greedy search of shape [batch_size, 1]
            final_states: (Nested lists of Tensors) : States to be fed to recurrent networks in next decoding time
            step
            dec_vob_probs: Target token probabilities tensor of shape [batch_size, 1, tgt_vocab_size]
        """
        sequences, src_mask = encoder_output[0], encoder_output[-1]
        tgt_embeds_in, tgt_mask = (
            self.tgt_embedding(tgt_in),
            self.tgt_embedding.compute_mask(tgt_in),
        )
        dec_sequences_1, hid_out_1, cell_out_1 = self.decoder_1(
            inputs=tgt_embeds_in,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[0],
        )

        attention_states = self.attention(sequences, dec_sequences_1, src_mask)
        dec_vocab_probs = self.decode2vocab(attention_states)
        tgt_out = argmax(dec_vocab_probs, axis=-1)
        final_state_1 = [hid_out_1, cell_out_1]
        final_states = [final_state_1]
        return tgt_out, final_states, dec_vocab_probs


class Decoder_Attention_LSTM_2(Model):
    """2 layered LSTM based Attention type decoder
    Inherits from Keras model
    """

    def __init__(
        self,
        embed_size,
        hidden_size,
        linear_dropout,
        recurrent_dropout,
        weight_decay,
        tgt_tokenizer,
    ):
        """Initializes the model

        Args:
            embed_size (int): Embedding size
            hidden_size (int): hidden size of recuurent layer
            linear_dropout (float): linear dropout of recurrent layer
            recurrent_dropout (float): recurrent dropout of recurrent layer
            weight_decay (float): weight decay
            tgt_tokenizer (hugging_face tokenizer): target i.e. native script words tokenizer
        """
        super().__init__()

        tgt_vocab_size = tgt_tokenizer.get_vocab_size()
        tgt_pad_id = tgt_tokenizer.token_to_id("<pad>")
        assert tgt_pad_id == 0, "Target pad token id is not 0."

        regularizer = l2(weight_decay)
        self.rnn_type = "LSTM"
        self.decoder_type = "Attention"
        self.num_layers = 2

        # self.tgt_start_id = #self.tgt_tokenizer.token_to_id('<s>')
        # self.tgt_end_id = #self.tgt_tokenizer.token_to_id('</s>')
        self.enc_hid2dec_hid_init_1 = Dense(
            hidden_size, kernel_regularizer=regularizer, use_bias=False
        )

        self.enc_cell2dec_cell_init_1 = Dense(
            hidden_size, kernel_regularizer=regularizer, use_bias=False
        )

        self.enc_hid2dec_hid_init_2 = Dense(
            hidden_size, kernel_regularizer=regularizer, use_bias=False
        )

        self.enc_cell2dec_cell_init_2 = Dense(
            hidden_size, kernel_regularizer=regularizer, use_bias=False
        )

        self.tgt_embedding = Embedding(tgt_vocab_size, embed_size, mask_zero=True)
        self.decoder_1 = LSTM(
            hidden_size,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
            dropout=linear_dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            return_state=True,
        )
        self.decoder_2 = LSTM(
            hidden_size,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
            dropout=linear_dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            return_state=True,
        )
        self.attention = Luong_Attention(hidden_size, regularizer)
        self.decode2vocab = Dense(
            tgt_vocab_size, kernel_regularizer=regularizer, activation="softmax"
        )

    def call(self, tgt_ids, encoder_output, training=False):
        """Takes in Encoder output, target ids and produce probabilities of target vocab tokens
        for each sample and for each time step

        Args:
            tgt_ids (Tensor): Target ids tensor of shape [batch_size, max_tgt_len]
            encoder_output (Nested lists of tensors): Output of encoder
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.

        Returns:
            Tensor: Target token probabilities of shape [batch_size, max_tgt_len, tgt_vocab_size]
        """

        sequences, enc_final_state, src_mask = encoder_output
        initial_states = self.initialize_states(enc_final_state)
        del enc_final_state
        tgt_embeds, tgt_mask = (
            self.tgt_embedding(tgt_ids),
            self.tgt_embedding.compute_mask(tgt_ids),
        )
        del tgt_ids
        dec_sequences_1, dec_hid_1, dec_cell_1 = self.decoder_1(
            inputs=tgt_embeds,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[0],
        )
        del tgt_embeds, dec_hid_1, dec_cell_1
        dec_sequences_2, dec_hid_2, dec_cell_2 = self.decoder_2(
            inputs=dec_sequences_1,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[1],
        )
        del dec_sequences_1, dec_hid_2, dec_cell_2, initial_states
        attention_states = self.attention(sequences, dec_sequences_2, src_mask)
        del sequences, dec_sequences_2, src_mask
        dec_vocab_probs = self.decode2vocab(attention_states)
        return dec_vocab_probs

    def initialize_states(self, encoder_final_state):
        """Initializes states of recurrent networks of model from encoder final states

        Args:
            encoder_final_state (tensor): List of final states of encoder

        Returns:
            Nested list of tensors: Tensors to initialize states of model's recurrent networks
        """
        enc_hid, enc_cell = encoder_final_state[0], encoder_final_state[1]
        dec_hid_init_1 = self.enc_hid2dec_hid_init_1(enc_hid)
        dec_cell_init_1 = self.enc_cell2dec_cell_init_1(enc_cell)
        dec_hid_init_2 = self.enc_hid2dec_hid_init_2(enc_hid)
        dec_cell_init_2 = self.enc_cell2dec_cell_init_2(enc_cell)
        del encoder_final_state, enc_hid, enc_cell
        initial_state_1 = [dec_hid_init_1, dec_cell_init_1]
        initial_state_2 = [dec_hid_init_2, dec_cell_init_2]
        initial_states = [initial_state_1, initial_state_2]
        return initial_states

    def decode_step(self, tgt_in, initial_states, encoder_output, training=False):
        """Performs a single decoding step

        Args:
            tgt_in (Tensor): tgt_ids for one time step of shape [batch_size,1]
            initial_states (Nested lists of Tensors): States from previous time step to be fed to recurrent networks
            encoder_output (Nested lists of tensor): Output of the encoder
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.. Defaults to False.

        Returns:
            tgt_out: tgt_ids for next time step obtained through greedy search of shape [batch_size, 1]
            final_states: (Nested lists of Tensors) : States to be fed to recurrent networks in next decoding time
            step
            dec_vob_probs: Target token probabilities tensor of shape [batch_size, 1, tgt_vocab_size]
        """
        sequences, src_mask = encoder_output[0], encoder_output[-1]
        tgt_embeds_in, tgt_mask = (
            self.tgt_embedding(tgt_in),
            self.tgt_embedding.compute_mask(tgt_in),
        )
        dec_sequences_1, hid_out_1, cell_out_1 = self.decoder_1(
            inputs=tgt_embeds_in,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[0],
        )
        dec_sequences_2, hid_out_2, cell_out_2 = self.decoder_2(
            inputs=dec_sequences_1,
            mask=tgt_mask,
            training=training,
            initial_state=initial_states[1],
        )
        attention_states = self.attention(sequences, dec_sequences_2, src_mask)
        dec_vocab_probs = self.decode2vocab(attention_states)
        tgt_out = argmax(dec_vocab_probs, axis=-1)
        final_state_1 = [hid_out_1, cell_out_1]
        final_state_2 = [hid_out_2, cell_out_2]
        final_states = [final_state_1, final_state_2]
        return tgt_out, final_states, dec_vocab_probs
