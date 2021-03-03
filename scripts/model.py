import tensorflow as tf
from tensorflow.keras import Model

from . import transliteration_tokenizers
from .decoders import (
    Decoder_Attention_GRU_1,
    Decoder_Attention_GRU_2,
    Decoder_Attention_LSTM_1,
    Decoder_Attention_LSTM_2,
    Decoder_Seq2seq_GRU_1,
    Decoder_Seq2seq_GRU_2,
    Decoder_Seq2seq_LSTM_1,
    Decoder_Seq2seq_LSTM_2,
)
from .encoders import Encoder_GRU_1, Encoder_GRU_2, Encoder_LSTM_1, Encoder_LSTM_2

encoder_dict = {
    "Encoder_GRU_1": Encoder_GRU_1,
    "Encoder_LSTM_1": Encoder_LSTM_1,
    "Encoder_GRU_2": Encoder_GRU_2,
    "Encoder_LSTM_2": Encoder_LSTM_2,
}

decoder_dict = {
    "Decoder_Seq2seq_GRU_1": Decoder_Seq2seq_GRU_1,
    "Decoder_Seq2seq_LSTM_1": Decoder_Seq2seq_LSTM_1,
    "Decoder_Attention_GRU_1": Decoder_Attention_GRU_1,
    "Decoder_Attention_LSTM_1": Decoder_Attention_LSTM_1,
    "Decoder_Seq2seq_GRU_2": Decoder_Seq2seq_GRU_2,
    "Decoder_Seq2seq_LSTM_2": Decoder_Seq2seq_LSTM_2,
    "Decoder_Attention_GRU_2": Decoder_Attention_GRU_2,
    "Decoder_Attention_LSTM_2": Decoder_Attention_LSTM_2,
}


def model_creator(params_dict):
    """Creates Encoder-Decoder model based on values in params dict

    Args:
        params_dict (dict): params needed to initialize the encoder and decoder
            : model_type - Type of decoder - Seq2seq or Attention
            : rnn_type - Type of recurrent networks used in Encoder and Decoder
            : encoder_layers - No. of recurrent network layers in Encoder
            : decoder_layers - No. of recurrent network layers in Decoder
            : embed_size - Embedding size used in encoder and decoder
            : hidden_size - Hidden size of recurrent networks in encoder and decoder
            : linear_dropout - Linear dropout used in recurrent networks in encoder and decoder
            : recurrent_dropout - Recurrent dropout used in recurrent networks in encoder and decoder
            : weight_decay - L2 weight decay value
            : src_tokenizer - source i.e romanized words tokenizer
            : tgt_tokenizer - target i.e. native script words tokenizer

    Returns:
        Transliteraion Model: Model initialized with given params
    """
    model_type = params_dict.pop("model_type")
    rnn_type = params_dict.pop("rnn_type")
    encoder_layers = params_dict.pop("encoder_layers")
    decoder_layers = params_dict.pop("decoder_layers")

    encoder_type = f"Encoder_{rnn_type}_{encoder_layers}"
    decoder_type = f"Decoder_{model_type}_{rnn_type}_{decoder_layers}"

    Encoder = encoder_dict[encoder_type]
    Decoder = decoder_dict[decoder_type]

    return Transliteration_Model(Encoder, Decoder, **params_dict)


class Transliteration_Model(Model):
    """Model which transliterates romanized words to native script words.
    Inherited from Keras model

    """

    def __init__(
        self,
        Encoder,
        Decoder,
        embed_size,
        hidden_size,
        linear_dropout,
        recurrent_dropout,
        weight_decay,
        src_tokenizer,
        tgt_tokenizer,
    ):
        """Initializes the model

        Args:
            Encoder (Keras Model): Encoder network
            Decoder (Keras Model): Decoder network
            hidden_size (int): Hidden size of network
            linear_dropout(float) - Linear dropout used in recurrent networks in encoder and decoder
            recurrent_dropout(float) - Recurrent dropout used in recurrent networks in encoder and decoder
            weight_decay(float) - L2 weight decay value
            src_tokenizer(hugging_face tokenizer) - source i.e romanized words tokenizer
            tgt_tokenizer(hugging_face tokenizer) - target i.e. native script words tokenizer
        """

        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.linear_dropout = linear_dropout
        self.recurrent_dropout = recurrent_dropout
        self.weight_decay = weight_decay

        self.src_vocab_size = self.src_tokenizer.get_vocab_size()
        self.tgt_vocab_size = self.tgt_tokenizer.get_vocab_size()
        self.src_pad_id = self.src_tokenizer.token_to_id("<pad>")
        self.tgt_pad_id = self.tgt_tokenizer.token_to_id("<pad>")
        self.tgt_start_id = self.tgt_tokenizer.token_to_id("<s>")
        self.tgt_end_id = self.tgt_tokenizer.token_to_id("</s>")

        self.encoder = Encoder(
            embed_size,
            hidden_size,
            linear_dropout,
            recurrent_dropout,
            weight_decay,
            src_tokenizer,
        )
        self.decoder = Decoder(
            embed_size,
            hidden_size,
            linear_dropout,
            recurrent_dropout,
            weight_decay,
            tgt_tokenizer,
        )

        self.rnn_type = self.encoder.rnn_type
        self.encoder_layers = self.encoder.num_layers
        self.decoder_type = self.decoder.decoder_type
        self.decoder_layers = self.decoder.num_layers

    def call(self, inputs, training=False):
        """Takes in source ids and return probabilities of target id tokens

        Args:
            inputs (tuple of tensors): Inputs to the model
                : src_ids: Tensor of shape [batch_size, max_src_len]
                : tgt_ids: Tensor of shape [batch_size, max_tgt_len]
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.

        Returns:
            Tensor: probabilities of target id tokens whose shape is [batch,max_tgt_len,tgt_voacab_size]
        """
        src_ids, tgt_ids = inputs
        encoder_output = self.encode(src_ids, training)
        del src_ids
        decode_vocab_scores = self.decode(tgt_ids, encoder_output, training)
        return decode_vocab_scores

    def encode(self, src_ids, training=False):
        """Encode the source tokens

        Args:
            src_ids (Tensor): Source ids of shape [batch_size, max_src_len]
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.

        Returns:
            Nested list of Tensors - Refer encoder.call method  for exact details. Varies from one to other
        """
        return self.encoder(src_ids, training)

    def decode(self, tgt_ids, encoder_output, training=False):
        """Perform the decoding operation

        Args:
            tgt_ids (Tensor): Target ids of shape [batch_size, max_tgt_len]
            encoder_output (Nested list of tensors): Encoder output
            training (bool, optional):  Specifies whether to operate in training mode or
            inference mode. Defaults to False.

        Returns:
            Tensor: probabilities of target id tokens whose shape is [batch,max_tgt_len,tgt_voacab_size]
        """
        return self.decoder(tgt_ids, encoder_output, training)

    def decode_step(self, tgt_in, initial_state, encoder_output, training=False):
        """Performs a single decoding step

        Args:
            tgt_in (Tensor): tgt_ids for one time step of shape [batch_size,1]
            initial_states (Nested lists of Tensors): States from previous time step to be fed to recurrent networks
            encoder_output (Nested lists of tensor): Output of the encoder
            training (bool, optional): Specifies whether to operate in training mode or
            inference mode. Defaults to False.
        Returns:
            tgt_out: tgt_ids for next time step obtained through greedy search of shape [batch_size, 1]
            final_states: (Nested lists of Tensors) : States to be fed to recurrent networks in next decoding time
            step
            dec_vob_probs: Target token probabilities tensor of shape [batch_size, 1, tgt_vocab_size]
        """
        return self.decoder.decode_step(tgt_in, initial_state, encoder_output, training)

    def ids_to_word(self, ids, ids_type):
        """Given a sequence of token ids, connvert it to a word

        Args:
            ids (List of ints): Sequence of token ids
            ids_type (string): "src" or "tgt" depending on whether ids are of source or target

        Returns:
            string: Word obtained on converting ids back to tokens using tokenizer
        """
        if ids_type == "src":
            tokenizer = self.src_tokenizer
        elif ids_type == "tgt":
            tokenizer = self.tgt_tokenizer
        out = []
        for num in ids:
            if num == tokenizer.token_to_id("</s>"):
                break
            if num not in [
                tokenizer.token_to_id("<pad>"),
                tokenizer.token_to_id("<s>"),
            ]:
                out.append(num)
        return "".join([tokenizer.id_to_token(num) for num in out])
