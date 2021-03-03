import re
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.math import argmax, log

from .model import Transliteration_Model


class BeamSearch:
    """Class implementing Beam Search"""

    def __init__(self, model, beam_size, max_options=5):
        """Initializes the beam searcher

        Args:
            model (Transliteration Model): Model on which beam search is used
            beam_size (int): Beam size in beam serach
            max_options (int, optional): If the completed options crosses max_options, then serach terminated. Defaults to 5.
        """
        num_states_dict = {"GRU": 1, "LSTM": 2}
        self.model = model
        self.beam_size = beam_size
        self.start_id = self.model.tgt_start_id
        self.end_id = self.model.tgt_end_id
        self.max_options = max_options
        self.rnn_type = self.model.rnn_type
        self.decoder_type = self.model.decoder_type
        self.encoder_layers = self.model.encoder_layers
        self.decoder_layers = self.model.decoder_layers
        self.num_states = num_states_dict[self.rnn_type]
        self.max_options = max_options
        self.option = {
            "ids": [],
            "avg_score": 0,
            "scores": [],
            "states": [[]] * self.decoder_layers,
        }
        # Option is a data structure which stores the entire information regarding a candidate i.e.
        # ids - List of sequence of ids generated till now
        # avg_score - obtained by calculating score function on scores
        # scores - list of log loss probability scores for each id in ids
        # states - Nested list of hidden states(format below) to be fed to the decoder recurrent network if this option is explored further
        # states_format = [[hid_1, cell_1], [hid_2, cell_2]]
        self.completed_options = []

    def process(self, src_ids):
        """Core function which performs the beam search. Intakes a single example, performs beam search and returns best possible words
        sorted in the decreasing order of their likeliness

        Args:
            src_ids (Tensor): Token ids of romanized word in tensor format of shape [1, src_len]

        Returns:
            List of strings: Best possible native script words for given romanized word
        """
        self.reset()  # Resets self.completed_options to empty list

        max_len = src_ids.shape[1] + 10  # Max number of steps search is continued

        encoder_output_original = self.model.encode(
            src_ids
        )  # Get Encounder output - Remember here the batch size is 1

        (
            sequences,
            encoder_final,
            src_mask,
        ) = encoder_output_original  # Unpack encoder output

        initial_states = self.model.decoder.initialize_states(
            encoder_final
        )  # Initialize hidden states of decoder

        tgt_in = tf.reshape(
            [self.start_id], (-1, 1)
        )  # Initial target ids are just start tokens

        _, final_states, vocab_probs = self.model.decode_step(
            tgt_in, initial_states, encoder_output_original
        )  # Perform one single step of search
        # This step is outside of loop because here for decoder step batch size is same as encoder batch size i.e 1
        # Also we have to separately initialize hidden and cell states. But in a loop we get them from previous steps

        old_options = [
            self.option
        ]  # Since we are starting from 0, the empty option is itself old option

        good_options = self.options_from_outputs(
            vocab_probs, final_states, old_options
        )  # good options by selecting beam search num of options
        # for each entry.

        best_options = self.extract_best_options(
            good_options
        )  # Filtering and getting the best options from good options

        i = 0

        # Selecting best options in every time step and continuing the loop.
        # 2nd condition because if all best options have last id as stop id, then all options shifted to completed list and it becomes null list
        while (
            i < max_len
            and len(best_options) > 0
            and len(self.completed_options) < self.max_options
        ):
            best_options = self.step(best_options, encoder_output_original)

        # Sorting the completed options i.e options which had the end token
        self.completed_options = sorted(
            self.completed_options, key=lambda option: option["avg_score"], reverse=True
        )

        # Extracting ids from these options
        completed_samples = [option["ids"] for option in self.completed_options]

        # Converting ids to words
        words = [self.model.ids_to_word(ids, "tgt") for ids in completed_samples]

        # Filtering null strings
        full_words = [word for word in words if word != ""]

        return full_words
        ## Process ==> (tgt_inps, states, encoder_output)->(all_scores, fin_states) -> good_options -> best_options -> (inps,states,enc_out)

        ## Step ==> best_options -> (tgt_inps, states, encoder_output)->(all_scores, fin_states) -> good_options -> best_options

        ## Output -> best_scores_ids -> good_options (batch * beam) -> overall_best_options -> inputs --> outputs

    def step(self, best_options, encoder_output_original):
        """Performs single step of beam search. Intakes best options at a given step and outputs best options for next time step

        Args:
            best_options (List of options): Best options at a given time step
            encoder_output_original (Nested list of tensors): Output of encoder - Used only in Attention decoders

        Returns:
            Nested list of Tensors: Best options to be fed to step method at next time step
        """

        # Create tensors from list of options which can be fed to decode_step method of model
        tgt_ids, initial_states, new_encoder_output = self.inputs_from_options(
            best_options, encoder_output_original
        )

        # Perform a decode_step and get final vocab probabilities and states of recurrent networks(fed to the same in next step)
        _, final_states, vocab_probs = self.model.decode_step(
            tgt_ids, initial_states, new_encoder_output
        )

        # Create good options from outputs. For each sample in batch, select beam search number of good options
        # Example : if batch_size = 3, beam_size = 5. Then total batch_size*beam_size =  15 good options
        good_options = self.options_from_outputs(
            vocab_probs, final_states, best_options
        )

        # Select beam_size number of best options from these good options based on their scores. So, select best 5 out of 15
        new_best_options = self.extract_best_options(good_options)

        return new_best_options

    def options_from_outputs(self, vocab_probs, final_states, old_options):
        """Creates beam_size number of options for each sample in batch

        Args:
            vocab_probs (Tensor): Probabilities of token_ids at given time step.Shape [batch_size, 1, tgt_vocab_size]
            final_states (Nested list of tensors): final_states after the decode_step. Used in next time step
            old_options (List of options): Old options used to obtain the above vocab_probs and final_states

        Returns:
            List of options: batch_size*beam_size number of good options
        """
        # For each sample in batch, extract beam size number of good ids and their corresponding log probability scores
        ids_scores = self.extract_good_ids_scores(vocab_probs)

        # Keep in mind for all good ids of a particular sample in a batch, the states will remain same. It is only scores which differ

        # In next 2 steps, the packed hidden states for an entire batch are separated into hidden states for each sample
        # Lets taken an example of 2 layered LSTM decoder

        # Input before 1st step [ [hid_1,cell_1], [hid_2,cell_2]]
        # Now len(states) = 2(bcoz 2 layers), len(states[0]) = 2(bcoz hid and cell states), len(states[0][0]) = 4(batch_size is 1st dim of Tensor)
        #
        #   |   |   |hid_1_sample_1|    |cell_1_sample_1|   |   |   |hid_2_sample_1|    |cell_2_sample_1|   |   |
        #   |   |   |hid_1_sample_2|    |cell_1_sample_2|   |   |   |hid_2_sample_2|    |cell_2_sample_2|   |   |
        #   |   |   |hid_1_sample_3|    |cell_1_sample_3|   |   |   |hid_2_sample_3|    |cell_2_sample_3|   |   |
        #   |   |   |hid_1_sample_4|    |cell_1_sample_4|   |   |   |hid_2_sample_4|    |cell_2_sample_4|   |   |

        # Step 1
        unstacked_states = [
            list(zip(*[tf.unstack(state) for state in layers]))
            for layers in final_states
        ]

        # Output after 1st step - Here it separates states and  packs them by sample within a layer
        # Now len(states) = 2(bcoz 2 layers), len(states[0]) = 4(bcoz batch_size), len(states[0][0]) = 2(bcoz cell and hid states)
        #
        #   |   |   (hid_1_sample_1, cell_1_sample_1)   |   |   (hid_2_sample_1, cell_2_sample_1)   |   |
        #   |   |   (hid_1_sample_2, cell_1_sample_2)   |   |   (hid_2_sample_2, cell_2_sample_2)   |   |
        #   |   |   (hid_1_sample_3, cell_1_sample_3)   |   |   (hid_2_sample_3, cell_2_sample_3)   |   |
        #   |   |   (hid_1_sample_4, cell_1_sample_4)   |   |   (hid_2_sample_4, cell_2_sample_4)   |   |

        # Step 2
        final_states = list(zip(*unstacked_states))

        # Output after 2nd step. Here it separates states and packs them by sample across layers
        # Now len(states) = 4(bcoz batch_size), len(states[0]) = 2(bcoz 2 layers), len(states[0][0]) = 2(bcoz cell and hid states)

        #   |   ((hid_1_sample_1, cell_1_sample_1), (hid_2_sample_1,cell_2_sample_1))   |
        #   |   ((hid_1_sample_2, cell_1_sample_2), (hid_2_sample_2,cell_2_sample_2))   |
        #   |   ((hid_1_sample_3, cell_1_sample_3), (hid_2_sample_3,cell_2_sample_3))   |
        #   |   ((hid_1_sample_4, cell_1_sample_4), (hid_2_sample_4,cell_2_sample_4))   |

        new_options = []
        for i in range(
            len(final_states)
        ):  # for each sample (or beam), kepp in mind states remain same at sample level

            for j in range(
                self.beam_size
            ):  # for each beam(or ray), scores and ids difffer

                # Info needed to update old option
                update_info = (*ids_scores[i][j], final_states[i])

                # Old option updated and new option added to accumulator list
                new_option = self.update_option(old_options[i], update_info)
                new_options.append(new_option)
        return new_options

    def inputs_from_options(self, options, encoder_output_original):
        """From options, create inputs to the decoder_step

        Args:
            options (List of options): Options
            encoder_output_original (Nested list of tensors): Output of encoder for given sample. Here input batch_size to encoder is 1

        Returns:
            tgt_ids: Target ids - Tensor whose shape is (len(options), 1)
            final_states: Nested list of tensor containing stated to be fed to the recirrent network of decoder
            new_encoder_output: Encoder output scaled to the new batch size i.e. len(options)
        """
        batch_size = len(options)
        sequences, encoder_final, src_mask = encoder_output_original

        # Scaling sequences and src_mask to new batch size.Encoder final is not scaled bcaoz it is not used after 1st decoder step because
        # then batch size is still 1
        new_sequences = tf.concat([sequences] * batch_size, axis=0)
        new_src_mask = tf.concat([src_mask] * batch_size, axis=0)
        new_encoder_output = (new_sequences, encoder_final, new_src_mask)

        # Collecting last item of ids
        tgt_ids = tf.reshape([option["ids"][-1] for option in options], (-1, 1))

        # Collecting all states tensor and packing them to a single Tensor of shape [batch_size, num_layers,num_staes,hid_size]
        states = tf.convert_to_tensor([option["states"] for option in options])
        _, num_layers, num_states, __ = states.shape

        # Collecting final states and packing them at batch level. First at layer level, then at state level(hid or cell)
        posns = [(x, y) for x in range(num_layers) for y in range(num_states)]
        final_states = [[0] * num_states] * num_layers
        for posn in posns:
            layer, state = posn
            final_states[layer][state] = states[:, layer, state, :]

        return tgt_ids, final_states, new_encoder_output

    def extract_best_options(self, options):
        """Extract best options for next time step from given options.And if option is completed i.e. end id is last id append it to completed
        options

        Args:
            options (List of options): List of hypothesis being tracked

        Returns:
            List of options: Best options
        """
        options_sorted = sorted(
            options, key=lambda option: option["avg_score"], reverse=True
        )  # Sorting options based on their avg score
        best_beam_options = options_sorted[: self.beam_size]  # Filtering best options
        best_options = []
        for option in best_beam_options:
            if option["ids"][-1] == self.end_id:
                # If end token is emitted, shift the option into completed options
                self.completed_options.append(option)
            else:
                best_options.append(option)

        return best_options

    def update_option(self, old_option, updated_info):
        """Updates old options with given info and create new options

        Args:
            old_option (Option): Old option
            updated_info (Tuple): Information needed to update

        Returns:
            Option: Updated new option
        """
        token_id, present_score, new_states = updated_info

        # deepcopy because for a given sample, all beam size number of good candidates share same old option.
        # So, it must not be changed inplace. So, new copies needed
        option = deepcopy(old_option)
        option["ids"].append(token_id)
        option["scores"].append(present_score)
        option["avg_score"] = self.scoring_fn(option["scores"])
        option["states"] = new_states
        return option

    def extract_good_ids_scores(self, vocab_probs):
        """Extract good candidates ids and scores from a given probability score

        Args:
            vocab_probs (Tensor): target token probabilities at a given time step of shape (batch_size, 1, hidden size)

        Returns:
            List of list of tuples: good ids and scores
            len(ids_scores) = batch_size, len(ids_scores[0]) = beam_size, len(ids_scores[0][0])  = 2 (id, score)
        """
        # Calculating log probability scores
        log_scores = log(vocab_probs)
        # Sirting ids and scores
        sorted_ids = tf.argsort(log_scores, axis=-1, direction="DESCENDING")
        sorted_scores = tf.sort(log_scores, axis=-1, direction="DESCENDING")

        # Squeezing time step dimension and slicing them to get best beam sized number of candidates for each sample in batch
        best_scores = tf.squeeze(sorted_scores[:, :, : self.beam_size], axis=1)
        best_ids = tf.squeeze(sorted_ids[:, :, : self.beam_size], axis=1)

        # Now len(best_ids_list) = batch_size, len(best_ids_list[0]) = beam_size. Same for best_scores_list
        best_ids_list, best_scores_list = (
            best_ids.numpy().tolist(),
            best_scores.numpy().tolist(),
        )

        # Packing scores and ids at sample level
        # Now len(ids_scores) = batch_size, len(ids_scores[0]) = 2 (ids, scores), len(ids_scores[0][0]) = beam_size
        ids_scores = list(zip(best_ids_list, best_scores_list))

        # Packing (score,id) within sample level
        # Now len(ids_scores) = batch_size, len(ids_scores[0]) = 2 (ids, scores), len(ids_scores[0][0]) = beam_size
        ids_scores = [list(zip(*id_score)) for id_score in ids_scores]
        return ids_scores

    def scoring_fn(self, scores):
        """Function used to score options.Here just a simple arithmetic mean

        Args:
            scores (List of floats): List of log probability scores of ids within a given option

        Returns:
            float: Average log probability score
        """
        return sum(scores) / len(scores)

    def reset(self):
        """Resets completed options to null list"""
        self.completed_options = []

    def translit_best_five(self, word):
        """Given a romanized word, return atleatst best five native script words obtained through beam search

        Args:
            word (str): Romanized word

        Returns:
            List of strings: List of best native script words
        """
        word = word.lower()  # lower casing word

        # Tokenizing word and converting it into a tensor of shape (1, src_len)
        x_in = np.array(self.model.src_tokenizer.encode(word).ids).reshape(1, -1)

        # Extracting best words through beam search
        words = self.process(x_in)
        return words[:5]

    def translit_best_word(self, word):
        """Given a romanized word, return best native script word obtained through beam search

        Args:
            word (str): Romanized word

        Returns:
            str: Best native script word
        """
        word = word.lower()  # lower casing word

        # Tokenizing word and converting it into a tensor of shape (1, src_len)
        x_in = np.array(self.model.src_tokenizer.encode(word).ids).reshape(1, -1)

        # Extracting best words through beam search
        words = self.process(x_in)
        return words[0]

    def translit_sequence(self, seq):
        """Given a sequence of roman alphabets with/without punctuation, translit the letters while preserving the punctuation(or non alphabets)

        Args:
            seq (str): sequence of roman alphabets with/without punctuation

        Returns:
            str: sequence of native script letters post transliteration while preserving punctuation
        """

        # Inserting space on wither side of non alphabets and splitting the sequence on spaces
        tokens = re.sub(r"([^A-Za-z]+)", r" \1 ", seq).split()

        # Transliterating alphabetical sequence(best beam search word) but non alphabet sequence remains unchanged
        words = [
            self.translit_best_word(token) if token.isalpha() else token
            for token in tokens
        ]

        # Joining alphabets and others while preserving the order
        translit_seq = "".join(words)
        return translit_seq

    def translit_sentence(self, sentence):
        """Transliterate romanized word sentence with/without punctuation

        Args:
            sentence (str): Sentence of romanized words

        Returns:
            str: Sentence of native script words
        """
        # Split sentence to sequences on space, transliterate them and join back
        translit_sent = " ".join(
            [self.translit_sequence(seq) for seq in sentence.split()]
        )
        return translit_sent

    def translit_ids_best_five(self, ids):
        """Given tokenized ids of romanized word, give at max best 5 native script words
        Args:
            ids (List of ints): Tokenized ids of romanized word

        Returns:
            List of str: At max best 5 native script words obtained through beam search
        """
        # Reshape list to a tensor and then beam search
        x_in = np.array(ids).reshape(1, -1)
        words = self.process(x_in)
        return words[:5]

    def translit_ids_best_word(self, ids):
        """Given tokenized ids of romanized word, give best  native script word
        Args:
            ids (List of ints): Tokenized ids of romanized word

        Returns:
            str: Best native script word obtained through beam search
        """
        x_in = np.array(ids).reshape(1, -1)
        words = self.process(x_in)
        return words[0]

    def beam_search_metric(self, file):
        """Given a file of romanized and native script words datset, calculate effectiveness of beam search.

        At individual sample level,
        Best_5_score is 1, if native script word is in best 5 words obtained when romanized word is fed to beam search or else 0
        Best_word_score is 1, if native script word is the best word obtained when romanized word is fed to beam search or else 0

        Overall, we average these scores to get dataset level metric


        Args:
            file (pathlib Path): Location of dataset file

        Returns:
            dictionary: Dataset level average of best_5 and best_word scores
        """
        df = pd.read_csv(file, sep="\t", header=None, names=["tgt", "src", "count"])
        src_words = df.src.tolist()
        tgt_words = df.tgt.tolist()
        dataset_len = len(df)
        best_five_tgt = [self.translit_best_five(word) for word in src_words]
        best_five_percent = round(
            sum([tgt_words[i] in best_five_tgt[i] for i in range(len(tgt_words))])
            / dataset_len,
            3,
        )
        best_word_percent = round(
            sum([tgt_words[i] == best_five_tgt[i][0] for i in range(len(tgt_words))])
            / dataset_len,
            3,
        )
        return {
            "best_five_percent": best_five_percent,
            "best_word_percent": best_word_percent,
        }
