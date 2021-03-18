from pathlib import Path

import numpy as np
import yaml

from .model import model_creator
from .transliteration_tokenizers import create_source_target_tokenizers


def model_loader(file_dict):
    """Given path to directory containing config file and weights file, create model based on config and load the weights

    Args:
        file_dict (dict): File dict containing
            "src_corpus_file" - source corpus file
            "tgt_corpus_file" - target corpus file
            "model_dir" - directory containing config file and weights file.

    Returns:
        Transliteration MOdel: Model loaded with weights in the file
    """
    model_dir = file_dict.pop("model_dir")
    src_corpus_file = file_dict.pop("src_corpus_file")
    tgt_corpus_file = file_dict.pop("tgt_corpus_file")

    config_file = model_dir / "config.yaml"
    weights_file = model_dir / "model-best.h5"

    # Dumping config file data into a dict
    with open(config_file) as f:
        config_dict = yaml.safe_load(f)

    pop_keys = [
        "wandb_version",
        "_wandb",
        "batch_size",
        "clipnorm",
        "dataset_type",
        "epochs",
        "learning_rate",
        "momentum",
        "optim_type",
    ]  # Remove check every for subsequent languages

    # Removing unwanted keys from config_dict
    _ = [config_dict.pop(key) for key in pop_keys]

    # creating params dict from config dict
    params_dict = {key: value["value"] for key, value in config_dict.items()}
    src_vocab_size = params_dict.pop("src_vocab_size")
    tgt_vocab_size = params_dict.pop("tgt_vocab_size")
    src_tokenizer, tgt_tokenizer = create_source_target_tokenizers(
        src_corpus_file, tgt_corpus_file, src_vocab_size, tgt_vocab_size
    )
    params_dict["src_tokenizer"] = src_tokenizer
    params_dict["tgt_tokenizer"] = tgt_tokenizer

    # Creating model from params dict
    model = model_creator(params_dict)

    # creating a random input tensor of dimension 2 and dtype int
    inputs = (np.random.randint(0, 10, (1, 5)), np.random.randint(0, 10, (1, 4)))

    # calling model on random inputs to buils the layers inside.
    # Next time write build_config and get_config methods for layers and models
    _ = model(inputs)

    # Loading weights into models
    model.load_weights(str(weights_file))
    return model
