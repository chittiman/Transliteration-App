import yaml
from tokenizers import Tokenizer
import numpy as np

from pathlib import Path
from scripts.transliteration_tokenizers import create_source_target_tokenizers
from scripts.beam_search import BeamSearch
from .model import model_creator

def transliterate_sentence(sentence, language):
    language = language.lower()
    searcher = beam_searcher_loader(language)
    converted_sentence = searcher.translit_sentence(sentence)
    return converted_sentence

def beam_searcher_loader(language):
    cur_dir = Path.cwd()
    data_dir = cur_dir / "data"
    lang_dir = data_dir / language
    model_dir = lang_dir / "models"
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
        "src_vocab_size",
        "tgt_vocab_size"
    ]

    # Removing unwanted keys from config_dict
    _ = [config_dict.pop(key) for key in pop_keys]

    # creating params dict from config dict
    params_dict = {key: value["value"] for key, value in config_dict.items()}
    
    src_tokenizer, tgt_tokenizer = load_tokenizers(language)
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
    searcher = BeamSearch(model, 5)
    return searcher

def create_tokenizers_from_config_file(language):
    cur_dir = Path.cwd()
    data_dir = cur_dir / "data"
    lang_dir = data_dir / language
    
    corpus_dir = lang_dir / "corpus"
    src_corpus_file = corpus_dir / "source_corpus.txt"
    tgt_corpus_file = corpus_dir / "target_corpus.txt"
    
    model_dir = lang_dir / "models"
    config_file = model_dir / "config.yaml"  
    src_tokenizer_file = model_dir / "src_tokenizer.json"
    tgt_tokenizer_file = model_dir / "tgt_tokenizer.json"
    
    with open(config_file) as f:
        config_dict = yaml.safe_load(f)
        
    src_vocab_size = config_dict["src_vocab_size"]["value"]
    tgt_vocab_size = config_dict["tgt_vocab_size"]["value"]
    
    src_tokenizer, tgt_tokenizer = create_source_target_tokenizers(src_corpus_file, tgt_corpus_file,\
                                                                   src_vocab_size, tgt_vocab_size)
    src_tokenizer.save(str(src_tokenizer_file))
    tgt_tokenizer.save(str(tgt_tokenizer_file))
    
def load_tokenizers(language):
    cur_dir = Path.cwd()
    data_dir = cur_dir / "data"
    lang_dir = data_dir / language
    model_dir = lang_dir / "models"
    src_tokenizer_file = model_dir / "src_tokenizer.json"
    tgt_tokenizer_file = model_dir / "tgt_tokenizer.json"
    src_tokenizer = Tokenizer.from_file(str(src_tokenizer_file))
    tgt_tokenizer = Tokenizer.from_file(str(tgt_tokenizer_file))
    return src_tokenizer,tgt_tokenizer    
    