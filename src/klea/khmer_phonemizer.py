r"""
Khmer Phonemizer - A Free, Standalone and Open-Source Khmer Grapheme-to-Phonemes.
"""
import os
import csv
from g2p import PhonetisaurusGraph

def _read_lexicon_file(file):
    lexicon = {}
    with open(file) as infile:
        for line in csv.reader(infile, delimiter="\t"):
            word, phonemes = line
            word, phonemes = word.strip(), phonemes.strip().split()
            lexicon[word] = phonemes
    return lexicon

_graph_file = os.path.join(os.path.dirname(__file__), "km_phonemizer.npz")
_lexicon_file = os.path.join(os.path.dirname(__file__), "km_lexicon.tsv")
_lexicon_dict = _read_lexicon_file(_lexicon_file)
_graph = PhonetisaurusGraph.load(_graph_file, preload=False)

def _phoneticize(word: str, beam: int, min_beam: int, beam_scale: float):
    results = _graph.g2p_one(word, beam=beam, min_beam=min_beam, beam_scale=beam_scale)
    results = list(results)
    if len(results) == 0:
        return None
    return results[0]


def phonemize_single(
    word,
    beam: int = 500,
    min_beam: int = 100,
    beam_scale: float = 0.6,
    use_lexicon: bool = True,
):
    r"""
    Phonemize a single word. The word must match [a-zA-Z\u1780-\u17dd]+
    """
    if word is None:
        return None
    word = word.lower()
    if use_lexicon and word in _lexicon_dict:
        return _lexicon_dict[word]
    return _phoneticize(word, beam=beam, min_beam=min_beam, beam_scale=beam_scale)
