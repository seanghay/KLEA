#!/usr/bin/env python3
import torch
import utils
from models import SynthesizerTrn

_pad = "_"
_punctuation = ". "
_letters_ipa = "acefhijklmnoprstuwzĕŋŏŭɑɓɔɗəɛɡɨɲʋʔʰː"

symbols = [_pad] + list(_punctuation) + list(_letters_ipa)

hps = utils.get_hparams_from_file("config.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model
)

ckpt = torch.load("./G_60000.pth", map_location="cpu")
net_g.load_state_dict(ckpt["model"])
net_g.eval()
net_g.dec.remove_weight_norm()


def infer_forward(text, text_lengths, scales, sid=None):
    noise_scale = scales[0]
    length_scale = scales[1]
    noise_scale_w = scales[2]
    audio = net_g.infer(
        text,
        text_lengths,
        noise_scale=noise_scale,
        length_scale=length_scale,
        noise_scale_w=noise_scale_w,
        sid=sid,
    )[0].unsqueeze(1)
    return audio


net_g.forward = infer_forward

dummy_input_length = 50

num_symbols = len(symbols)
sequences = torch.randint(
    low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long
)
sequence_lengths = torch.LongTensor([sequences.size(1)])

# noise, noise_w, length
scales = torch.FloatTensor([0.667, 1.0, 0.8])
dummy_input = (sequences, sequence_lengths, scales, None)

torch.onnx.export(
    model=net_g,
    args=dummy_input,
    f=str("output.onnx"),
    verbose=False,
    opset_version=15,
    input_names=["input", "input_lengths", "scales", "sid"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "phonemes"},
        "input_lengths": {0: "batch_size"},
        "output": {0: "batch_size", 1: "time"},
    },
)
