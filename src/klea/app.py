# -*- coding: utf-8 -*-
import gradio as gr
from models import SynthesizerTrn
from khmer_phonemizer import phonemize_single
import utils
import commons
import torch
import khmernormalizer

_pad = "_"
_punctuation = ". "
_letters_ipa = "acefhijklmnoprstuwzĕŋŏŭɑɓɔɗəɛɡɨɲʋʔʰː"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters_ipa)

# Special symbol ids
SPACE_ID = symbols.index(" ")

_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def text_to_sequence(text):
    sequence = []
    for symbol in text:
        symbol_id = _symbol_to_id[symbol]
        sequence += [symbol_id]
    return sequence


def get_text(text, hps):
    text_norm = text_to_sequence(text)

    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


hps = utils.get_hparams_from_file("config.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model
)

_ = net_g.eval()
_ = utils.load_checkpoint("G_60000.pth", net_g, None)

def generate_voice(text):
    text = khmernormalizer.normalize(text)
    text = " ".join(phonemize_single(text) + ["."])
    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )

    return (hps.data.sampling_rate, audio)


with gr.Blocks(
    title="Khmer Word to Speech",
    theme=gr.themes.Default(
        font=[gr.themes.GoogleFont("Noto Sans Khmer"), "Arial", "sans-serif"]
    ),
) as blocks:
    gr.Markdown("# Khmer Word to Speech")

    input_text = gr.Text(label="ពាក្យខ្លី", lines=1)
    examples = gr.Examples(examples=["មនុស្សជាតិ", "ភ្នំព្រះ"], inputs=[input_text])
    run_button = gr.Button(value="បង្កើត")

    out_audio = gr.Audio(
        label="សំឡេងដែលបានបង្កើត",
        type="numpy",
    )

    inputs = [input_text]
    outputs = [out_audio]

    run_button.click(
        fn=generate_voice,
        inputs=inputs,
        outputs=outputs,
        queue=True,
    )


blocks.queue(concurrency_count=1).launch(debug=True)
