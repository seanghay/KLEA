import onnxruntime
import numpy as np
from wavfile import write as write_wav
from utils import get_hparams_from_file
from commons import intersperse
from khmer_phonemizer import phonemize_single

def audio_float_to_int16(
    audio: np.ndarray, max_wav_value: float = 32767.0
) -> np.ndarray:
    """Normalize audio and convert to int16 range"""
    audio_norm = audio * (max_wav_value / max(0.01, np.max(np.abs(audio))))
    audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
    audio_norm = audio_norm.astype("int16")
    return audio_norm

symbols = [
    "_",
    ".",
    " ",
    "a",
    "c",
    "e",
    "f",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "r",
    "s",
    "t",
    "u",
    "w",
    "z",
    "ĕ",
    "ŋ",
    "ŏ",
    "ŭ",
    "ɑ",
    "ɓ",
    "ɔ",
    "ɗ",
    "ə",
    "ɛ",
    "ɡ",
    "ɨ",
    "ɲ",
    "ʋ",
    "ʔ",
    "ʰ",
    "ː",
]
symbol_to_id = {s: i for i, s in enumerate(symbols)}


def text_to_sequence(text):
    sequence = []
    for symbol in text:
        symbol_id = symbol_to_id[symbol]
        sequence += [symbol_id]
    return sequence


def get_text(text, hps):
    text_norm = text_to_sequence(text)
    if hps.data.add_blank:
        text_norm = intersperse(text_norm, 0)
    return text_norm


def infer():
    session_options = onnxruntime.SessionOptions()
    providers = ["CPUExecutionProvider"]
    model = onnxruntime.InferenceSession(
        "./output.onnx", sess_options=session_options, providers=providers
    )

    hps = get_hparams_from_file("config.json")
    text = " ".join(phonemize_single("ទិញបាយ") + ["."])
    stn_tst = get_text(text, hps)

    text = np.expand_dims(np.array(stn_tst, dtype=np.int64), 0)
    text_lengths = np.array([text.shape[1]], dtype=np.int64)
    scales = np.array(
        [0.667, 1, 0.8],
        dtype=np.float32,
    )
    sample_rate = 22050
    sid = None
    audio = model.run(
        None,
        {
            "input": text,
            "input_lengths": text_lengths,
            "scales": scales,
            "sid": sid,
        },
    )[0].squeeze((0, 1))
    audio = audio_float_to_int16(audio.squeeze())
    write_wav("audio.wav", sample_rate, audio)
      
     
if __name__ == "__main__":
    infer()
