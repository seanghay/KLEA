# KLEA

An open-source Khmer Word to Speech Model.

### Setup

```shell
pip install -r requirements.txt
```

### Inference

```shell
python infer.py "មនុស្សខ្មែរ"
```

This will output a file called `audio.wav` in the current directory. Output audio sample rate is 22.05 kHz.

## Reference

- [VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://github.com/jaywalnut310/vits)