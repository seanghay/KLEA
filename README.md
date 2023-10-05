# KLEA

An open-source Khmer Word to Speech Model.

<a target="_blank" href="https://colab.research.google.com/github/https://colab.research.google.com/drive/1Dao9iXxaEVrGTzoUVQL63UbKDK3G55ts?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### 1. Setup

```shell
pip install -r requirements.txt
```

### 2. Download Checkpoint

[G_35000.pth](https://huggingface.co/spaces/seanghay/KLEA/resolve/main/G_35000.pth)

```shell
wget https://huggingface.co/spaces/seanghay/KLEA/resolve/main/G_35000.pth
```

Please the checkpoint in the current directory.

### 3. Inference

```shell
python infer.py "មនុស្សខ្មែរ"
```

This will output a file called `audio.wav` in the current directory. Output audio sample rate is 22.05 kHz.

### Gradio

```
python app.py
```

## Reference

- [VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://github.com/jaywalnut310/vits)
