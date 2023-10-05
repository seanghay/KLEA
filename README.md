# KLEA

An open-source Khmer Word to Speech Model. Just single word not sentence!

<a target="_blank" href="https://colab.research.google.com/drive/1Dao9iXxaEVrGTzoUVQL63UbKDK3G55ts?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### 1. Setup

```shell
pip install -r requirements.txt
```

### 2. Download Checkpoint

[G_60000.pth](https://huggingface.co/spaces/seanghay/KLEA/resolve/main/G_60000.pth)

```shell
wget https://huggingface.co/spaces/seanghay/KLEA/resolve/main/G_60000.pth
```

Place the checkpoint in the current directory.

### 3. Inference

```shell
python infer.py "មនុស្សខ្មែរ"
```

This will output a file called `audio.wav` in the current directory. Output audio sample rate is 22.05 kHz.

### Gradio

```
python app.py
```


### Colab

<img width="837" alt="image" src="https://github.com/seanghay/KLEA/assets/15277233/ac0da746-2a6c-439f-85da-a70e35efb85f">



### Dataset

This model was trained on kheng.info dataset. You can find it on http://kheng.info or at https://hf.co/datasets/seanghay/khmer_kheng_info_speech

## Reference

- [VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://github.com/jaywalnut310/vits)
- [kheng.info](https://kheng.info/about/) is an online audio dictionary for the Khmer language with over 3000 recordings. Kheng.info is backed by multiple dictionaries and a large text corpus, and supports search in English and Khmer with search results ordered by word frequency.
