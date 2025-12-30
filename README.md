# KLEA

An open-source Khmer Word to Speech Model. Just single word not sentence!

<a target="_blank" href="https://colab.research.google.com/drive/1Dao9iXxaEVrGTzoUVQL63UbKDK3G55ts?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Using as a Python Package

```bash
sudo apt-get install libsndfile1 python3-dev
wget https://huggingface.co/spaces/seanghay/KLEA/resolve/main/G_60000.pth
# G_60000.pth must be in the same folder where you `uv run` 
uv run --with 'klea @ git+https://github.com/djsamseng/KLEA' python -c 'import klea; klea.run_for_word("ទឹកធ្លាក់", "ទឹកធ្លាក់.wav")'
ffplay ទឹកធ្លាក់.wav
```

### Dataset

This model was trained on kheng.info dataset. You can find it on http://kheng.info or at https://hf.co/datasets/seanghay/khmer_kheng_info_speech

## Reference

- [VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://github.com/jaywalnut310/vits)
- [kheng.info](https://kheng.info/about/) is an online audio dictionary for the Khmer language with over 3000 recordings. Kheng.info is backed by multiple dictionaries and a large text corpus, and supports search in English and Khmer with search results ordered by word frequency.

