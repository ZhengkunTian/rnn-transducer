# RNN-Transducer
A Pytorch Implementation of Transducer Model for End-to-End Speech Recognition. 

**Email**: zhengkun.tian@nlpr.ia.ac.cn

# Environment
- pytorch >= 0.4
- warp-transducer

## Preparation
We utilize Kaldi for data preparation. At least these files(text, feats.scp) should be included in the training/development/test set. If you apply cmvn, utt2spk and cmvn.scp are required. The format of these file is consistent with Kaidi. The format of vocab is as follows.

```
<blk> 0
<unk> 1
我 2
你 3
...
```
## Train
```python
python train.py -config config/aishell.yaml
```

## Eval
```
python eval.py -config config/aishell.yaml
```

## Experiments
The details of our RNN-Transducer are as follows.
```yaml
model:
    enc:
        type: lstm
        hidden_size: 320
        n_layers: 4
        bidirectional: True
    dec:
        type: lstm
        hidden_size: 512
        n_layers: 1
    embedding_dim: 512
    vocab_size: 4232
    dropout: 0.2
```
All experiments are conducted on AISHELL-1. During decoding, we use beam search with width of 5 for all the experiments. A character-level 5-gram language model from training text, is integrated into beam searching by shallow fusion. 

| MODEL | DEV(CER) | TEST(CER) |
|:---: | :---:|:---: |
| RNNT+pretrain+LM | 10.13 | 11.82 |

## Acknowledge
Thanks to [warp-transducer](https://github.com/HawkAaron/warp-transducer).
