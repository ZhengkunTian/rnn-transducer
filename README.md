# RNN-Transducer
A Pytorch Implementation of Transducer Model for End-to-End Speech Recognition. 

**Email**: zhengkun.tian@nlpr.ia.ac.cn

# Environment
- pytorch >= 0.4
- warp-transducer

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
