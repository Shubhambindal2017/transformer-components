# Transformer Components
Implementation of key components of the Transformer architecture.

### Assumptions
Throughout this process, I made a few assumptions:
```
1. Attention here is self-attention, and not cross-attention.
2. The embedding dimension passed in multi-head-attention is always divisble by num_heads.
3. Defaulted to self-attention for the encoder (type argument in the Scaled Dot Product Attention class), with masking options available for the decoder.
4. Haven't yet added dropout, layer normalization, and residual connections - can be added.
```

### Test Results
Run unit_tests.py
```
- Scaled Dot-Product Attention: Passed.
- Multi-Head Attention: Passed.
- Position-Wise Feed-Forward Network: Passed.
- Positional Encoding: Passed.
- Transformer Encoder Layer: Passed.
```

### Reference:
```
@inproceedings{NIPS2017_3f5ee243,
 author = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, \L ukasz and Polosukhin, Illia},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {I. Guyon and U. Von Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {Attention is All you Need},
 url = {https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf},
 volume = {30},
 year = {2017}
}
```