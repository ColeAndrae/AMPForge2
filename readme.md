# AMPForge2

A transformer-based model for generating novel antimicrobial peptides (AMPs) with predicted minimum inhibitory concentration (MIC) values.

<img width="933" height="707" alt="image" src="https://github.com/user-attachments/assets/4ef8eb93-038e-4832-a4ff-a0aeca786800" />

## Overview

AMPForge2 is a deep learning framework that combines autoregressive language modeling, contrastive learning, and regression to generate novel antimicrobial peptides. The model is trained on three types of data:

1. **General peptides** from PeptideAtlas for basic protein sequence understanding
2. **Labeled AMP/non-AMP data** for antimicrobial activity classification 
3. **MIC value data** for quantitative antimicrobial potency prediction

## Key Features

- **Multi-stage training**: Sequential training on general peptides, contrastive AMP learning, and MIC prediction
- **Transformer architecture**: 8-layer transformer with multi-head attention and positional encodings
- **Contrastive learning**: KL-divergence based loss to distinguish AMPs from non-AMPs
- **MIC prediction**: Regression head for quantitative antimicrobial potency estimation
- **Novel peptide generation**: Autoregressive generation with MIC-based filtering

## Model Architecture

- **Vocabulary size**: 23 tokens (20 amino acids + 3 special tokens)
- **Embedding dimension**: 256
- **Attention heads**: 8
- **Layers**: 8
- **Context length**: 27 tokens
- **Parameters**: ~6.3M

## Requirements

- Python 3.8+
- PyTorch 1.9+
- pandas
- tqdm

## Quick Start

1. Prepare your data files:
   - `Generalized_Peptide_Data.txt`: General peptide sequences
   - `AMP_Data.txt`: Known antimicrobial peptides
   - `non-AMP_Data.txt`: Non-antimicrobial peptide sequences
   - `MIC_Value_Data.csv`: Peptide sequences with MIC values

2. Run the training pipeline:
```bash
python train.py
```

The model will automatically:
- Load and preprocess all datasets
- Train sequentially on general peptides, contrastive AMP data, and MIC prediction
- Generate 10 novel AMPs with predicted MIC values

## Data Format

### Input Files
- **Text files**: One peptide sequence per line
- **CSV file**: Two columns - 'sequence' and 'value' (MIC in μg/ml)

### Generated Output
Novel peptides with sequences between 6-25 amino acids and predicted MIC values < 20 μg/ml.

## Training Process

1. **General pretraining** (5000 steps): Autoregressive language modeling on general peptides
2. **Contrastive learning** (5000 steps): Combined autoregressive + contrastive loss to learn AMP representations
3. **MIC prediction** (5000 steps): Regression training for quantitative potency prediction

## License

MIT License - See LICENSE file for details.

## Citation

This work is based on the AMPCLGPT research. If you use AMPForge2 in your research, please cite both:

```bibtex
@article{hu2025harnessing,
  title={Harnessing Generative Pre-trained Transformer for Antimicrobial Peptide Generation and MIC Prediction with Contrastive Learning},
  author={Hu, Keer and Xiao, Yang and Liu, Xiao and Ma, Shaohua},
  journal={bioRxiv preprint},
  year={2025},
  doi={10.1101/2025.03.07.642021},
  note={Original research that this work is based upon}
}

@software{andrae2025ampforge2,
  title={AMPForge2: Transformer-based Antimicrobial Peptide Generation},
  author={Andrae, Cole},
  year={2025},
  version={2.2.0},
  note={Implementation based on AMPCLGPT methodology}
}
```
