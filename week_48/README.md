# No Libraries, No Shortcuts: LLM from Scratch with PyTorch

A comprehensive, hands-on guide to building, training, and fine-tuning a Transformer architecture from the ground up using pure PyTorch.

**Author:** Simanga Mchunu  
**Original Article:** Towards AI

## Overview

This project teaches you how to build a GPT-style language model entirely from scratch—no high-level abstractions, no shortcuts. You'll understand every component of the Transformer architecture by implementing it yourself, from tokenization through attention mechanisms to the complete training pipeline.

By the end, you'll have a working LLM capable of generating Coldplay-style lyrics and producing coherent English text.

## What You'll Learn

- **Tokenization**: Converting raw text into numerical tokens using tiktoken
- **Embeddings & Positional Encoding**: Representing tokens as dense vectors with positional information
- **Self-Attention**: Understanding how tokens interact and share information
- **Multi-Head Attention**: Combining multiple attention perspectives for richer representations
- **Feed-Forward Networks**: Transforming attention outputs into deeper features
- **Decoder Architecture**: Building residual connections and layer normalization
- **Model Pretraining**: Training on IMDb reviews to learn English patterns
- **Fine-Tuning**: Adapting the model to Coldplay lyrics for style-specific generation
- **Advanced Training Concepts**: Gradient clipping, early stopping, learning rate scheduling

## Architecture

The model implements a GPT-style decoder-only Transformer with:

- **Token Embeddings**: Maps vocabulary tokens to dense vectors
- **Positional Embeddings**: Encodes sequence position information
- **Multi-Head Self-Attention**: Parallel attention mechanisms with causal masking
- **Feed-Forward Networks**: Up-projection → GELU activation → Down-projection
- **Residual Connections**: Preserves gradient flow through deep networks
- **Layer Normalization**: Stabilizes training by normalizing activations
- **LM Head**: Linear projection to vocabulary for next-token prediction

## Key Components

### SelfAttention
Implements scaled dot-product attention with causal masking to prevent the model from attending to future tokens.

### MultiHeadAttention
Runs multiple attention heads in parallel, each focusing on different aspects of the data, then concatenates results.

### FeedForward
Two-layer network with GELU activation that processes attention outputs.

### Decoder
Combines attention, feed-forward, and normalization layers with residual connections.

### GPT
Complete Transformer model stacking multiple decoder blocks with embeddings and output projection.

## Training Pipeline

### Data Preparation
- Load IMDb dataset and combine with Coldplay lyrics
- Clean text to remove non-ASCII characters
- Create sliding-window dataset for next-token prediction
- Separate input and target sequences

### Training Features
- **Loss Function**: Cross-entropy loss for next-token prediction
- **Optimizer**: AdamW with weight decay for improved generalization
- **Learning Rate Scheduling**: CosineWithWarmup for stable convergence
- **Gradient Clipping**: Prevents exploding gradients in deep networks
- **Early Stopping**: Monitors validation loss to prevent overfitting
- **Checkpointing**: Saves best model state automatically

### Fine-Tuning
After pretraining on IMDb, the model is fine-tuned on Coldplay lyrics with:
- Lower learning rates to preserve pretrained weights
- Fewer epochs due to small dataset size
- Adjusted hyperparameters for stability

## Hyperparameters

```python
settings = {
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "num_epochs": 300,
    "batch_size": 32,
    "warmup_steps": 1500,
    "max_lr": 3e-4,
    "min_lr": 3e-5,
    "eval_freq": 200,
    "gradient_clip": 1.0,
    "patience": 50,
}

settings_ft = {
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "num_epochs": 5,
    "batch_size": 4,
    "warmup_steps": 100,
}
```

## Installation

```bash
pip install torch transformers datasets tiktoken
```

## Quick Start

### 1. Prepare Data
```python
from datasets import load_dataset
import re

ds = load_dataset("stanfordnlp/imdb")

def keep_english_only(text):
    return re.sub(r"[^\x00-\x7F]+", "", text)

train_text = " ".join([keep_english_only(t) for t in ds['train']['text']])
```

### 2. Create Model
```python
from torch import nn

model = GPT(
    num_heads=8,
    vocab_size=5000,
    embed_dim=256,
    attention_dim=256,
    num_blocks=8,
    context_length=256,
    dropout_rate=0.1
)
```

### 3. Train
```python
train_losses, val_losses, tokens = train_model(
    model, train_loader, val_loader, device, settings
)
```

### 4. Generate
```python
token_ids = generate(
    model=model,
    context=text_to_token_ids("I want something", tokenizer, device),
    max_new_tokens=50,
    context_length=256
)
print(token_ids_to_text(token_ids, tokenizer))
```

## Sample Outputs

**After Pretraining (IMDb):**
```
the movie starts slow and i thought it was going to be boring, 
but then going to be interesting. the acting is okay, some are 
boring felt like they just gave up.
```

**After Fine-Tuning (Coldplay):**
```
lights go out and the stars begin to fall i hear your voice 
across the night. lights are running in circles chasing the echoes. 
you are the star that keeps me alive. Oh-ooh-oh-ooh oh, oh
```

## File Structure

```
.
├── model.py              # GPT model architecture
├── attention.py          # Attention mechanisms
├── data.py              # Data loading and preprocessing
├── train.py             # Training loop and utilities
├── generate.py          # Text generation
├── requirements.txt     # Dependencies
├── checkpoints/         # Saved model weights
└── README.md
```

## Technical Highlights

- **No External Model Architectures**: Everything built from first principles
- **Scaled Dot-Product Attention**: Proper scaling and causal masking
- **Efficient Batch Processing**: All operations vectorized for GPU efficiency
- **Comprehensive Logging**: Track training/validation loss and learning rates
- **Checkpoint Management**: Automatically save best models
- **Flexible Generation**: Supports temperature and top-k sampling

## Understanding Key Concepts

### Causal Masking
Prevents the model from attending to future tokens during training, enforcing unidirectional attention for proper autoregressive generation.

### Residual Connections
Allow gradients to flow directly through skip connections, enabling training of very deep networks without vanishing gradients.

### Layer Normalization
Normalizes activations across features for a single example, stabilizing training and enabling faster convergence.

### Learning Rate Warmup
Gradually increases learning rate from 0 to maximum over initial steps, preventing divergence during early training.

## Performance Metrics

The model tracks:
- **Training Loss**: Cross-entropy loss on training batches
- **Validation Loss**: Cross-entropy loss on held-out validation set
- **Tokens Seen**: Total number of tokens processed
- **Learning Rate**: Adaptive learning rate over time

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers library
- datasets library
- tiktoken

## References

[1] Vaswani et al. (2017). Attention is all you need. arXiv:1706.03762

[2] Radford et al. (2018). Improving Language Understanding by Generative Pre-Training

[3] Ba, Kiros, Hinton (2016). Layer Normalization. arXiv:1607.06450

## Notes

- This implementation prioritizes clarity over efficiency. Production models use optimized kernels and mixed precision training.
- The small model size (8 layers, 256 dimensions) is for educational purposes. Increase for better results.
- Training on consumer hardware may take hours to days depending on GPU.
- Fine-tuning requires significantly less compute than pretraining.

## Contributing

Feel free to extend this project with:
- Quantization for faster inference
- Multi-GPU training support
- Different tokenization strategies
- Additional fine-tuning datasets
- Advanced sampling techniques

## License

MIT License - See LICENSE file for details

---

**Ready to build your own LLM?** Clone the repository, follow the installation steps, and start training!

---
## AUTHOR
- Simanga Mchunu