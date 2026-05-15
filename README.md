# StoryWritingAI

A decoder-only transformer model designed to generate creative, coherent stories with human-like quality. Built from scratch with engineering optimizations to run efficiently on standard hardware.

## 🎯 Project Overview

StoryWritingAI is a GPT-style language model trained on diverse literary datasets to understand narrative structure, vocabulary depth, and creative flow. The project demonstrates both architectural sophistication and practical engineering—scaling from a lightweight initial prototype to a powerful 45M-parameter master model.

### Key Achievements

| Metric | Initial Version | Final Master Run |
|--------|-----------------|------------------|
| **Vocabulary Size** | 8,000 tokens | 16,000 tokens |
| **Model Depth** | 4 Layers | 6 Layers |
| **Embedding Width** | 256 dimensions | 512 dimensions |
| **Training Data** | 5k lines (Simple) | 60k+ lines (Complex Mix) |
| **Memory Strategy** | Full Load (RAM intensive) | Lazy Loading (Byte-indexed) |
| **Generation Method** | Greedy/Top-K | Nucleus (Top-P) + Top-K |

---

## 🧠 Core Architecture

StoryWritingAI implements a **Decoder-Only Transformer**, the same architecture powering the GPT series.

### Architectural Components

**Multi-Head Self-Attention**
- Enables words to contextualize other words in the sequence
- Example: The model learns that "He" refers to "The Knight" through attention weights
- Scales across multiple representation subspaces simultaneously

**GELU Activation Function**
- Gaussian Error Linear Unit provides smoother gradient flow than traditional ReLU
- Better numerical stability during backpropagation
- Improved convergence properties

**Pre-Norm Scaling**
- Layer normalization applied *before* attention and feed-forward blocks
- Significantly improves training stability compared to post-norm approaches
- Enables training of deeper networks without divergence

**Master Configuration**
- 6 transformer layers
- 512 embedding dimensions
- ~45M total parameters
- Designed to capture complex literary styles and narrative patterns

---

## 📚 Training Data

The model trains on a carefully curated mix of datasets, each serving a specific purpose:

| Dataset | Purpose | Writing Style |
|---------|---------|---------------|
| **TinyStories** | Foundational Grammar | Simple, consistent, and logical |
| **WritingPrompts** | Creative Flow | Modern, narrative-driven, and descriptive |
| **Project Gutenberg** | Vocabulary Depth | Classical, complex, and sophisticated |
| **BookCorpus** | Story Structure | Modern prose and dialogue patterns |

This diversity ensures the model learns grammar, creativity, literary depth, and modern narrative conventions.

---

## ⚙️ Engineering Optimizations

Training a 45M-parameter model on 60,000+ lines of text would typically crash a standard laptop. We solved this with three targeted optimizations:

### 1. Lazy Loading with Byte-Offsets
- **Problem**: Loading 60k+ lines into RAM requires 8GB+ of memory
- **Solution**: Index the dataset by byte position instead of loading entirely into memory
- **Result**: Reduced RAM overhead from 8GB+ to nearly 0MB
- **Implementation**: Efficient file seeking and streaming during batching

### 2. Gradient Accumulation
- **Problem**: Small batch sizes (8 samples) lead to unstable training gradients
- **Solution**: Process 8 samples but only update weights every 4 steps
- **Effect**: Simulates a batch size of 32 with server-grade stability
- **Benefit**: Train on consumer hardware without sacrificing convergence quality

### 3. Cosine Annealing with Warmup
- **Warmup Phase**: Start with a low learning rate to stabilize initial training
- **Decay Phase**: Gradually reduce learning rate using cosine schedule
- **Result**: Model "settles" into optimal weight configuration
- **Prevents**: Overshooting and instability in later training phases

---

## 🎤 Inference Optimizations

To generate human-like stories without repetition or nonsense, we implement a dual-sampling strategy:

### Top-K Sampling
- Restricts model choices to the 50 most probable next words
- Prevents selection of absurd, low-probability tokens
- Reduces exposure to model's "tail" of uncertainty

### Nucleus (Top-P) Sampling
- Dynamically truncates the probability distribution
- Keeps tokens until cumulative probability reaches threshold (0.9)
- Adapts sampling based on model confidence

### Temperature Control
- **Setting**: 0.8–0.85 (the "Goldilocks zone")
- **Too Low** (< 0.5): Repetitive, predictable stories
- **Too High** (> 1.0): Incoherent, nonsensical output
- **Sweet Spot**: Balanced creativity and coherence

**Combined Effect**: Stories that are creative yet coherent, diverse yet meaningful.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 8GB+ RAM (for training; less for inference)
- Git

### Installation

```bash
git clone https://github.com/yourusername/StoryWritingAI.git
cd StoryWritingAI
pip install -r requirements.txt
```

---

## 📖 Training Workflow

Follow these steps in order to train the model:

### Step 1: Download & Mix the Dataset

Download raw text from Hugging Face, shuffle it, and create training/validation splits.

```bash
python -m src.dataset.download_dataset
```

**Expected Output**:
- `data/cleaned/train.jsonl` — Training dataset
- `data/cleaned/validation.jsonl` — Validation dataset

### Step 2: Train the Tokenizer

Build the vocabulary "map" that converts English text into token IDs.

```bash
python -m src.tokenizer.train_tokenizer
```

**Configuration**: Ensure `vocab_size=16000` in your config for the Master Run.

**Expected Output**: `outputs/tokenizer.json`

⚠️ **Critical**: Do not skip this step. The model cannot understand text without a trained tokenizer.

### Step 3: Launch Model Training

Start the training loop with the prepared data and tokenizer.

```bash
python -m src.training.train
```

**Monitoring**:
- Watch the loss metric—it should start dropping immediately
- Model saves a checkpoint at the end of every epoch
- Checkpoints are saved as `master_gpt_epX.pt` (where X is the epoch number)

**Duration**: Varies by hardware (typically 4–12 hours on consumer GPUs/CPUs)

### Step 4: Generate Stories

Once you have at least one checkpoint, test the trained model.

```bash
python -m src.inference.generate
```

This will prompt you for a story prompt and generate creative fiction based on the trained weights.

---

## ⚠️ Critical Reminders

### Vocabulary & Tokenizer Synchronization
- **If you change training data**: You must re-train the tokenizer
- **If tokenizer changes**: You must start model training from scratch
- **Why**: The model's vocabulary is frozen at checkpoint time; mismatches cause crashes

### Module Imports
- Always run commands from the **root directory** (the main `StoryWritingAI` folder)
- Always use the `-m` flag: `python -m src...`
- This ensures Python can find internal imports correctly

### Hardware Recommendations

| Task | Minimum | Recommended |
|------|---------|-------------|
| **Inference (Generate)** | 4GB RAM, CPU | 8GB RAM, GPU |
| **Training (Master Run)** | 8GB RAM, CPU | 16GB+ RAM, GPU (CUDA) |
| **Dataset Preparation** | 4GB RAM | 8GB+ RAM |

---

## 📁 Project Structure

```
StoryWritingAI/
├── src/
│   ├── dataset/
│   │   └── download_dataset.py       # Download & prepare training data
│   ├── tokenizer/
│   │   └── train_tokenizer.py        # Train vocabulary
│   ├── training/
│   │   └── train.py                  # Main training loop
│   ├── inference/
│   │   └── generate.py               # Story generation
│   ├── model/
│   │   └── transformer.py            # Architecture definition
│   └── utils/
│       └── config.py                 # Configuration & hyperparameters
├── data/
│   └── cleaned/                      # Training/validation data (generated)
├── outputs/
│   ├── tokenizer.json                # Trained vocabulary
│   └── master_gpt_epX.pt             # Model checkpoints
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔧 Configuration

Edit `src/utils/config.py` to customize:

- **Vocabulary size**: `vocab_size` (default: 16000)
- **Model depth**: `num_layers` (default: 6)
- **Embedding dimensions**: `d_model` (default: 512)
- **Learning rate**: `learning_rate` (default: 1e-4)
- **Batch size**: `batch_size` (default: 8)
- **Gradient accumulation steps**: `accumulation_steps` (default: 4)
- **Temperature**: `temperature` (default: 0.85)

---

## 📊 Performance Insights

### Training Stability
- Loss should decrease monotonically (with minor fluctuations)
- Warmup phase (first 5-10% of training) stabilizes the model
- Cosine annealing prevents loss plateauing

### Generation Quality
- Early epochs (1-2): Basic grammar, limited creativity
- Mid epochs (3-5): Better narrative flow, more variety
- Late epochs (6+): Complex, sophisticated stories with depth

### Memory Efficiency
- Lazy loading reduces per-sample overhead by ~99%
- Gradient accumulation enables 4x larger effective batch size
- Byte-offset indexing eliminates dataset replication in RAM

---

## 🎓 Learning Resources

### Understanding the Architecture
- **Transformer Paper**: "Attention Is All You Need" (Vaswani et al., 2017)
- **GPT Series**: OpenAI's blog posts on GPT-2 and GPT-3 architecture
- **Token Representation**: Hugging Face tokenizer documentation

### Optimization Techniques
- **Gradient Accumulation**: Training with Effective Batch Sizes larger than Memory (PyTorch documentation)
- **Learning Rate Scheduling**: Cosine Annealing with Warm Restarts (Loshchilov & Hutter, 2016)
- **Sampling Strategies**: "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2020)

---

## 🤝 Contributing

Contributions are welcome! Potential areas for improvement:

- [ ] Implement attention visualization tools
- [ ] Add support for multi-GPU training
- [ ] Create a web interface for story generation
- [ ] Expand dataset sources
- [ ] Implement beam search for better quality generation
- [ ] Add metrics for evaluating story coherence (BLEU, ROUGE, METEOR)

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Datasets**: TinyStories, WritingPrompts, Project Gutenberg, BookCorpus
- **Framework**: PyTorch
- **Inspiration**: OpenAI GPT series, Hugging Face Transformers

---

## 📧 Questions or Feedback?

If you have questions, suggestions, or find bugs, please open an issue on GitHub. Happy story generation! 📖✨