# LLM Text Detection

This project implements a BERT-based model for detecting LLM-generated text vs human-written text.

## Project Structure

- `detect_llm.py`: Main script for training the BERT model
- `evaluate.py`: Script for evaluating the trained model
- `utils.py`: Shared utility functions for data processing and metrics

## Setup

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Create and activate conda environment:
```bash
conda create -n nlp python=3.11
conda activate nlp
```

3. Install dependencies:
```bash
pip install torch transformers datasets evaluate scikit-learn
```

## Data

The project uses the Ghostbuster dataset (English) for training and evaluation. Place your data in the `data/ghostbuster-data` directory with the following structure:

```
data/ghostbuster-data/
├── essay/
│   ├── human/
│   └── [llm-generated]/
├── reuter/
│   ├── human/
│   └── [llm-generated]/
└── wp/
    ├── human/
    └── [llm-generated]/
```

## Usage

1. Train the model:
```bash
python detect_llm.py
```

2. Evaluate the trained model:
```bash
python evaluate.py
```

## Model

The model uses the BERT base uncased architecture fine-tuned for binary classification (human vs LLM-generated text). The trained model will be saved in the `model_output_basic` directory. 