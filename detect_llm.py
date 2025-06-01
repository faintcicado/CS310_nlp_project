import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
import os

# Import functions from utils.py
from utils import (
    load_ghostbuster_data,
    create_dataset,
    tokenize_function,
    compute_metrics
)

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def is_valid_text(text: str) -> bool:
    """
    Check if the text is valid (contains actual sentences rather than just numbers)
    """
    # Skip empty texts
    if not text:
        return False
        
    # Split into words and check first few words
    words = text.split()
    if len(words) < 10:  # Text should have at least 10 words
        return False
        
    # Check if text contains too many numbers
    first_20_words = words[:20] if len(words) >= 20 else words
    number_count = sum(1 for w in first_20_words if all(c.isdigit() or c in '.-' for c in w))
    if number_count > len(first_20_words) / 2:  # If more than half are numbers, skip
        return False
        
    # Check if text starts with common words, capital letters, or common section headers
    first_word = words[0].lower()
    common_starters = {'the', 'in', 'a', 'this', 'it', 'there', 'when', 'while', 'although', 'however', 'introduction', 'abstract', 'background', 'conclusion'}
    if not (first_word in common_starters or words[0][0].isupper()):
        return False
        
    # Check for presence of actual sentences (looking for periods followed by spaces and capital letters)
    text_sample = ' '.join(words[:100])  # Look at first 100 words
    sentences = [s.strip() for s in text_sample.split('.') if s.strip()]
    if not sentences:
        return False
        
    # Check if the text contains tokenized words (common in numerical feature files)
    if any(w.startswith('Ġ') for w in words[:20]):
        return False
        
    return True

def read_text_files(directory: str) -> List[str]:
    """
    Recursively read all text files in a directory
    Skip files that contain numerical features instead of actual text
    """
    texts = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt') and not file.startswith('.'):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if is_valid_text(text):
                            texts.append(text)
                except Exception as e:
                    print(f"Error reading file {file}: {str(e)}")
    return texts

def load_bert_model(model_path='bert-base-uncased/'):
    """
    Load BERT model and tokenizer from local path
    Args:
        model_path: path to the BERT model directory
    Returns:
        model: BERT model for sequence classification
        tokenizer: BERT tokenizer
    """
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    # Load model for binary classification
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2  # Binary classification: human vs LLM
    )
    
    # Move model to GPU if available
    model = model.to(device)
    
    return model, tokenizer

def train_model(train_dataset, val_dataset, model, tokenizer, output_dir='./results'):
    """
    Train the model with GPU support and optimized configuration.
    """
    # Define training arguments with GPU optimization
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # Increased epochs since we have GPU
        per_device_train_batch_size=16,  # Increased batch size for GPU
        per_device_eval_batch_size=16,
        logging_dir='./logs',
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=500,
        fp16=True if torch.cuda.is_available() else False,  # Enable mixed precision training if GPU available
    )
    
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=val_dataset,
        # compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("Starting training with GPU optimization...")
    trainer.train()
    print("Training completed.")
    
    return trainer

if __name__ == "__main__":
    try:
        print("Loading Ghostbuster dataset...")
        texts, labels = load_ghostbuster_data()
        print(f"\nTotal samples: {len(texts)}")
        print(f"Human samples: {labels.count(0)}")
        print(f"LLM samples: {labels.count(1)}")
        
        print("\nCreating train/val splits...")
        dataset = create_dataset(texts, labels)
        print(f"Training samples: {len(dataset['train'])}")
        
        print("\nLoading BERT model...")
        model, tokenizer = load_bert_model()
        print("Successfully loaded BERT model and tokenizer")
        
        print("\nPreparing training dataset...")
        tokenized_train_dataset = dataset['train'].map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=['text']
        )
        print("Training dataset preparation completed")

        print("\nStarting model training...")
        trainer = train_model(
            tokenized_train_dataset, 
            None,  # No val_dataset for now
            model,
            tokenizer
        )
        print("Training completed!")
        
        print("\nSaving the model...")
        model_save_path = './model_output_basic'
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Model saved to {model_save_path}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc() 