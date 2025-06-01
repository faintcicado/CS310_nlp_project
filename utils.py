import os
from typing import List, Dict, Tuple
import random
from datasets import Dataset, DatasetDict
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

def load_ghostbuster_data(data_dir: str = 'data/ghostbuster-data') -> Tuple[List[str], List[int]]:
    """
    Load ghostbuster dataset and create labels
    Args:
        data_dir: root directory of ghostbuster data
    Returns:
        texts: list of text samples
        labels: list of labels (0 for human, 1 for LLM)
    """
    domains = ['essay', 'reuter', 'wp']
    all_texts = []
    all_labels = []
    
    for domain in domains:
        domain_path = os.path.join(data_dir, domain)
        if not os.path.exists(domain_path):
            print(f"Warning: {domain_path} does not exist")
            continue
            
        # Load human texts (label 0)
        human_path = os.path.join(domain_path, 'human')
        if os.path.exists(human_path):
            human_texts = read_text_files(human_path)
            all_texts.extend(human_texts)
            all_labels.extend([0] * len(human_texts))
            print(f"Loaded {len(human_texts)} human texts from {domain}")
        
        # Load LLM generated texts (label 1)
        for subdir in os.listdir(domain_path):
            subdir_path = os.path.join(domain_path, subdir)
            # Skip human directory, hidden directories, logprobs directory, and prompt directory
            if (os.path.isdir(subdir_path) and 
                subdir != 'human' and 
                not subdir.startswith('.') and 
                not subdir == 'logprobs' and 
                not subdir == 'prompt'):
                llm_texts = read_text_files(subdir_path)
                all_texts.extend(llm_texts)
                all_labels.extend([1] * len(llm_texts))
                print(f"Loaded {len(llm_texts)} LLM texts from {domain}/{subdir}")
    
    print(f"\nTotal dataset statistics:")
    print(f"Total samples: {len(all_texts)}")
    print(f"Human samples: {all_labels.count(0)}")
    print(f"LLM samples: {all_labels.count(1)}")
    
    return all_texts, all_labels

def create_dataset(texts: List[str], labels: List[int], 
                  train_ratio: float = 0.8, 
                  seed: int = 42) -> DatasetDict:
    """
    Create train and validation datasets
    """
    # Create indices for splitting
    indices = list(range(len(texts)))
    random.seed(seed)
    random.shuffle(indices)
    
    split_idx = int(len(indices) * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create train dataset
    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels
    })
    
    # Create validation dataset
    val_texts = [texts[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    val_dataset = Dataset.from_dict({
        'text': val_texts,
        'label': val_labels
    })
    
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

def tokenize_function(examples, tokenizer, max_length=512):
    """
    Tokenize the text data
    """
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

def compute_metrics(pred):
    """
    Compute evaluation metrics for the model
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    } 