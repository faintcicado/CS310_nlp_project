import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import os
import json
from typing import List, Dict, Tuple
import random

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_json_file(file_path: str) -> List[str]:
    """
    Load text data from json file
    """
    texts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Assuming each item in json has a 'text' field
            for item in data:
                if isinstance(item, dict) and 'text' in item:
                    texts.append(item['text'])
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
    return texts

def load_chinese_data(data_dir: str = 'data/face2_zh_json', data_type: str = 'news') -> Tuple[List[str], List[int]]:
    """
    Load Chinese dataset for specified type (news, webnovel, or wiki)
    Args:
        data_dir: root directory of Chinese data
        data_type: type of data to load (news, webnovel, or wiki)
    Returns:
        texts: list of text samples
        labels: list of labels (0 for human, 1 for LLM)
    """
    all_texts = []
    all_labels = []
    
    # Load human texts (label 0)
    human_file = os.path.join(data_dir, 'human/zh_unicode', f'{data_type}-zh.json')
    if os.path.exists(human_file):
        human_texts = load_json_file(human_file)
        all_texts.extend(human_texts)
        all_labels.extend([0] * len(human_texts))
        print(f"Loaded {len(human_texts)} human texts from {data_type}")
    
    # Load LLM generated texts (label 1)
    llm_file = os.path.join(data_dir, 'generated/zh_qwen2', f'{data_type}-zh.qwen2-72b-base.json')
    if os.path.exists(llm_file):
        llm_texts = load_json_file(llm_file)
        all_texts.extend(llm_texts)
        all_labels.extend([1] * len(llm_texts))
        print(f"Loaded {len(llm_texts)} LLM texts from {data_type}")
    
    print(f"\nTotal {data_type} dataset statistics:")
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
    Tokenize the Chinese text data
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
    
    # Calculate accuracy
    accuracy = (preds == labels).mean()
    
    # Calculate precision, recall, and F1 for each class
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_model(train_dataset, val_dataset, model, tokenizer, output_dir='./results'):
    """
    Train the model with GPU support and optimized configuration
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir='./logs',
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=500,
        evaluation_strategy="epoch",
        fp16=True if torch.cuda.is_available() else False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    print("Training completed.")
    
    return trainer

if __name__ == "__main__":
    try:
        # Load news data for training
        print("Loading Chinese news dataset for training...")
        texts, labels = load_chinese_data(data_type='news')
        
        print("\nCreating train/val splits...")
        dataset = create_dataset(texts, labels)
        print(f"Training samples: {len(dataset['train'])}")
        print(f"Validation samples: {len(dataset['validation'])}")
        
        print("\nLoading Chinese BERT model...")
        model_path = 'bert-base-chinese'
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2
        )
        model = model.to(device)
        print("Successfully loaded Chinese BERT model and tokenizer")
        
        print("\nPreparing training dataset...")
        tokenized_train_dataset = dataset['train'].map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=['text']
        )
        
        tokenized_val_dataset = dataset['validation'].map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=['text']
        )
        print("Dataset preparation completed")
        
        print("\nStarting model training...")
        trainer = train_model(
            tokenized_train_dataset,
            tokenized_val_dataset,
            model,
            tokenizer,
            output_dir='./model_output_zh_chinese'
        )
        
        print("\nSaving the model...")
        model_save_path = './model_output_zh_chinese'
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Model saved to {model_save_path}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc() 