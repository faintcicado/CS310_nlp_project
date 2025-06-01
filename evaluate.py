import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
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

def evaluate_model(model, eval_dataset, tokenizer):
    """
    Evaluate the model on the validation dataset
    Args:
        model: The loaded BERT model
        eval_dataset: Tokenized validation dataset
        tokenizer: BERT tokenizer
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    # Define evaluation arguments
    eval_args = TrainingArguments(
        output_dir="./eval_results",
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=16,
        dataloader_drop_last=False,
    )

    # Create Trainer for evaluation
    trainer = Trainer(
        model=model,
        args=eval_args,
        compute_metrics=compute_metrics
    )

    # Run evaluation
    metrics = trainer.evaluate(eval_dataset)
    return metrics

def print_detailed_metrics(metrics, eval_dataset):
    """
    Print detailed evaluation metrics and dataset statistics
    """
    print("\nDataset Statistics:")
    print(f"Total validation samples: {len(eval_dataset)}")
    
    # Count labels in validation set
    label_counts = {}
    for item in eval_dataset:
        label = item['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"Human samples (label 0): {label_counts.get(0, 0)}")
    print(f"LLM samples (label 1): {label_counts.get(1, 0)}")

    print("\nModel Performance Metrics:")
    for key, value in metrics.items():
        # Remove 'eval_' prefix from metric names for cleaner output
        clean_key = key.replace('eval_', '')
        print(f"{clean_key}: {value:.4f}")

def main():
    try:
        model_path = './model_output_basic'
        if not os.path.exists(model_path):
            print(f"Error: No saved model found at {model_path}")
            print("Please train the model first using detect_llm.py")
            return

        print("\nLoading saved model for evaluation...")
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = model.to(device)
        print("Model loaded successfully!")

        print("\nLoading and preparing validation dataset...")
        texts, labels = load_ghostbuster_data()
        dataset = create_dataset(texts, labels)
        
        # Prepare validation dataset
        tokenized_val_dataset = dataset['validation'].map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=['text']
        )

        print("\nStarting model evaluation...")
        metrics = evaluate_model(model, tokenized_val_dataset, tokenizer)
        
        # Print detailed results
        print_detailed_metrics(metrics, dataset['validation'])

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 