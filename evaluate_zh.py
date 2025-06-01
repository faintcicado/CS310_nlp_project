import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import os
import json
from typing import List, Dict, Tuple
import pandas as pd

# Import functions from detect_llm_zh
from detect_llm_zh import (
    load_chinese_data,
    tokenize_function,
    compute_metrics
)

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def evaluate_model(model, eval_dataset, tokenizer):
    """
    Evaluate the model on the test dataset
    """
    eval_args = TrainingArguments(
        output_dir="./eval_results_zh",
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=16,
        dataloader_drop_last=False,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        compute_metrics=compute_metrics
    )

    metrics = trainer.evaluate(eval_dataset)
    return metrics

def print_detailed_metrics(data_type: str, metrics: Dict, dataset):
    """
    Print detailed evaluation metrics and dataset statistics
    """
    print(f"\n=== Evaluation Results for {data_type} ===")
    print("\nDataset Statistics:")
    print(f"Total test samples: {len(dataset)}")
    
    # Count labels
    label_counts = {}
    for item in dataset:
        label = item['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"Human samples: {label_counts.get(0, 0)}")
    print(f"LLM samples: {label_counts.get(1, 0)}")

    print("\nModel Performance Metrics:")
    for key, value in metrics.items():
        clean_key = key.replace('eval_', '')
        print(f"{clean_key}: {value:.4f}")

def evaluate_out_of_domain():
    """
    Evaluate model on out-of-domain test sets (webnovel and wiki)
    """
    try:
        model_path = './model_output_zh_chinese'
        if not os.path.exists(model_path):
            print(f"Error: No saved model found at {model_path}")
            print("Please train the model first using detect_llm_zh.py")
            return

        print("\nLoading saved Chinese BERT model for evaluation...")
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = model.to(device)
        print("Model loaded successfully!")

        # Store results for all domains
        all_results = []

        # Evaluate on webnovel and wiki datasets
        for data_type in ['webnovel', 'wiki']:
            print(f"\nEvaluating on {data_type} dataset...")
            
            # Load test data
            texts, labels = load_chinese_data(data_type=data_type)
            test_dataset = Dataset.from_dict({
                'text': texts,
                'label': labels
            })
            
            # Prepare test dataset
            tokenized_test_dataset = test_dataset.map(
                lambda x: tokenize_function(x, tokenizer),
                batched=True,
                remove_columns=['text']
            )

            # Run evaluation
            metrics = evaluate_model(model, tokenized_test_dataset, tokenizer)
            
            # Print results
            print_detailed_metrics(data_type, metrics, test_dataset)
            
            # Store results
            metrics['data_type'] = data_type
            all_results.append(metrics)

        # Create a summary DataFrame
        df_results = pd.DataFrame(all_results)
        print("\n=== Summary of Out-of-Domain Performance ===")
        print(df_results.to_string(index=False))

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    evaluate_out_of_domain() 