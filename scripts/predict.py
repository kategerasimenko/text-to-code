import os
import json
import random

import typer
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    DataCollatorForSeq2Seq
)

from config import (
    DEVICE, LABEL_PAD_TOKEN_ID,
    GENERATION_LEN, N_BEAMS
)
from evaluator.evaluator import (
    evaluate as benchmark_evaluate
)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

prediction_app = typer.Typer(add_completion=False)


def compute_test_metrics(pred_file, dataset_file, model_dir, data_part):
    """
    Evaluate preds with benchmark and save scores to file.
    """
    bleu, em = benchmark_evaluate(pred_file, dataset_file)
    result = {'bleu': bleu, 'match': em}
    with open(os.path.join(model_dir, f'scores_{data_part}.json'), 'w') as f:
        json.dump(result, f, indent=2)


def preprocess_dataset(ds, tokenizer, max_seq_len):
    """
    Tokenize input for prediction.
    """
    def process(examples):
        return tokenizer(examples['nl'], max_length=max_seq_len, truncation=True)

    ds = ds.map(process, batched=True)
    return ds


def run_prediction(dataset, model, tokenizer, model_dir, batch_size, data_part, max_seq_len, compute_metrics=True):
    """
    Run prediction (generation), save predictions, and calculate metrics.
    """
    model.eval()

    # data collator which pads inputs on the go based on max length inside the batch
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=LABEL_PAD_TOKEN_ID
    )

    dataset = preprocess_dataset(dataset, tokenizer, max_seq_len)
    tokens = dataset.remove_columns(['nl', 'id', 'code'])  # a bit of hardcoding
    test_dataloader = DataLoader(tokens, batch_size=batch_size, collate_fn=collator)

    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            preds = model.generate(**batch, num_beams=N_BEAMS, max_length=GENERATION_LEN)
            decoded_preds = tokenizer.batch_decode(
                preds,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            all_preds.extend(decoded_preds)

    with open(os.path.join(model_dir, f'predictions_{data_part}.txt'), 'w') as f:
        f.write('\n'.join(all_preds))

    if compute_metrics:
        dataset.to_json(os.path.join(model_dir, f'labels_{data_part}.jsonl'))

        compute_test_metrics(
            dataset_file=os.path.join(model_dir, f'labels_{data_part}.jsonl'),
            pred_file=os.path.join(model_dir, f'predictions_{data_part}.txt'),
            model_dir=model_dir,
            data_part=data_part
        )
