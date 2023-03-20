import json
import random
from pathlib import Path

import typer
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    DataCollatorForSeq2Seq
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

ROOT_FOLDER = str(Path(__file__).parent.parent)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_BEAMS = 3
MAX_SEQ_LEN = 512
GENERATION_LEN = 512
LABEL_PAD_TOKEN_ID = -100


def compute_test_metrics(pred_file, dataset_file, model_dir, data_part):
    bleu, em = benchmark_evaluate(pred_file, dataset_file)
    result = {'bleu': bleu, 'match': em}
    with open(model_dir / f'scores_{data_part}.json', 'w') as f:
        json.dump(result, f, indent=2)


def preprocess_dataset(ds, tokenizer):
    def process(examples):
        return tokenizer(examples['nl'], max_length=MAX_SEQ_LEN, truncation=True)

    ds = ds.map(process, batched=True)
    return ds


def run_prediction(dataset, model, tokenizer, model_dir, batch_size, data_part, compute_metrics=True):
    model.eval()

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=LABEL_PAD_TOKEN_ID,
        # pad_to_multiple_of=8
    )

    dataset = preprocess_dataset(dataset, tokenizer)
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

    with open(model_dir / f'predictions_{data_part}.txt', 'w') as f:
        f.write('\n'.join(all_preds))

    if compute_metrics:
        dataset.to_json(model_dir / f'labels_{data_part}.jsonl')

        compute_test_metrics(
            dataset_file=str(model_dir / f'labels_{data_part}.jsonl'),
            pred_file=str(model_dir / f'predictions_{data_part}.txt'),
            model_dir=model_dir,
            data_part=data_part
        )


@prediction_app.command()
def run_test_prediction(
        data_part: str = typer.Option('validation', help='Data part to predict on'),
        model_dir: str = typer.Option(..., help='Path to model for inference'),
        batch_size: int = typer.Option(32, help='Batch size'),
):
    model_dir = Path(model_dir)
    path_to_model = str(model_dir / 'model')
    model = AutoModelForSeq2SeqLM.from_pretrained(path_to_model).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)

    ds = load_dataset('code_x_glue_tc_text_to_code', split=data_part)

    run_prediction(
        ds,
        model,
        tokenizer,
        model_dir,
        batch_size=batch_size,
        data_part=data_part,
        compute_metrics=True
    )


if __name__ == '__main__':
    prediction_app()