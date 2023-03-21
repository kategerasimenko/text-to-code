import os
import json
import random
from pathlib import Path

import numpy as np
import torch
import typer
import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,
    Seq2SeqTrainer, EarlyStoppingCallback
)

from predict import run_prediction
from evaluator.evaluator import (
    evaluate as benchmark_evaluate
)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

app = typer.Typer(add_completion=False)

ROOT_FOLDER = str(Path(__file__).parent.parent)

MAX_SEQ_LEN = 512
LABEL_PAD_TOKEN_ID = -100
GENERATION_LEN = MAX_SEQ_LEN
N_BEAMS = 3

DEV_METRIC = evaluate.load("sacrebleu")
IS_CUDA_AVAILABLE = torch.cuda.is_available()


def preprocess_dataset(ds, tokenizer):
    def process(examples):
        model_inputs = tokenizer(examples['nl'], max_length=MAX_SEQ_LEN, truncation=True)
        labels = tokenizer(examples['code'], max_length=MAX_SEQ_LEN, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    ds = ds.map(process, batched=True)
    return ds


def postprocess_text(preds, labels):
    preds = [pred.replace('\n', ' ').strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_gen_metrics(eval_preds, tokenizer, model_ckpt_dir):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(
        preds,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # these were metrics inherent to HF Trainer setup, but the benchmark has different evaluation code
    # result = {
    #     'bleu': DEV_METRIC.compute(predictions=decoded_preds, references=decoded_labels)['score'],
    #     'match': sum(p == ls[0] for p, ls in zip(decoded_preds, decoded_labels)) / len(decoded_labels)
    # }

    label_file = os.path.join(model_ckpt_dir, 'temp_gold.jsonl')
    pred_file = os.path.join(model_ckpt_dir, 'preds.txt')

    with open(label_file, 'w') as f:
        f.write('\n'.join(json.dumps({'code': label[0]}) for label in decoded_labels))

    with open(pred_file, 'w') as f:
        f.write('\n'.join(decoded_preds))

    bleu, em = benchmark_evaluate(pred_file, label_file)
    result = {'bleu': bleu, 'match': em}

    return result


@app.command()
def main(
        base_model: str = typer.Option('t5-small', help='ModelHub pretrained model to finetune'),
        learning_rate: float = typer.Option(1e-4, help='Learning Rate'),
        max_epochs: int = typer.Option(20, help='Number of Epochs'),
        batch_size: int = typer.Option(32, help='Batch size')
):
    model_name = f'T2C_{base_model.rsplit("/", 1)[-1]}_{learning_rate}lr_{max_epochs}epochs'
    model_save_dir = os.path.join(ROOT_FOLDER, 'models', model_name)
    model_ckpt_dir = os.path.join(ROOT_FOLDER, 'checkpoints', model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if base_model.startswith('t5-'):
        tokenizer.add_tokens(['\x00', '~', '^', '}', '<', '`', '{'], special_tokens=False)
        new_num_tokens = len(tokenizer)
        model.resize_token_embeddings(new_num_tokens)

    ds = load_dataset('code_x_glue_tc_text_to_code')

    ds_for_train = preprocess_dataset(ds, tokenizer)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=LABEL_PAD_TOKEN_ID
    )

    def compute_dev_metrics(eval_preds):
        return compute_gen_metrics(eval_preds, tokenizer, model_ckpt_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir=model_ckpt_dir,
        report_to='none',
        evaluation_strategy='epoch',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=max_epochs,
        save_strategy='epoch',
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=GENERATION_LEN,
        generation_num_beams=N_BEAMS,
        metric_for_best_model='eval_bleu',
        greater_is_better=True,
        load_best_model_at_end=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_for_train['train'],
        eval_dataset=ds_for_train['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_dev_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5)
        ]
    )

    trainer.train()
    trainer.save_model(os.path.join(model_save_dir, 'model'))

    test_part = 'validation'
    run_prediction(
        dataset=ds[test_part],
        model=trainer.model,
        tokenizer=tokenizer,
        model_dir=model_save_dir,
        batch_size=batch_size,
        data_part=test_part,
        compute_metrics=True
    )


if __name__ == '__main__':
    app()

