import itertools

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer


def get_tokenizer(model):
    return AutoTokenizer.from_pretrained(model)


def tokenize(example, tokenizer, label2id):
    text = []
    # these are at the character level
    labels = []
    for t, l, ws in zip(
        example["tokens"], example["provided_labels"], example["trailing_whitespace"]
    ):
        text.append(t)
        labels.extend([l] * len(t))
        # if there is trailing whitespace
        if ws:
            text.append(" ")
            labels.append("O")
    tokenized = tokenizer("".join(text), return_offsets_mapping=True, truncation=False)
    labels = np.array(labels)
    text = "".join(text)
    token_labels = []
    for start_idx, end_idx in tokenized.offset_mapping:
        # CLS token
        if start_idx + end_idx == 0:
            token_labels.append(label2id["O"])
            continue
        # case when token starts with whitespace
        if text[start_idx].isspace():
            start_idx += 1
        while start_idx >= len(labels):
            start_idx -= 1
        token_labels.append(label2id[labels[start_idx]])
    length = len(tokenized.input_ids)
    return {**tokenized, "labels": token_labels, "length": length}


def get_labeling(json):
    all_labels = sorted(list(set(itertools.chain(*[x["labels"] for x in json]))))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {v: k for k, v in label2id.items()}
    return all_labels, label2id, id2label


def get_ds(json):
    ds = Dataset.from_dict(
        {
            "full_text": [x["full_text"] for x in json],
            "document": [x["document"] for x in json],
            "tokens": [x["tokens"] for x in json],
            "trailing_whitespace": [x["trailing_whitespace"] for x in json],
            "provided_labels": [x["labels"] for x in json],
        }
    )
    return ds


def check_tokenization(ds, tokenizer, id2label):
    # Confirm that alignment is good
    # run multiple times to see different rows
    x = ds.shuffle()[0]
    for t, l in zip(x["tokens"], x["provided_labels"]):
        if l != "O":
            print((t, l))
    print("*" * 100)
    for t, l in zip(tokenizer.convert_ids_to_tokens(x["input_ids"]), x["labels"]):
        if id2label[l] != "O":
            print((t, id2label[l]))
