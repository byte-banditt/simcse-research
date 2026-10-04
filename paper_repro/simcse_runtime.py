"""Modern-runtime port of the official unsupervised BERT SimCSE objective."""
import csv
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import BertModel


def contrastive_loss(projected, temperature=0.05):
    size = projected.shape[0]
    with torch.autocast(projected.device.type, enabled=False):
        logits = torch.nn.functional.cosine_similarity(projected[:, 0].float().unsqueeze(1),
                                                      projected[:, 1].float().unsqueeze(0), dim=-1) / temperature
        return torch.nn.functional.cross_entropy(logits, torch.arange(size, device=logits.device))


class UnsupervisedSimCSE(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path, local_files_only=True,
                                              add_pooling_layer=False, attn_implementation='eager')
        self.mlp = torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        torch.nn.init.normal_(self.mlp.weight, mean=0.0, std=self.bert.config.initializer_range)
        torch.nn.init.zeros_(self.mlp.bias)

    def loss(self, batch):
        size = batch['input_ids'].shape[0]
        doubled = {key: value.repeat_interleave(2, dim=0) for key, value in batch.items()}
        hidden = self.bert(**doubled).last_hidden_state[:, 0]
        projected = torch.tanh(self.mlp(hidden)).reshape(size, 2, -1)
        return contrastive_loss(projected)

    def embeddings(self, batch):
        # Unsupervised official --mlp_only_train: discard projection for evaluation.
        return self.bert(**batch).last_hidden_state[:, 0]


def load_stsb_dev(data_root):
    candidates = list(Path(data_root).rglob('sts-dev.csv'))
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one STS-B development file, found {len(candidates)}')
    rows = []
    with candidates[0].open(encoding='utf-8') as stream:
        for row in csv.reader(stream, delimiter='\t', quoting=csv.QUOTE_NONE):
            rows.append((row[5], row[6], float(row[4])))
    return candidates[0], rows


def evaluate_dev(model, tokenizer, rows, batch_size=64):
    previous = model.training
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            subset = rows[start:start + batch_size]
            # Match SentEval's whitespace token split and subsequent join.
            texts = [' '.join(row[0].split()) for row in subset] + [' '.join(row[1].split()) for row in subset]
            tokens = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda:0')
            vectors = torch.nn.functional.normalize(model.embeddings(tokens).float(), dim=-1)
            size = len(subset)
            scores.extend((vectors[:size] * vectors[size:]).sum(-1).cpu().tolist())
    model.train(previous)
    return {'spearman': float(spearmanr(scores, [row[2] for row in rows]).statistic),
            'pair_count': len(rows), 'similarity_scores': scores}
