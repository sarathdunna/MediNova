"""CheXbert report labeler (Smit et al., EMNLP 2020), inference only (proposal 5.2, step 1).

Faithful port of stanfordmlgroup/CheXbert `src/label.py` for current torch/transformers:
same text cleaning, bert-base-uncased tokenization, 512-token cut-off, architecture
(BERT [CLS] -> 13 four-way heads + one two-way "No Finding" head) and output mapping.
Checked against the labelled sample reports shipped with the original repo
(tests/test_chexbert.py).

Weights: `chexbert.pth` from Stanford AIMI (huggingface.co/StanfordAIMI/RRG_scorers),
used under Stanford's research licence. Not committed; see data/README.md.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn

from .labels import CHEXPERT_LABELS

# Head order in the checkpoint (CheXbert src/constants.py CONDITIONS).
CHEXBERT_CONDITIONS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices", "No Finding",
]
# Class index -> CheXpert value: 0 blank, 1 positive, 2 negative, 3 uncertain.
_CLASS_TO_VALUE = np.array([np.nan, 1.0, 0.0, -1.0])


class _CheXbert(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import BertConfig, BertModel

        # Config only: every weight is overwritten by the checkpoint.
        self.bert = BertModel(BertConfig.from_pretrained("bert-base-uncased"), add_pooling_layer=True)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.linear_heads = nn.ModuleList([nn.Linear(hidden, 4) for _ in range(13)])
        self.linear_heads.append(nn.Linear(hidden, 2))

    def forward(self, ids, mask):
        cls = self.bert(ids, attention_mask=mask)[0][:, 0]
        return [head(cls) for head in self.linear_heads]


def clean_report(text: str) -> str:
    """CheXbert's get_impressions_from_csv cleaning."""
    text = (text or "").strip().replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


class CheXbertLabeler:
    def __init__(self, checkpoint: str, batch_size: int = 16):
        from transformers import BertTokenizer

        self.tok = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = _CheXbert()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
        state = {k.removeprefix("module."): v for k, v in state.items()}
        state = {k: v for k, v in state.items() if not k.endswith("position_ids")}
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.batch_size = batch_size

    def _encode(self, text: str) -> list[int]:
        tokens = self.tok.tokenize(clean_report(text))
        if not tokens:
            return [self.tok.cls_token_id, self.tok.sep_token_id]
        ids = [self.tok.cls_token_id, *self.tok.convert_tokens_to_ids(tokens), self.tok.sep_token_id]
        return ids if len(ids) <= 512 else ids[:511] + [self.tok.sep_token_id]

    @torch.no_grad()
    def predict_classes(self, texts: Iterable[str]) -> np.ndarray:
        """(N, 14) raw class indices in CHEXBERT_CONDITIONS order."""
        encoded = [self._encode(t) for t in texts]
        out = []
        for b in range(0, len(encoded), self.batch_size):
            chunk = encoded[b : b + self.batch_size]
            width = max(len(c) for c in chunk)
            ids = torch.zeros(len(chunk), width, dtype=torch.long)
            mask = torch.zeros(len(chunk), width)
            for i, c in enumerate(chunk):
                ids[i, : len(c)] = torch.tensor(c)
                mask[i, : len(c)] = 1
            logits = self.model(ids, mask)
            out.append(torch.stack([h.argmax(1) for h in logits], 1).numpy())
            print(f"\r[chexbert] {min(b + self.batch_size, len(encoded))}/{len(encoded)}", end="")
        print()
        return np.concatenate(out)

    def label(self, texts: pd.Series) -> pd.DataFrame:
        """CheXpert-encoded labels (1 / 0 / -1 / NaN) in CHEXPERT_LABELS column order."""
        cls = self.predict_classes(texts.tolist())
        values = _CLASS_TO_VALUE[cls]
        # "No Finding" head is two-way: 0 blank, 1 positive.
        values[:, -1] = np.where(cls[:, -1] == 1, 1.0, np.nan)
        df = pd.DataFrame(values, index=texts.index, columns=CHEXBERT_CONDITIONS)
        return df[CHEXPERT_LABELS]
