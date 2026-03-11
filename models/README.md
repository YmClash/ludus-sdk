# Ludus SDK — Neural Model Guide

## Format attendu du modèle ONNX

| | Valeur |
|---|---|
| **Input name** | `input` (ou premier input nommé) |
| **Input shape** | `[1, 68]` float32 |
| **Output shape** | `[1, 1]` float32 |

### Encodage des 68 features

```
[0..63]  : board state (64 floats) — voir board_to_tensor_flat()
[64]     : from_row  / 7.0   (0.0–1.0)
[65]     : from_col  / 7.0
[66]     : to_row    / 7.0
[67]     : to_col    / 7.0
```

### Valeurs des pièces (board encoding)

| Pièce | Valeur |
|---|---|
| wK | +1.000 |
| wQ | +0.900 |
| wR | +0.500 |
| wB | +0.333 |
| wN | +0.320 |
| wP | +0.100 |
| vide | 0.000 |
| bP | -0.100 |
| ... | ... (symétrique) |

---

## Entraîner un modèle (Python / PyTorch)

```python
import torch
import torch.nn as nn

class ChessEval(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(68, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)

# Export to ONNX
model = ChessEval()
dummy = torch.zeros(1, 68)
torch.onnx.export(
    model, dummy, "chess_eval.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
)
```

### Données d'entraînement

Le modèle doit être entraîné à prédire la qualité d'un coup :
- **Score positif** → bon coup pour le joueur courant
- **Score négatif** → mauvais coup

Source de données suggestions :
- Parties annotées (Stockfish labels)
- Self-play avec MCTS
- Dataset Lichess (PGN → extraire (board, move, outcome))

---

## Compiler le bot neural

```bash
cd ludus-sdk

# Debug (rapide, binaire plus grand)
cargo build --target wasm32-unknown-unknown --features neural

# Release (optimisé, pour upload Ludus)
cargo build --release --target wasm32-unknown-unknown --features neural
```

Le `.wasm` se trouve dans :
```
target/wasm32-unknown-unknown/release/ludus_sdk.wasm
```

> **Note** : Le binaire inclut le modèle ONNX (`include_bytes!`).
> Un modèle de 200KB → binaire WASM ~2-4MB. Dans la limite des 16MB. ✅
