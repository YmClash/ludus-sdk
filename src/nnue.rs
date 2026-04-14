//! NNUE — Incrementally Updatable Neural Network evaluation.
//!
//! A faster alternative to `micro_neural` for chess-specific evaluation.
//! Uses 768 binary piece-square features processed through a FeatureTransformer
//! (FT), whose output (accumulator) is shared across all legal moves in a
//! position. Only the lightweight "head" network runs per-move.
//!
//! # Why NNUE is faster than classic MLP
//! Classic MLP:  N_legal_moves × 68 × H  ops  (full board re-encoded per move)
//! NNUE:         768→H once + N_legal × (H+4) × 32  (board encoded once)
//! → ~4× faster for 30 legal moves with H=64.
//!
//! # Binary weight format (.ludus-nnue)
//! ```text
//! [magic: b"NNUE"  4 bytes]
//! [H: u32                 ]  e.g. 64
//! [ft_weights: 768*H f32  ]  row-major [feature × H]
//! [ft_biases:  H    f32   ]
//! [n_head_layers: u32     ]
//! for each head layer:
//!   [in:  u32]
//!   [out: u32]
//!   [weights: in*out f32  ]
//!   [biases:  out    f32  ]
//! ```
//!
//! # Quickstart
//! ```rust
//! const NNUE: &[u8] = include_bytes!("../models/chess_eval.ludus-nnue");
//
// pub struct NnueBot;
// impl LudusBot for NnueBot {
//     fn next_move(state: &GameState) -> String {
//         nnue::best_move(NNUE, state)
//             .unwrap_or_else(|| ludus_sdk::random_move(state))
//     }
// }
// export_nnue_bot!(NnueBot);
// ```

use crate::GameState;

// ─── Architecture constant ────────────────────────────────────────────────────

/// 12 piece types × 64 squares = 768 binary input features.
const N_FEATURES: usize = 768;

// ─── Feature encoding ─────────────────────────────────────────────────────────

/// Piece type to feature plane index (0-11).
/// Order: wK wQ wR wB wN wP bK bQ bR bB bN bP
fn piece_plane(piece: &str) -> Option<usize> {
    match piece {
        "wK" => Some(0),  "wQ" => Some(1),  "wR" => Some(2),
        "wB" => Some(3),  "wN" => Some(4),  "wP" => Some(5),
        "bK" => Some(6),  "bQ" => Some(7),  "bR" => Some(8),
        "bB" => Some(9),  "bN" => Some(10), "bP" => Some(11),
        _ => None, // empty square
    }
}

/// Global feature index = piece_plane * 64 + square_idx (row*8 + col).
fn feature_idx(piece: &str, row: usize, col: usize) -> Option<usize> {
    piece_plane(piece).map(|p| p * 64 + row * 8 + col)
}

/// Encode a UCI move to [from_row, from_col, to_row, to_col] each in [0, 1].
fn uci_enc(uci: &str) -> [f32; 4] {
    let b = uci.as_bytes();
    if b.len() < 4 { return [0.0; 4]; }
    [
        b[1].wrapping_sub(b'1') as f32 / 7.0,
        b[0].wrapping_sub(b'a') as f32 / 7.0,
        b[3].wrapping_sub(b'1') as f32 / 7.0,
        b[2].wrapping_sub(b'a') as f32 / 7.0,
    ]
}

// ─── Weight format parsing ────────────────────────────────────────────────────

fn parse_u32(bytes: &[u8], off: &mut usize) -> Option<u32> {
    let s = bytes.get(*off..*off + 4)?;
    *off += 4;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn parse_f32(bytes: &[u8], off: &mut usize) -> Option<f32> {
    let s = bytes.get(*off..*off + 4)?;
    *off += 4;
    Some(f32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn parse_f32_vec(bytes: &[u8], off: &mut usize, n: usize) -> Option<Vec<f32>> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n { v.push(parse_f32(bytes, off)?); }
    Some(v)
}

// ─── Model ───────────────────────────────────────────────────────────────────

struct HeadLayer {
    in_size:  usize,
    out_size: usize,
    weights:  Vec<f32>, // [in × out] row-major
    biases:   Vec<f32>, // [out]
}

pub struct NnueModel {
    h:          usize,
    ft_weights: Vec<f32>, // [N_FEATURES × h]
    ft_biases:  Vec<f32>, // [h]
    head:       Vec<HeadLayer>,
}

impl NnueModel {
    /// Parse a `.ludus-nnue` binary blob.
    fn parse(bytes: &[u8]) -> Option<Self> {
        // Magic check
        if bytes.get(0..4) != Some(b"NNUE") {
            return None;
        }
        let mut off = 4;

        let h = parse_u32(bytes, &mut off)? as usize;
        let ft_weights = parse_f32_vec(bytes, &mut off, N_FEATURES * h)?;
        let ft_biases  = parse_f32_vec(bytes, &mut off, h)?;

        let n_head = parse_u32(bytes, &mut off)? as usize;
        let mut head = Vec::with_capacity(n_head);
        for _ in 0..n_head {
            let in_size  = parse_u32(bytes, &mut off)? as usize;
            let out_size = parse_u32(bytes, &mut off)? as usize;
            let weights  = parse_f32_vec(bytes, &mut off, in_size * out_size)?;
            let biases   = parse_f32_vec(bytes, &mut off, out_size)?;
            head.push(HeadLayer { in_size, out_size, weights, biases });
        }

        Some(NnueModel { h, ft_weights, ft_biases, head })
    }

    // ── FeatureTransformer ────────────────────────────────────────────────────

    /// Compute the full accumulator from a board (O(N_pieces × h)).
    ///
    /// Starts from biases, adds each active piece's feature row.
    /// Applies ClippedReLU ∈ [0, 1] at the end.
    pub fn init_accumulator(&self, board: &[Vec<String>]) -> Vec<f32> {
        let mut acc = self.ft_biases.clone();
        for row in 0..8_usize {
            for col in 0..8_usize {
                let piece = board
                    .get(row).and_then(|r| r.get(col))
                    .map(|s| s.as_str()).unwrap_or("");
                if let Some(fidx) = feature_idx(piece, row, col) {
                    let base = fidx * self.h;
                    for h in 0..self.h {
                        acc[h] += self.ft_weights[base + h];
                    }
                }
            }
        }
        // ClippedReLU ∈ [0, 1]
        for a in &mut acc { *a = a.clamp(0.0, 1.0); }
        acc
    }

    /// Incremental update: remove `old_piece` and add `new_piece` at `square`.
    ///
    /// Call this when a piece moves FROM a square (old_piece → "") or
    /// TO a square ("" → new_piece), or is promoted/captured.
    /// O(h) — much cheaper than a full `init_accumulator`.
    pub fn update_accumulator(
        &self,
        acc:       &mut Vec<f32>,
        old_piece: &str,
        new_piece: &str,
        row:       usize,
        col:       usize,
    ) {
        // Subtract old feature
        if let Some(fidx) = feature_idx(old_piece, row, col) {
            let base = fidx * self.h;
            for h in 0..self.h { acc[h] -= self.ft_weights[base + h]; }
        }
        // Add new feature
        if let Some(fidx) = feature_idx(new_piece, row, col) {
            let base = fidx * self.h;
            for h in 0..self.h { acc[h] += self.ft_weights[base + h]; }
        }
        // Re-clamp
        for a in acc.iter_mut() { *a = a.clamp(0.0, 1.0); }
    }

    // ── Head network ──────────────────────────────────────────────────────────

    /// Score a single move using a pre-computed accumulator.
    ///
    /// Input = [acc (h floats)] ++ [uci_enc (4 floats)]
    /// Runs the head MLP and returns the scalar score.
    pub fn score_move_with_acc(&self, acc: &[f32], uci: &str) -> Option<f32> {
        let move_enc = uci_enc(uci);
        let mut x: Vec<f32> = acc.to_vec();
        x.extend_from_slice(&move_enc);

        for (i, layer) in self.head.iter().enumerate() {
            let mut out = vec![0.0f32; layer.out_size];
            for j in 0..layer.out_size {
                let mut s = layer.biases[j];
                for k in 0..layer.in_size {
                    s += layer.weights[k * layer.out_size + j] * x[k];
                }
                // ReLU on hidden layers, linear on last
                out[j] = if i < self.head.len() - 1 { s.max(0.0) } else { s };
            }
            x = out;
        }
        x.into_iter().next()
    }
}

// ─── Static model storage (WASM single-threaded safe) ────────────────────────

static mut LOADED_NNUE: Option<NnueModel> = None;

/// Lazy-load the NNUE model. Persists across moves with warm sessions (Phase 1).
#[allow(static_mut_refs)]
fn get_nnue(bytes: &'static [u8]) -> Option<&'static NnueModel> {
    // SAFETY: wasm32-unknown-unknown is single-threaded.
    unsafe {
        if LOADED_NNUE.is_none() {
            LOADED_NNUE = NnueModel::parse(bytes);
        }
        LOADED_NNUE.as_ref()
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Score a single UCI move with the NNUE model.
///
/// Computes the full accumulator from `board`, then runs the head for `uci`.
/// For scoring many moves in the same position, prefer [`best_move`] which
/// computes the accumulator only once.
pub fn score_move(
    nnue_bytes: &'static [u8],
    board:      &[Vec<String>],
    uci:        &str,
) -> Option<f32> {
    let model = get_nnue(nnue_bytes)?;
    let acc   = model.init_accumulator(board);
    model.score_move_with_acc(&acc, uci)
}

/// Select the best move by scoring all legal moves with the NNUE model.
///
/// The FeatureTransformer accumulator is computed **once** for the current
/// board, then the head runs **once per legal move** — significantly faster
/// than re-encoding the full board for each candidate.
///
/// Returns `None` if the model fails to load or all moves fail inference.
pub fn best_move(nnue_bytes: &'static [u8], state: &GameState) -> Option<String> {
    if state.legal_moves.is_empty() { return None; }

    let model = get_nnue(nnue_bytes)?;

    // Compute accumulator ONCE for this position
    let acc = model.init_accumulator(&state.board);

    let mut best_score = f32::NEG_INFINITY;
    let mut best_mv: Option<&str> = None;

    // Score each legal move using the cached accumulator
    for mv in &state.legal_moves {
        if let Some(score) = model.score_move_with_acc(&acc, mv) {
            if score > best_score {
                best_score = score;
                best_mv    = Some(mv.as_str());
            }
        }
    }

    best_mv.map(str::to_owned)
}

/// Rank all legal moves, best first. Useful for debugging or beam search.
pub fn rank_moves(nnue_bytes: &'static [u8], state: &GameState) -> Vec<(String, f32)> {
    let Some(model) = get_nnue(nnue_bytes) else { return Vec::new() };
    let acc = model.init_accumulator(&state.board);

    let mut ranked: Vec<(String, f32)> = state.legal_moves.iter()
        .filter_map(|mv| model.score_move_with_acc(&acc, mv).map(|s| (mv.clone(), s)))
        .collect();

    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
    ranked
}
