//! Micro-Neural — MLP inference engine without external dependencies.
//!
//! Replaces `tract-onnx` for bots that need small WASM binaries.
//! Implements only what's needed: Linear layers + ReLU activation.
//!
//! # Binary format (.ludus-weights)
//! Instead of ONNX (which brings a 10MB runtime), we use a dead-simple format:
//!   [n_layers: u32]
//!   for each layer:
//!     [in: u32][out: u32]
//!     [weights: in*out f32 row-major]
//!     [biases:  out   f32]
//!
//! Export your PyTorch model with `ludus-nn/src/export_weights.py`.
//!
//! # Example
//! ```rust
//! const WEIGHTS: &[u8] = include_bytes!("../models/chess_eval.bin");
//!
/// pub struct NeuralBot;
//  impl LudusBot for NeuralBot {
//      fn next_move(state: &GameState) -> String {
//         ludus_sdk::micro_neural::best_move(WEIGHTS, state)
//             .unwrap_or_else(|| ludus_sdk::random_move(state))
//     }
// }
///

use crate::GameState;

// ─── Weight format parsing ────────────────────────────────────────────────────

struct Layer {
    in_size: usize,
    out_size:usize,
    weights: Vec<f32>, // [in_size * out_size], row-major
    biases: Vec<f32>,  // [out_size]
}

struct Mlp{
    layers: Vec<Layer>,
}

pub fn parse_f32(bytes: &[u8],offset:&mut usize) -> Option<f32> {
    let b = bytes.get(*offset..*offset+4)?;
    *offset += 4;
    Some(f32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

pub fn parse_u32(bytes: &[u8],offset:&mut usize) -> Option<u32> {
    let b = bytes.get(*offset..*offset+4)?;
    *offset += 4;
    Some(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

pub fn parse_weights(bytes: &[u8]) -> Option<Mlp> {
    let mut off = 0usize;
    let n_layers = parse_u32(bytes, &mut off)? as usize;
    let mut layers = Vec::with_capacity(n_layers);

    for _ in 0..n_layers{
        let in_size = parse_u32(bytes, &mut off)? as usize;
        let out_size = parse_u32(bytes,&mut off)? as usize;

        let mut weights = Vec::with_capacity(in_size * out_size);
        for _ in 0..in_size * out_size{
            weights.push(parse_f32(bytes, &mut off)?);
        }
        let mut biases = Vec::with_capacity(out_size);
        for _ in 0..out_size{
            biases.push(parse_f32(bytes, &mut off)?);
        }
        layers.push(Layer{in_size,out_size,weights,biases});
    }
    Some(Mlp{layers})
}


// ─── Model storage (WASM single-threaded safe) ───────────────────────────────

static mut LOADED_MLP: Option<Mlp> = None;

#[allow(static_mut_refs)]
fn get_mlp(bytes: &'static [u8]) -> Option<&'static Mlp> {
    /// SAFETY: wasm32-unknown-unknown is single-threaded, no concurrent access possible.
    unsafe {
        if LOADED_MLP.is_none(){
            LOADED_MLP = parse_weights(bytes);
        }
        LOADED_MLP.as_ref()
    }
}

// ─── Forward pass ────────────────────────────────────────────────────────────


fn forward(mlp: &Mlp, input: &[f32]) -> Vec<f32> {
    let mut x: Vec<f32> = input.to_vec();

    for (i, layer) in mlp.layers.iter().enumerate() {
        let mut out = vec![0.0f32; layer.out_size];
        // Matrix multiply: out[j] = sum_k(weights[k*out + j] * x[k]) + bias[j]
        for j in 0..layer.out_size {
            let mut s = layer.biases[j];
            for k in 0..layer.in_size {
                s += layer.weights[k * layer.out_size + j] * x[k];
            }
            // ReLU on all layers except the last
            out[j] = if i < mlp.layers.len() - 1 { s.max(0.0) } else { s };
        }
        x = out;
    }
    x
}

// ─── Encoding (mirrors encode.py / neural.rs) ────────────────────────────────

fn piece_to_value(piece: &str) -> f32 {
    match piece {
        "wK" =>  1.000, "bK" => -1.000,
        "wQ" =>  0.900, "bQ" => -0.900,
        "wR" =>  0.500, "bR" => -0.500,
        "wB" =>  0.333, "bB" => -0.333,
        "wN" =>  0.320, "bN" => -0.320,
        "wP" =>  0.100, "bP" => -0.100,
        _ => 0.0,
    }
}

fn board_flat(board: &[Vec<String>]) -> [f32; 64] {
    let mut out = [0.0f32; 64];
    for row in 0..8usize {
        for col in 0..8usize {
            out[row * 8 + col] = board
                .get(row).and_then(|r| r.get(col))
                .map(|s| piece_to_value(s.as_str()))
                .unwrap_or(0.0);
        }
    }
    out
}

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


// ─── Public API ──────────────────────────────────────────────────────────────

/// Score a single move. Returns `None` if weights fail to parse.
/// Higher score = better move for the current player.
pub fn score_move(weights: &'static [u8], board: &[Vec<String>], uci: &str) -> Option<f32> {
    let mlp = get_mlp(weights)?;
    let board_f = board_flat(board);
    let mv_f    = uci_enc(uci);
    let mut input = Vec::with_capacity(68);
    input.extend_from_slice(&board_f);
    input.extend_from_slice(&mv_f);
    forward(mlp, &input).into_iter().next()
}

/// Select the best move by scoring all legal moves.
/// Returns `None` if all inference calls fail (weights corrupt, etc.).
pub fn best_move(weights: &'static [u8], state: &GameState) -> Option<String> {
    if state.legal_moves.is_empty() { return None; }

    let mut best_score = f32::NEG_INFINITY;
    let mut best_mv: Option<&str> = None;

    for mv in &state.legal_moves {
        if let Some(score) = score_move(weights, &state.board, mv) {
            if score > best_score {
                best_score = score;
                best_mv = Some(mv.as_str());
            }
        }
    }
    best_mv.map(str::to_owned)
}

/// Rank all legal moves, best first.
pub fn rank_moves(weights: &'static [u8], state: &GameState) -> Vec<(String, f32)> {
    let mut ranked: Vec<(String, f32)> = state.legal_moves.iter()
        .filter_map(|mv| score_move(weights, &state.board, mv).map(|s| (mv.clone(), s)))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
    ranked
}
