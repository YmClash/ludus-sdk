//! Neural Intelligence — ONNX-powered move evaluation for Ludus bots.
//!
//! Enabled with `features = ["neural"]`. Requires `tract-onnx`.
//!
//! # Model Contract
//! The ONNX model must accept:
//! - **Input** : `[1, 68]` float32 — board (64 floats) + move (4 floats)
//! - **Output**: `[1, 1]`  float32 — quality score (higher = better for current player)
//!
//! # Encoding
//! - Board: flat 8×8, row 0 = rank 1, col 0 = file a. Values in **[-1.0, 1.0]**.
//! - Move : `[from_row, from_col, to_row, to_col]`, each in **[0.0, 1.0]** (÷7).
//!
//! # Quickstart
/*
//! ```rust
//! use ludus_sdk::{export_bot, neural, GameState, LudusBot};
//!
//! const MODEL: &[u8] = include_bytes!("../models/chess_eval.onnx");
//!
//! pub struct NeuralBot;
//! impl LudusBot for NeuralBot {
//!     fn next_move(state: &GameState) -> String {
//!         neural::best_move(MODEL, state)
//!             .unwrap_or_else(|| ludus_sdk::random_move(state))
//!     }
//! }
//! export_bot!(NeuralBot);
//! ```
 */

use crate::GameState;
use tract_onnx::prelude::*;

// ─── Internal model type ──────────────────────────────────────────────────────

type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

// WASM is single-threaded — static mut is safe here (no concurrent access).
// The model is compiled once on first call and reused for all subsequent calls.
static mut COMPILED_MODEL: Option<Model> = None;

fn compile_model(bytes: &[u8]) -> Option<Model> {
    tract_onnx::onnx()
        .model_for_read(&mut std::io::Cursor::new(bytes))
        .ok()?
        .into_optimized()
        .ok()?
        .into_runnable()
        .ok()
}

/// Lazy-init: compile on first call, reuse on subsequent calls.
///
/// # Safety
/// Safe in single-threaded WASM. Do not use in multi-threaded contexts.
fn get_model(bytes: &'static [u8]) -> Option<&'static Model> {
    unsafe {
        if COMPILED_MODEL.is_none() {
            COMPILED_MODEL = compile_model(bytes);
        }
        COMPILED_MODEL.as_ref()
    }
}

// ─── Piece encoding ───────────────────────────────────────────────────────────

/// Encode a piece string to a normalized float in [-1.0, 1.0].
///
/// White pieces are positive, black pieces negative, empty = 0.0.
/// Values are proportional to material value (King=±1.0, Pawn=±0.10).
pub fn piece_to_value(piece: &str) -> f32 {
    match piece {
        "wK" => 1.000,
        "bK" => -1.000,
        "wQ" => 0.900,
        "bQ" => -0.900,
        "wR" => 0.500,
        "bR" => -0.500,
        "wB" => 0.333,
        "bB" => -0.333,
        "wN" => 0.320,
        "bN" => -0.320,
        "wP" => 0.100,
        "bP" => -0.100,
        _ => 0.0,
    }
}

// ─── Board → Tensor conversions ──────────────────────────────────────────────

/// **Flat** board encoding: `[64]` float32, row-major.
///
/// Row 0 = rank 1 (white back rank), col 0 = file a, col 7 = file h.
/// Values in `[-1.0, 1.0]` via [`piece_to_value`].
pub fn board_to_tensor_flat(board: &[Vec<String>]) -> Vec<f32> {
    let mut out = vec![0.0f32; 64];
    for row in 0..8usize {
        for col in 0..8usize {
            out[row * 8 + col] = board
                .get(row)
                .and_then(|r| r.get(col))
                .map(|s| piece_to_value(s.as_str()))
                .unwrap_or(0.0);
        }
    }
    out
}

/// **AlphaZero-style** 12-plane encoding: `[768]` float32.
///
/// 12 binary planes (one per piece type/color), each 8×8.
/// Plane order: `wK wQ wR wB wN wP bK bQ bR bB bN bP`.
/// Value: `1.0` if that piece is on that square, else `0.0`.
pub fn board_to_tensor_planes(board: &[Vec<String>]) -> Vec<f32> {
    const PIECES: [&str; 12] = [
        "wK", "wQ", "wR", "wB", "wN", "wP", "bK", "bQ", "bR", "bB", "bN", "bP",
    ];
    let mut out = vec![0.0f32; 768];
    for (plane, &target) in PIECES.iter().enumerate() {
        for row in 0..8usize {
            for col in 0..8usize {
                let cell = board
                    .get(row)
                    .and_then(|r| r.get(col))
                    .map(|s| s.as_str())
                    .unwrap_or("");
                if cell == target {
                    out[plane * 64 + row * 8 + col] = 1.0;
                }
            }
        }
    }
    out
}

/// Encode a UCI move to `[from_row, from_col, to_row, to_col]`, each in `[0.0, 1.0]`.
///
/// Returns `[0.0; 4]` for invalid input.
pub fn uci_to_tensor(uci: &str) -> [f32; 4] {
    let b = uci.as_bytes();
    if b.len() < 4 {
        return [0.0; 4];
    }
    [
        b[1].wrapping_sub(b'1') as f32 / 7.0, // from_row
        b[0].wrapping_sub(b'a') as f32 / 7.0, // from_col
        b[3].wrapping_sub(b'1') as f32 / 7.0, // to_row
        b[2].wrapping_sub(b'a') as f32 / 7.0, // to_col
    ]
}

// ─── Inference API ───────────────────────────────────────────────────────────

/// Run the ONNX model with a raw float32 input vector.
///
/// `model_bytes` must be `'static` (e.g. from `include_bytes!`).
/// `input_shape` is e.g. `&[1, 68]`.
///
/// Returns the first output scalar, or `None` on any error.
pub fn run_model_raw(
    model_bytes: &'static [u8],
    input: Vec<f32>,
    input_shape: &[usize],
) -> Option<f32> {
    let model = get_model(model_bytes)?;

    // Validate size
    let expected: usize = input_shape.iter().product();
    if input.len() != expected {
        return None;
    }

    // Build tensor
    let arr = tract_onnx::tract_ndarray::Array::from_shape_vec(
        tract_onnx::tract_ndarray::IxDyn(input_shape),
        input,
    )
        .ok()?;
    let tensor: Tensor = arr.into();

    // Run inference
    let outputs = model.run(tvec![tensor.into()]).ok()?;

    // Extract scalar result
    outputs[0].to_scalar::<f32>().ok().copied()
}

/// Score a single UCI move using a **move-evaluation model** `[1, 68] → [1, 1]`.
///
/// Input = board_flat (64 floats) + uci_move encoding (4 floats).
/// Higher score = better move for the current player.
pub fn score_move(
    model_bytes: &'static [u8],
    board: &[Vec<String>],
    uci_move: &str,
) -> Option<f32> {
    let mut input = board_to_tensor_flat(board);
    input.extend_from_slice(&uci_to_tensor(uci_move));
    run_model_raw(model_bytes, input, &[1, 68])
}
/*
/// Select the **best move** by scoring all legal moves with the ONNX model.
///
/// Iterates `state.legal_moves`, calls [`score_move`] for each, returns the
/// move with the highest score. Returns `None` if inference fails for all moves.
///
/// # Fallback pattern
/// ```rust
///neural::best_move(MODEL, state)
///     .unwrap_or_else(|| ludus_sdk::random_move(state))
/// ```

*/
pub fn best_move(model_bytes: &'static [u8], state: &GameState) -> Option<String> {
    if state.legal_moves.is_empty() {
        return None;
    }

    let mut best_score = f32::NEG_INFINITY;
    let mut best_mv: Option<&str> = None;

    for mv in &state.legal_moves {
        if let Some(score) = score_move(model_bytes, &state.board, mv) {
            if score > best_score {
                best_score = score;
                best_mv = Some(mv.as_str());
            }
        }
    }

    best_mv.map(str::to_owned)
}

/// Score every legal move and return them sorted (best first).
///
/// Useful for debugging or implementing beam search / top-k selection.
/// Moves that fail inference are excluded from the result.
pub fn rank_moves(model_bytes: &'static [u8], state: &GameState) -> Vec<(String, f32)> {
    let mut ranked: Vec<(String, f32)> = state
        .legal_moves
        .iter()
        .filter_map(|mv| score_move(model_bytes, &state.board, mv).map(|s| (mv.clone(), s)))
        .collect();

    // Sort descending by score (best move first)
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}
