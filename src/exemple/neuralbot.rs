//! NeuralBot — ONNX-powered chess bot example.
//!
//! # Prerequisites
//! 1. Train a move-evaluation model (see `models/README.md`).
//! 2. Copy it to `ludus-sdk/models/chess_eval.onnx`.
//! 3. Build with the `neural` feature:
//!    ```bash
//!    cargo build --release \
//!        --target wasm32-unknown-unknown \
//!        --features neural
//!    ```
//! 4. Upload `target/wasm32-unknown-unknown/release/ludus_sdk.wasm` to your profile.
//!
//! # How it works
//! For each call to `next_move`, the bot:
//! 1. Encodes the board as 64 floats + each legal move as 4 floats → [1, 68] tensor.
//! 2. Runs the ONNX model to get a quality score per move.
//! 3. Returns the move with the highest score.
//! 4. Falls back to `random_move` if inference fails.

use ludus_sdk::{export_bot, neural, GameState, LudusBot};

/// The ONNX model, embedded at compile time.
/// Accepts [1, 68] float32 input, produces [1, 1] float32 output.
///
/// To use a different model, change the path here.
const MODEL: &[u8] = include_bytes!("../models/chess_eval.onnx");

pub struct NeuralBot;

impl LudusBot for NeuralBot {
    fn next_move(state: &GameState) -> String {
        // Rank all legal moves with the neural network.
        // Falls back to pseudo-random if the model fails.
        neural::best_move(MODEL, state).unwrap_or_else(|| ludus_sdk::random_move(state))
    }
}

export_bot!(NeuralBot);

// ─── Optional: Debug / Inspection ────────────────────────────────────────────
// Uncomment to enable a ranked-moves function for testing your model
// (not exported to WASM — for local use only).
//
// #[allow(dead_code)]
// fn debug_rank(state: &GameState) {
//     for (mv, score) in neural::rank_moves(MODEL, state) {
//         println!("{}: {:.4}", mv, score);
//     }
// }
