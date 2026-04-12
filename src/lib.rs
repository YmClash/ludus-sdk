// ─── WASM: getrandom stub ────────────────────────────────────────────────────
// tract-onnx pulls `rand` / `getrandom` as transitive deps.
// On `wasm32-unknown-unknown` (wasmtime, no browser JS), getrandom has no
// default backend and fails to compile. We register a zeroing stub — neural
// inference itself never calls getrandom, so this is perfectly safe.
#[cfg(target_arch = "wasm32")]
mod wasm_getrandom {
    fn stub(buf: &mut [u8]) -> Result<(), getrandom::Error> {
        for b in buf.iter_mut() { *b = 0; }
        Ok(())
    }
    getrandom::register_custom_getrandom!(stub);
}


/// Ludus Bot SDK
///
/// # Write a bot
/// 1. Implement the [`LudusBot`] trait
/// 2. Use [`export_bot!`] to export your bot
/// 3. Compile: `cargo build --release --target wasm32-unknown-unknown`
/// 4. Upload the `.wasm` file to your Ludus profile
///
/// # Minimal example
/// ```rust
/// use ludus_sdk::{LudusBot, GameState, export_bot};
///
/// pub struct MyBot;
/// impl LudusBot for MyBot {
///     fn next_move(state: &GameState) -> String {
///         ludus_sdk::random_move(state)
///     }
/// }
/// export_bot!(MyBot);
/// ```
///
/// # Neural bot (requires `features = ["neural"]`)
/// ```rust
/// use ludus_sdk::{LudusBot, GameState, neural, export_bot};
///
/// const MODEL: &[u8] = include_bytes!("../models/chess_eval.onnx");
///
/// pub struct NeuralBot;
/// impl LudusBot for NeuralBot {
///     fn next_move(state: &GameState) -> String {
///         neural::best_move(MODEL, state)
///             .unwrap_or_else(|| ludus_sdk::random_move(state))
///     }
/// }
/// export_bot!(NeuralBot);
/// ```
use serde::{Deserialize, Serialize};

// ─── Neural module (feature-gated, tract-onnx, ~15MB WASM) ──────────────────
#[cfg(feature = "neural")]
pub mod neural;

// ─── Micro-neural (no deps, ~100KB WASM) ─────────────────────────────────────
// Recommended for most bots. Uses a custom binary weight format (.bin).
// Export your model with: python ludus-nn/src/export_weights.py
pub mod micro_neural;

// ─── NNUE (768 binary features, faster than micro_neural on large legal-move sets) ───
// No external deps. Export model with: python ludus-nn/src/export_nnue.py
pub mod nnue;
// ─── Game State ──────────────────────────────────────────────────────────────

/// The game state your bot receives, serialized as JSON by the Ludus server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    /// Whose turn: `"white"` / `"black"` (Chess, Checkers) or `"0"` / `"1"` (Ludo).
    pub turn: String,

    /// Game status: `"InProgress"`, `"Check"`, `"Checkmate"`, `"Draw"`, etc.
    pub status: String,

    /// Legal moves in UCI notation (`"e2e4"`, `"e7e8q"`, `"12-16"`, …).
    /// **Your bot must return one of these strings.**
    pub legal_moves: Vec<String>,

    /// Move history (algebraic notation, oldest first).
    pub move_history: Vec<String>,

    /// Board state as a 2D array `[row][col]`.
    ///
    /// **Chess**: `"wK"`, `"bP"`, … — empty = `""`.
    /// Row 0 = rank 1 (white back rank), col 0 = file a.
    ///
    /// **Checkers**: `"r"` / `"b"` / `"R"` (king) / `"B"` (king) / `""`.
    ///
    /// **Ludo**: `[[red tokens], [blue tokens], [current_player, dice]]`.
    #[serde(default)]
    pub board: Vec<Vec<String>>,
}

impl GameState {
    /// Returns `true` if the game is over (no legal moves).
    pub fn is_terminal(&self) -> bool {
        self.legal_moves.is_empty()
    }

    /// Returns the piece at a chess square (`"e2"` → `"wP"`, `""` if empty).
    pub fn piece_at(&self, square: &str) -> &str {
        let b = square.as_bytes();
        if b.len() < 2 {
            return "";
        }
        let col = b[0].wrapping_sub(b'a') as usize;
        let row = b[1].wrapping_sub(b'1') as usize;
        self.board
            .get(row)
            .and_then(|r| r.get(col))
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    /// Color prefix of the piece at a chess square: `"w"`, `"b"`, or `""`.
    pub fn piece_color_at(&self, square: &str) -> &str {
        let p = self.piece_at(square);
        if p.starts_with('w') {
            "w"
        } else if p.starts_with('b') {
            "b"
        } else {
            ""
        }
    }
}

// ─── Bot Trait ───────────────────────────────────────────────────────────────

/// Implement this trait to define your bot's strategy.
///
/// `next_move` receives the full game state and must return one of the
/// strings in `state.legal_moves`.
pub trait LudusBot {
    fn next_move(state: &GameState) -> String;
}

// ─── Built-in helpers ────────────────────────────────────────────────────────

/// Pick a pseudo-random legal move (recommended default).
///
/// Uses a LCG seeded from game history — avoids the threefold-repetition
/// Draw that always occurs when picking `legal_moves[0]`.
pub fn random_move(state: &GameState) -> String {
    if state.legal_moves.is_empty() {
        return String::new();
    }
    if state.legal_moves.len() == 1 {
        return state.legal_moves[0].clone();
    }
    let idx = (lcg_seed(state) as usize) % state.legal_moves.len();
    state.legal_moves[idx].clone()
}

/// Always pick the first legal move. **Causes Draw by repetition** in
/// symmetric positions — prefer [`random_move`].
pub fn first_move(state: &GameState) -> String {
    state.legal_moves.first().cloned().unwrap_or_default()
}

/// LCG-based seed derived from game history (no OS randomness, WASM-safe).
fn lcg_seed(state: &GameState) -> u64 {
    let mut s: u64 = 0x9e3779b97f4a7c15;
    s = s.wrapping_add(state.move_history.len() as u64);
    s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    for mv in state.move_history.iter().rev().take(4) {
        for b in mv.bytes() {
            s ^= b as u64;
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
        }
    }
    s ^= state.legal_moves.len() as u64;
    s.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

// ─── WASM Export Macro ───────────────────────────────────────────────────────

/// Export your bot as a WASM function callable by the Ludus server.
///
/// Generates the `next_move(ptr: i32, len: i32) -> i32` ABI function.
/// When compiled with `features = ["neural"]`, also exports `__LUDUS_NEURAL__`
/// so the sandbox automatically grants unlimited inference fuel.
#[macro_export]
macro_rules! export_bot {
    ($bot:ty) => {
        static mut RESPONSE_BUF: [u8; 256] = [0u8; 256];

        #[unsafe(no_mangle)]
        pub extern "C" fn next_move(ptr: i32, len: i32) -> i32 {
            let input_slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) };

            let result: String = if let Ok(json_str) = std::str::from_utf8(input_slice) {
                if let Ok(state) = serde_json::from_str::<ludus_sdk::GameState>(json_str) {
                    <$bot>::next_move(&state)
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            let bytes = result.as_bytes();
            let write_len = bytes.len().min(255);
            unsafe {
                RESPONSE_BUF[..write_len].copy_from_slice(&bytes[..write_len]);
                RESPONSE_BUF[write_len] = 0;
                RESPONSE_BUF.as_ptr() as i32
            }
        }

        // Signal to the Ludus sandbox to grant unlimited inference fuel.
        // Detected via module.exports() in wasm.rs — safe for all bot types.
        // micro_neural needs ~10M instructions; standard bots use <1M anyway.
        // #[cfg(feature = "neural")]
        #[unsafe(no_mangle)]
        pub static __LUDUS_NEURAL__: i32 = 1;
    };
}

/// Export a NNUE-powered bot as a WASM function callable by the Ludus server.
///
/// **Rétrocompatible** : generates the same `next_move(ptr, len) -> ptr` ABI
/// as [`export_bot!`] — no server changes required.
///
/// Differences from `export_bot!`:
/// - Your bot uses `nnue::best_move()` internally for ~4× faster evaluation.
/// - Exports `__LUDUS_NNUE__` (alongside `__LUDUS_NEURAL__`) for future
///   NNUE-aware orchestration.
///
/// # Usage
/// ```rust
/// const NNUE: &[u8] = include_bytes!("../models/chess_eval.ludus-nnue");
///
/// pub struct NnueBot;
//// impl LudusBot for NnueBot {
///     fn next_move(state: &GameState) -> String {
///         ludus_sdk::nnue::best_move(NNUE, state)
///             .unwrap_or_else(|| ludus_sdk::random_move(state))
///     }
/// }
/// export_nnue_bot!(NnueBot);
/// ```
#[macro_export]
macro_rules! export_nnue_bot {
    ($bot:ty) => {
        static mut __NNUE_RESPONSE_BUF: [u8; 256] = [0u8; 256];

        #[no_mangle]
        pub extern "C" fn next_move(ptr: i32, len: i32) -> i32 {
            let input_slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) };

            let result: String = if let Ok(json_str) = std::str::from_utf8(input_slice) {
                if let Ok(state) = serde_json::from_str::<ludus_sdk::GameState>(json_str) {
                    <$bot>::next_move(&state)
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            let bytes = result.as_bytes();
            let write_len = bytes.len().min(255);
            unsafe {
                __NNUE_RESPONSE_BUF[..write_len].copy_from_slice(&bytes[..write_len]);
                __NNUE_RESPONSE_BUF[write_len] = 0;
                __NNUE_RESPONSE_BUF.as_ptr() as i32
            }
        }

        // Same fuel signal as export_bot! (50M instruction budget)
        #[no_mangle]
        pub static __LUDUS_NEURAL__: i32 = 1;

        // Additional signal: NNUE-capable bot (reserved for future orchestration)
        #[no_mangle]
        pub static __LUDUS_NNUE__: i32 = 1;
    };
}

