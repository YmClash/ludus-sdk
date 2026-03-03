/// Ludus Bot SDK
///
/// # How to write a bot
///
/// 1. Add this crate as a dependency
/// 2. Implement the `LudusBot` trait
/// 3. Use the `export_bot!` macro to export it
/// 4. Compile with: `cargo build --release --target wasm32-unknown-unknown`
/// 5. Upload the resulting `.wasm` file to your Ludus profile
///
/// # Example
/// ```rust
/// use ludus_sdk::{LudusBot, GameState, export_bot};
///
/// struct RandomBot;
///
/// impl LudusBot for RandomBot {
///     fn next_move(state: &GameState) -> String {
///         // Pick the first legal move
///         state.legal_moves.first().cloned().unwrap_or_default()
///     }
/// }
///
/// export_bot!(RandomBot);
/// ```
use serde::{Deserialize, Serialize};

// ─── Game State (what the server sends to your bot) ─────

/// The game state your bot receives. This is a subset of the full
/// game state, serialized as JSON by the Ludus server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    /// Whose turn it is: "white" or "black" (Chess/Checkers), or "0"/"1" (Ludo)
    pub turn: String,

    /// Current game status: "InProgress", "Check", etc.
    pub status: String,

    /// List of legal moves in algebraic notation (e.g. "e2e4", "12-16", "47-0")
    pub legal_moves: Vec<String>,

    /// Move history (all moves played so far)
    pub move_history: Vec<String>,
}

// ─── Bot Trait ───────────────────────────────────────────

/// Implement this trait to define your bot's strategy.
///
/// `next_move` receives the full game state and must return
/// one of the strings in `state.legal_moves`.
pub trait LudusBot {
    fn next_move(state: &GameState) -> String;
}

// ─── WASM Export Macro ──────────────────────────────────

/// Export your bot as a WASM function.
///
/// This macro generates the `next_move(ptr: i32, len: i32) -> i32`
/// function that the Ludus server will call.
///
/// # Usage
/// Place this at the bottom of your bot file:
/// ```rust
/// ///export_bot!(MyBotType);
/// ```
#[macro_export]
macro_rules! export_bot {
    ($bot:ty) => {
        /// Static buffer for the bot's response string.
        /// (No allocator needed for simple bots)
        static mut RESPONSE_BUF: [u8; 256] = [0u8; 256];

        /// Entry point called by the Ludus server.
        ///
        /// - Reads JSON game state from WASM memory at `ptr` for `len` bytes
        /// - Parses it as a `GameState`
        /// - Calls `<$bot>::next_move()`
        /// - Writes the result as a null-terminated string back into WASM memory
        /// - Returns pointer to the result
        #[no_mangle]
        pub extern "C" fn next_move(ptr: i32, len: i32) -> i32 {
            // Safety: we are the sole user of WASM linear memory
            let input_slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) };

            // Parse game state; if parse fails, return the first legal move hardcoded
            let result: String = if let Ok(json_str) = std::str::from_utf8(input_slice) {
                if let Ok(state) = serde_json::from_str::<ludus_sdk::GameState>(json_str) {
                    <$bot>::next_move(&state)
                } else {
                    // Fallback: empty string signals error to server
                    String::new()
                }
            } else {
                String::new()
            };

            // Write result into static response buffer
            let bytes = result.as_bytes();
            let write_len = bytes.len().min(255);
            unsafe {
                RESPONSE_BUF[..write_len].copy_from_slice(&bytes[..write_len]);
                RESPONSE_BUF[write_len] = 0; // null terminator
                RESPONSE_BUF.as_ptr() as i32
            }
        }
    };
}

// ─── Example Bot (for reference / testing) ──────────────

/// A simple bot that always picks the first legal move.
/// Useful as a starting template.
pub struct FirstMoveBot;

impl LudusBot for FirstMoveBot {
    fn next_move(state: &GameState) -> String {
        state.legal_moves.first().cloned().unwrap_or_default()
    }
}
