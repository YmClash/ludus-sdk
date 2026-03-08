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
/// # Minimal example
/// ```rust
/// use ludus_sdk::{LudusBot, GameState, export_bot};
///
/// pub struct MyBot;
///
/// impl LudusBot for MyBot {
///     fn next_move(state: &GameState) -> String {
///         ludus_sdk::random_move(state)
///     }
/// }
///
/// export_bot!(MyBot);
/// ```
use serde::{Deserialize, Serialize};

// ─── Game State (what the server sends to your bot) ─────

/// The game state your bot receives. This is a subset of the full
/// game state, serialized as JSON by the Ludus server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    /// Whose turn it is: "white" or "black" (Chess/Checkers), or "0"/"1" (Ludo)
    pub turn: String,

    /// Current game status: "InProgress", "Check", "Checkmate", "Draw", etc.
    pub status: String,

    /// List of legal moves in UCI notation (e.g. "e2e4", "12-16", "47-0").
    /// Your bot MUST return one of these strings.
    pub legal_moves: Vec<String>,

    /// Move history (all moves played so far as algebraic notation)
    pub move_history: Vec<String>,

    /// Board state as a 2D array [row][col].
    ///
    /// **Chess**: "wK", "bP", "wR", etc. — empty square = "".
    /// Rows: 0 = rank 1 (white back rank), 7 = rank 8 (black back rank).
    /// Cols: 0 = file a, 7 = file h.
    ///
    /// **Checkers**: "r" (red), "b" (black), "R" (red king), "B" (black king), "" (empty).
    ///
    /// **Ludo**: [[red_tokens...], [blue_tokens...], [current_player, last_dice_roll]]
    /// Token positions: "Yard", "T{n}" (track), "H{n}" (home stretch), "Home".
    #[serde(default)]
    pub board: Vec<Vec<String>>,
}

impl GameState {
    /// Returns true if no moves are available (game is over).
    pub fn is_terminal(&self) -> bool {
        self.legal_moves.is_empty()
    }

    /// Returns the piece at a chess square (e.g. "e2" → "wP").
    /// Returns `""` for empty or if coordinates are invalid.
    /// Only meaningful for Chess games.
    pub fn piece_at(&self, square: &str) -> &str {
        let bytes = square.as_bytes();
        if bytes.len() < 2 {
            return "";
        }
        let col = (bytes[0].wrapping_sub(b'a')) as usize;
        let row = (bytes[1].wrapping_sub(b'1')) as usize;
        if row >= 8 || col >= 8 || self.board.len() <= row {
            return "";
        }
        if self.board[row].len() <= col {
            return "";
        }
        &self.board[row][col]
    }

    /// Get the color prefix of the piece at a square: "w", "b", or "".
    /// Useful for Chess bots.
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

// ─── Bot Trait ───────────────────────────────────────────

/// Implement this trait to define your bot's strategy.
///
/// `next_move` receives the full game state and must return
/// one of the strings in `state.legal_moves`.
pub trait LudusBot {
    fn next_move(state: &GameState) -> String;
}

// ─── Built-in helpers ────────────────────────────────────

/// Pick a pseudo-random legal move.
///
/// Uses a simple LCG seeded from the game history to avoid deterministic
/// repetition. **This is the recommended default for basic bots** — it
/// prevents threefold-repetition draws that occur when always picking
/// `legal_moves[0]`.
///
/// # Example
/// ```rust
/// ///fn next_move(state: &GameState) -> String {
///     ///ludus_sdk::random_move(state)
/// }
/// ```
pub fn random_move(state: &GameState) -> String {
    if state.legal_moves.is_empty() {
        return String::new();
    }
    if state.legal_moves.len() == 1 {
        return state.legal_moves[0].clone();
    }

    // Seed from the number of moves played + XOR of move string lengths
    // (no OS randomness needed — works in WASM/no_std)
    let seed = lcg_seed(state);
    let idx = (seed as usize) % state.legal_moves.len();
    state.legal_moves[idx].clone()
}

/// LCG-based seed derived from game state (deterministic, WASM-safe).
fn lcg_seed(state: &GameState) -> u64 {
    let mut s: u64 = 0x9e3779b97f4a7c15; // fixed golden-ratio seed

    // Mix in move history length
    s = s.wrapping_add(state.move_history.len() as u64);
    s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);

    // Mix in each move string (last 4 moves to stay fast)
    for mv in state.move_history.iter().rev().take(4) {
        for b in mv.bytes() {
            s ^= b as u64;
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
        }
    }

    // Mix in legal_moves count (varies with position)
    s ^= state.legal_moves.len() as u64;
    s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);

    s
}

/// Pick the first legal move. Useful as a deterministic baseline,
/// but **will cause Draw by repetition** in symmetric positions — use
/// `random_move` instead for real bots.
pub fn first_move(state: &GameState) -> String {
    state.legal_moves.first().cloned().unwrap_or_default()
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
    };
}
