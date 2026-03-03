# Ludus Bot SDK 🤖

Write your chess/checkers/ludo bot in Rust, compile it to WASM, and upload it to your Ludus profile. The server will play for you automatically!

## Quick Start

### 1. Create a new Rust project

```bash
cargo new my-chess-bot
cd my-chess-bot
```

### 2. Add dependencies to `Cargo.toml`

```toml
[dependencies]
ludus-sdk = { path = "../ludus-sdk" }  # or: git = "https://github.com/YmClash/ludus-sdk.git"
serde_json = "1"

[lib]
crate-type = ["cdylib"]

[profile.release]
opt-level = "s"
lto = true
panic = "abort"
```

### 3. Write your bot in `src/lib.rs`

```rust
use ludus_sdk::{LudusBot, GameState, export_bot};

pub struct MyBot;

impl LudusBot for MyBot {
    fn next_move(state: &GameState) -> String {
        // state.legal_moves contains all valid moves in algebraic notation
        // Chess: "e2e4", "g1f3", etc.
        // Checkers: "12-16", "22x13" (x = capture)
        // Ludo: "47-0" (position-to-position)

        // Example: pick the first available move
        state.legal_moves
            .first()
            .cloned()
            .unwrap_or_default()
    }
}

// This generates the WASM entry point!
export_bot!(MyBot);
```

### 4. Compile to WASM

```bash
# Install the WASM target first (once)
rustup target add wasm32-unknown-unknown

# Build
cargo build --release --target wasm32-unknown-unknown
```

Your bot is at: `target/wasm32-unknown-unknown/release/my_chess_bot.wasm`

### 5. Upload to Ludus

Go to your [profile page](http://localhost:3001/profile), scroll to **🧠 Mon Bot**, and upload the `.wasm` file.

---

## GameState Reference

```rust
pub struct GameState {
    pub turn: String,           // "white" | "black" | "0" | "1"
    pub status: String,         // "InProgress" | "Check" | "Checkmate" etc.
    pub legal_moves: Vec<String>, // all legal moves you can make
    pub move_history: Vec<String>, // moves played so far
}
```

## Constraints

| Limit | Value |
|-------|-------|
| Max instructions | 1 000 000 |
| Max memory | 16 MB |
| Max .wasm file size | 4 MB |
| Imports | None (pure function) |

