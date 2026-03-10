//! Ludus Chess Bot — Exemple de référence
//!
//! Pour compiler en WASM :
//!   cargo build --release --target wasm32-unknown-unknown
//!
//! Le fichier .wasm se trouve dans :
//!   target/wasm32-unknown-unknown/release/ludus_sdk.wasm
//!
//! Uploadez-le dans votre profil Ludus.

use ludus_sdk::{export_bot, GameState, LudusBot};

pub struct ChessBot;

impl LudusBot for ChessBot {
    fn next_move(state: &GameState) -> String {
        // Utilise un coup pseudo-aléatoire pour éviter les Draw
        // par répétition de position (threefold repetition).
        //
        // Conseil : pour un bot plus fort, implémentez ici votre
        // propre algorithme (minimax, alpha-beta, MCTS...).
        // Le state.board contient l'échiquier complet.
        ludus_sdk::random_move(state)
    }
}

export_bot!(ChessBot);
