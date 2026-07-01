//! Sparsity levels controlling per-array capacity slack.
//!
//! A sparse array over-allocates its dense storage in fixed steps. A larger
//! step wastes more memory but reallocates less often on insert. Sparsity does
//! not change lookup speed or iteration order. It changes only the memory and
//! insert-speed trade-off.

/// A sparsity level. `STEP` is the amount a full sparse array grows by.
pub trait Sparsity: Clone {
    /// Capacity growth step for a sparse array.
    const STEP: u8;
}

/// Least memory slack. Grows by 2. Slower inserts.
#[derive(Clone)]
pub struct High;
impl Sparsity for High {
    const STEP: u8 = 2;
}

/// Balanced slack. Grows by 4. The default.
#[derive(Clone)]
pub struct Medium;
impl Sparsity for Medium {
    const STEP: u8 = 4;
}

/// Most memory slack. Grows by 8. Faster inserts.
#[derive(Clone)]
pub struct Low;
impl Sparsity for Low {
    const STEP: u8 = 8;
}
