//! Bit-population-count helpers used to index sparse arrays.
//!
//! A sparse array keeps a bitmap of occupied slots. The dense offset of a slot
//! is the number of set bits below it. Both a hardware-backed path and a
//! branch-free SWAR fallback are provided. They return identical results for
//! every input.

/// Count the set bits in a 64-bit value using the SWAR Hamming-weight method.
///
/// This mirrors the Wikipedia "Hamming weight" algorithm. It is the portable
/// path used when no hardware instruction is available.
#[inline]
pub fn fallback_popcountll(mut x: u64) -> u32 {
    const M1: u64 = 0x5555_5555_5555_5555;
    const M2: u64 = 0x3333_3333_3333_3333;
    const M4: u64 = 0x0f0f_0f0f_0f0f_0f0f;
    const H01: u64 = 0x0101_0101_0101_0101;

    x -= (x >> 1) & M1;
    x = (x & M2) + ((x >> 2) & M2);
    x = (x + (x >> 4)) & M4;
    (x.wrapping_mul(H01) >> 56) as u32
}

/// Count the set bits in a 32-bit value using the SWAR Hamming-weight method.
#[inline]
pub fn fallback_popcount(mut x: u32) -> u32 {
    const M1: u32 = 0x5555_5555;
    const M2: u32 = 0x3333_3333;
    const M4: u32 = 0x0f0f_0f0f;
    const H01: u32 = 0x0101_0101;

    x -= (x >> 1) & M1;
    x = (x & M2) + ((x >> 2) & M2);
    x = (x + (x >> 4)) & M4;
    x.wrapping_mul(H01) >> 24
}

/// Count the set bits in a 64-bit value.
///
/// Uses the compiler intrinsic, which lowers to a hardware instruction when the
/// target supports it. Equal to [`fallback_popcountll`] for all inputs.
#[inline]
pub fn popcountll(x: u64) -> u32 {
    x.count_ones()
}

/// Count the set bits in a 32-bit value.
///
/// Uses the compiler intrinsic. Equal to [`fallback_popcount`] for all inputs.
#[inline]
pub fn popcount(x: u32) -> u32 {
    x.count_ones()
}
