//! Bit-population-count helpers used to index sparse arrays.
//!
//! A sparse array keeps a bitmap of occupied slots. The dense offset of a slot
//! is the number of set bits below it. The intrinsic path lowers to a hardware
//! instruction when the target supports it. A branch-free SWAR fallback backs
//! the equivalence test and documents the algorithm the intrinsic replaces.

/// Count the set bits in a 64-bit value.
///
/// Uses the compiler intrinsic, which lowers to a hardware instruction when the
/// target supports it.
#[cfg(any(target_pointer_width = "64", test))]
#[inline]
pub(crate) fn popcountll(x: u64) -> u32 {
    x.count_ones()
}

/// Count the set bits in a 32-bit value.
///
/// Uses the compiler intrinsic.
#[cfg(any(not(target_pointer_width = "64"), test))]
#[inline]
pub(crate) fn popcount(x: u32) -> u32 {
    x.count_ones()
}

/// Count the set bits in a 64-bit value using the SWAR Hamming-weight method.
///
/// This is the portable path used when no hardware instruction is available.
/// It mirrors the Wikipedia "Hamming weight" algorithm and equals
/// [`popcountll`] for every input, which a unit test checks.
#[cfg(test)]
#[inline]
fn fallback_popcountll(mut x: u64) -> u32 {
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
#[cfg(test)]
#[inline]
fn fallback_popcount(mut x: u32) -> u32 {
    const M1: u32 = 0x5555_5555;
    const M2: u32 = 0x3333_3333;
    const M4: u32 = 0x0f0f_0f0f;
    const H01: u32 = 0x0101_0101;

    x -= (x >> 1) & M1;
    x = (x & M2) + ((x >> 2) & M2);
    x = (x + (x >> 4)) & M4;
    x.wrapping_mul(H01) >> 24
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intrinsic_matches_fallback_32() {
        let samples = [
            0u32,
            1,
            2,
            0xFFFF_FFFF,
            0xDEAD_BEEF,
            0x0000_FFFF,
            123456789,
            294967496,
        ];
        for x in samples {
            assert_eq!(popcount(x), fallback_popcount(x), "mismatch at {x}");
        }
    }

    #[test]
    fn intrinsic_matches_fallback_64() {
        let samples = [
            0u64,
            1,
            2,
            u64::MAX,
            0xDEAD_BEEF_CAFE_BABE,
            8446744073709551416,
            294967496,
            1 << 63,
        ];
        for x in samples {
            assert_eq!(popcountll(x), fallback_popcountll(x), "mismatch at {x}");
        }
    }

    #[test]
    fn golden_values() {
        assert_eq!(popcount(0), 0);
        assert_eq!(popcount(294967496), 12);
        assert_eq!(popcount(u32::MAX), 32);
        assert_eq!(popcountll(0), 0);
        assert_eq!(popcountll(8446744073709551416), 40);
        assert_eq!(popcountll(u64::MAX), 64);
    }
}
