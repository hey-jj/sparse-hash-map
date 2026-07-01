//! Population-count golden values. The hardware path and the SWAR fallback must
//! return the same result for every input.

use sparse_hash_map::popcount::{fallback_popcount, fallback_popcountll, popcount, popcountll};

const U32_CASES: &[(u32, u32)] = &[(0, 0), (1, 1), (2, 1), (294967496, 12), (u32::MAX, 32)];

const U64_CASES: &[(u64, u32)] = &[
    (0, 0),
    (1, 1),
    (2, 1),
    (294967496, 12),
    (8446744073709551416, 40),
    (u64::MAX, 64),
];

#[test]
fn test_popcount_1() {
    for &(input, expected) in U32_CASES {
        assert_eq!(popcount(input), expected, "popcount({input})");
    }
}

#[test]
fn test_popcountll_1() {
    for &(input, expected) in U64_CASES {
        assert_eq!(popcountll(input), expected, "popcountll({input})");
    }
}

#[test]
fn test_fallback_popcount_1() {
    for &(input, expected) in U32_CASES {
        assert_eq!(
            fallback_popcount(input),
            expected,
            "fallback_popcount({input})"
        );
    }
}

#[test]
fn test_fallback_popcountll_1() {
    for &(input, expected) in U64_CASES {
        assert_eq!(
            fallback_popcountll(input),
            expected,
            "fallback_popcountll({input})"
        );
    }
}

#[test]
fn hardware_matches_fallback_32() {
    let samples = [
        0u32,
        1,
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
fn hardware_matches_fallback_64() {
    let samples = [
        0u64,
        1,
        u64::MAX,
        0xDEAD_BEEF_CAFE_BABE,
        8446744073709551416,
        1 << 63,
    ];
    for x in samples {
        assert_eq!(popcountll(x), fallback_popcountll(x), "mismatch at {x}");
    }
}
