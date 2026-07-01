//! Growth policies that map a hash to a bucket and decide the next table size.
//!
//! A policy owns the current bucket count and translates a hash into a bucket
//! index. Three strategies are provided:
//!
//! - [`PowerOfTwo`] keeps the bucket count a power of two and maps with a mask.
//!   Fast, and the default.
//! - [`Mod`] grows by a rational factor and maps with a modulo. Useful for
//!   slower growth.
//! - [`Prime`] uses a fixed table of primes. It spreads values better when the
//!   hash function is poor, such as an identity hash of pointers.
//!
//! Each policy is constructed from a minimum bucket count. The policy may round
//! that value up and reports the value it settled on. A request above the
//! policy maximum returns [`LengthError`].

/// The requested bucket count exceeds what a growth policy can represent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthError;

impl core::fmt::Display for LengthError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("the hash table exceeds its maximum size")
    }
}

impl std::error::Error for LengthError {}

/// Strategy for sizing the bucket array and mapping a hash to a bucket.
///
/// A policy is created with [`GrowthPolicy::new`], which returns the policy and
/// the bucket count it settled on. That count is at least the requested
/// minimum. [`GrowthPolicy::bucket_for_hash`] and [`GrowthPolicy::clear`] must
/// not allocate or panic.
pub trait GrowthPolicy: Clone {
    /// Build a policy for at least `min_bucket_count` buckets.
    ///
    /// Returns the policy and the bucket count it settled on. That count is the
    /// requested minimum rounded up as the policy requires. Returns
    /// [`LengthError`] when the request is above [`GrowthPolicy::max_bucket_count`].
    ///
    /// When `min_bucket_count` is 0 the settled count is 0 and
    /// `bucket_for_hash` returns 0 for every hash.
    fn new(min_bucket_count: usize) -> Result<(Self, usize), LengthError>;

    /// Map `hash` to a bucket in `[0, bucket_count)`.
    fn bucket_for_hash(&self, hash: usize) -> usize;

    /// The bucket count to use on the next growth.
    ///
    /// Returns [`LengthError`] when the table cannot grow further.
    fn next_bucket_count(&self) -> Result<usize, LengthError>;

    /// The largest bucket count the policy can represent.
    fn max_bucket_count(&self) -> usize;

    /// Reset the policy to the state of a 0-bucket table.
    fn clear(&mut self);

    /// Whether this policy keeps the bucket count a power of two.
    ///
    /// The engine uses this to pick the mask-based probe path.
    fn is_power_of_two() -> bool {
        false
    }
}

use crate::util::{is_power_of_two, round_up_to_power_of_two};

/// Grow by a power-of-two factor and map with a mask.
///
/// `FACTOR` must be a power of two and at least 2. The default is 2.
#[derive(Clone)]
pub struct PowerOfTwo<const FACTOR: usize = 2> {
    mask: usize,
}

impl<const FACTOR: usize> GrowthPolicy for PowerOfTwo<FACTOR> {
    fn new(min_bucket_count: usize) -> Result<(Self, usize), LengthError> {
        assert!(
            is_power_of_two(FACTOR) && FACTOR >= 2,
            "growth factor must be a power of two >= 2"
        );

        let max = Self::max_bucket_count_static();
        if min_bucket_count > max {
            return Err(LengthError);
        }

        if min_bucket_count > 0 {
            let rounded = round_up_to_power_of_two(min_bucket_count);
            Ok((Self { mask: rounded - 1 }, rounded))
        } else {
            Ok((Self { mask: 0 }, 0))
        }
    }

    #[inline]
    fn bucket_for_hash(&self, hash: usize) -> usize {
        hash & self.mask
    }

    fn next_bucket_count(&self) -> Result<usize, LengthError> {
        if (self.mask + 1) > Self::max_bucket_count_static() / FACTOR {
            return Err(LengthError);
        }
        Ok((self.mask + 1) * FACTOR)
    }

    fn max_bucket_count(&self) -> usize {
        Self::max_bucket_count_static()
    }

    fn clear(&mut self) {
        self.mask = 0;
    }

    fn is_power_of_two() -> bool {
        true
    }
}

impl<const FACTOR: usize> PowerOfTwo<FACTOR> {
    #[inline]
    fn max_bucket_count_static() -> usize {
        // Largest power of two representable in a usize.
        (usize::MAX / 2) + 1
    }
}

/// Grow by the rational factor `NUM / DEN` and map with a modulo.
///
/// The factor must be at least 1.1. The default is 3/2.
#[derive(Clone)]
pub struct Mod<const NUM: usize = 3, const DEN: usize = 2> {
    modulo: usize,
}

impl<const NUM: usize, const DEN: usize> Mod<NUM, DEN> {
    #[inline]
    fn factor() -> f64 {
        NUM as f64 / DEN as f64
    }

    #[inline]
    fn max_bucket_count_static() -> usize {
        (usize::MAX as f64 / Self::factor()) as usize
    }
}

impl<const NUM: usize, const DEN: usize> GrowthPolicy for Mod<NUM, DEN> {
    fn new(min_bucket_count: usize) -> Result<(Self, usize), LengthError> {
        assert!(Self::factor() >= 1.1, "growth factor should be >= 1.1");

        if min_bucket_count > Self::max_bucket_count_static() {
            return Err(LengthError);
        }

        let modulo = if min_bucket_count > 0 {
            min_bucket_count
        } else {
            1
        };
        // Mod does not change the requested count. It uses it as the modulus.
        Ok((Self { modulo }, min_bucket_count))
    }

    #[inline]
    fn bucket_for_hash(&self, hash: usize) -> usize {
        hash % self.modulo
    }

    fn next_bucket_count(&self) -> Result<usize, LengthError> {
        let max = Self::max_bucket_count_static();
        if self.modulo == max {
            return Err(LengthError);
        }

        let next = (self.modulo as f64 * Self::factor()).ceil();
        if !next.is_normal() {
            return Err(LengthError);
        }

        if next > max as f64 {
            Ok(max)
        } else {
            Ok(next as usize)
        }
    }

    fn max_bucket_count(&self) -> usize {
        Self::max_bucket_count_static()
    }

    fn clear(&mut self) {
        self.modulo = 1;
    }
}

/// The prime table used by [`Prime`]. Ascending, 40 entries.
///
/// Exposed so callers can reason about the growth sequence. A `Prime` policy
/// steps through these on each growth.
pub const PRIMES_TABLE: [usize; 40] = [
    1, 5, 17, 29, 37, 53, 67, 79, 97, 131, 193, 257, 389, 521, 769, 1031, 1543, 2053, 3079, 6151,
    12289, 24593, 49157, 98317, 196613, 393241, 786433, 1572869, 3145739, 6291469, 12582917,
    25165843, 50331653, 100663319, 201326611, 402653189, 805306457, 1610612741, 3221225473,
    4294967291,
];

/// Grow by stepping through a fixed table of primes and map with a modulo.
///
/// A modulo by a compile-time-unknown prime is still fast here because the
/// table is small and the index selects the divisor. This policy spreads values
/// better than [`PowerOfTwo`] when the hash is poor.
#[derive(Clone)]
pub struct Prime {
    iprime: usize,
}

impl GrowthPolicy for Prime {
    fn new(min_bucket_count: usize) -> Result<(Self, usize), LengthError> {
        // Lower bound: first prime not less than the request.
        let iprime = PRIMES_TABLE.partition_point(|&p| p < min_bucket_count);
        if iprime == PRIMES_TABLE.len() {
            return Err(LengthError);
        }

        let settled = if min_bucket_count > 0 {
            PRIMES_TABLE[iprime]
        } else {
            0
        };
        Ok((Self { iprime }, settled))
    }

    #[inline]
    fn bucket_for_hash(&self, hash: usize) -> usize {
        hash % PRIMES_TABLE[self.iprime]
    }

    fn next_bucket_count(&self) -> Result<usize, LengthError> {
        if self.iprime + 1 >= PRIMES_TABLE.len() {
            return Err(LengthError);
        }
        Ok(PRIMES_TABLE[self.iprime + 1])
    }

    fn max_bucket_count(&self) -> usize {
        PRIMES_TABLE[PRIMES_TABLE.len() - 1]
    }

    fn clear(&mut self) {
        self.iprime = 0;
    }
}
