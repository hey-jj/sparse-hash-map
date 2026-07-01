//! Memory-efficient sparse hash map and set.
//!
//! [`SparseMap`] and [`SparseSet`] are open-addressing containers whose main
//! goal is low memory use at low load factor. They keep reasonable lookup speed.
//! They fit the same niche as space-efficient sparse hash tables, not dense
//! Swiss tables.
//!
//! # How the memory trick works
//!
//! A flat table pays the full size of a bucket for every empty slot. Here
//! buckets are grouped into sparse arrays of up to 64 logical indices each. An
//! array stores only its present values, packed together, plus a bitmap marking
//! which indices are occupied. The dense position of an index is the number of
//! occupied bits below it, found with a population count. An empty logical
//! bucket costs about one bit.
//!
//! # Example
//!
//! ```
//! use sparse_hash_map::SparseMap;
//!
//! let mut map: SparseMap<String, i32> = SparseMap::new();
//! map.insert("a".to_string(), 1);
//! map.insert("b".to_string(), 2);
//!
//! assert_eq!(map.get("a"), Some(&1));
//! assert_eq!(map.len(), 2);
//!
//! *map.get_mut("a").unwrap() = 10;
//! assert_eq!(map.get("a"), Some(&10));
//! ```
//!
//! # Iterator value semantics
//!
//! Map iteration yields `(&K, &V)`. The key is never mutable through an
//! iterator, so it cannot change under the map. To mutate a value, use
//! [`SparseMap::get_mut`] or [`SparseMap::iter_mut`].
//!
//! # Growth policies
//!
//! The default policy keeps the bucket count a power of two and maps a hash with
//! a mask. [`Mod`] grows by a rational factor. [`Prime`] steps through a table
//! of primes and spreads values better when the hash is poor.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod growth_policy;
pub mod hasher;
mod map;
pub mod popcount;
pub mod serialize;
mod set;
mod sparse_array;
mod sparse_hash;
pub mod sparsity;

pub use crate::growth_policy::{GrowthPolicy, LengthError, Mod, PowerOfTwo, Prime};
pub use crate::hasher::{EqKey, FnHash, HashKey, StdEq, StdHash};
pub use crate::map::SparseMap;
pub use crate::serialize::{
    Deserialize, DeserializeError, Deserializer, Serialize, Serializer,
    SERIALIZATION_PROTOCOL_VERSION,
};
pub use crate::set::SparseSet;
pub use crate::sparse_array::BITMAP_NB_BITS;
pub use crate::sparsity::{High, Low, Medium, Sparsity};

/// A prime-growth map. Better with poor hash functions.
pub type SparsePgMap<K, V, H = StdHash, E = StdEq, S = Medium> = SparseMap<K, V, H, E, Prime, S>;

/// A prime-growth set. Better with poor hash functions.
pub type SparsePgSet<K, H = StdHash, E = StdEq, S = Medium> = SparseSet<K, H, E, Prime, S>;

use core::hash::Hash;

impl<K, V> FromIterator<(K, V)> for SparseMap<K, V>
where
    K: Hash + Eq,
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut map = SparseMap::new();
        let (lower, _) = iter.size_hint();
        map.reserve(lower);
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

impl<K, V, const N: usize> From<[(K, V); N]> for SparseMap<K, V>
where
    K: Hash + Eq,
{
    fn from(array: [(K, V); N]) -> Self {
        array.into_iter().collect()
    }
}

impl<K> FromIterator<K> for SparseSet<K>
where
    K: Hash + Eq,
{
    fn from_iter<I: IntoIterator<Item = K>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut set = SparseSet::new();
        let (lower, _) = iter.size_hint();
        set.reserve(lower);
        for k in iter {
            set.insert(k);
        }
        set
    }
}

impl<K, const N: usize> From<[K; N]> for SparseSet<K>
where
    K: Hash + Eq,
{
    fn from(array: [K; N]) -> Self {
        array.into_iter().collect()
    }
}
