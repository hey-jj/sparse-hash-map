//! Shared test harness: fixture generators, collision hashers, test value types,
//! and a binary codec. These fix the key and value sequences and collision
//! patterns the tests rely on.

#![allow(dead_code)]

use std::hash::{BuildHasher, Hash};

use sparse_hash_map::{Deserializer, EqKey, HashKey, Serializer, StdEq, StdHash};

/// A hasher that returns the key value as the hash. Exact bucket placement.
#[derive(Clone, Default)]
pub struct IdentityHash;

macro_rules! impl_identity_hash {
    ($($t:ty),*) => {$(
        impl HashKey<$t> for IdentityHash {
            fn hash_key(&self, key: &$t) -> usize {
                *key as usize
            }
        }
    )*};
}
impl_identity_hash!(i32, i64, u32, u64, usize);

/// A hasher that forces heavy collisions by reducing a stable hash modulo `MOD`.
///
/// The value maps through the standard hasher, then modulo `MOD`. `MOD = 9` is
/// the workhorse. Parity here is about the collision regime, not an exact bucket.
#[derive(Clone)]
pub struct ModHash<const MOD: usize> {
    build: std::collections::hash_map::RandomState,
}

impl<const MOD: usize> Default for ModHash<MOD> {
    fn default() -> Self {
        Self {
            build: std::collections::hash_map::RandomState::new(),
        }
    }
}

impl<const MOD: usize, Q: Hash + ?Sized> HashKey<Q> for ModHash<MOD> {
    fn hash_key(&self, key: &Q) -> usize {
        (self.build.hash_one(key) as usize) % MOD
    }
}

/// A move-only value backed by a string. It has no `Clone`, so it stresses the
/// move-only code paths of the container.
#[derive(Debug)]
pub struct MoveOnly {
    value: String,
}

impl MoveOnly {
    pub fn from_i64(v: i64) -> Self {
        Self {
            value: v.to_string(),
        }
    }
    pub fn from_string(v: String) -> Self {
        Self { value: v }
    }
    pub fn value(&self) -> &str {
        &self.value
    }
}

impl PartialEq for MoveOnly {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}
impl Eq for MoveOnly {}

impl Hash for MoveOnly {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl sparse_hash_map::Serialize for MoveOnly {
    fn serialize<S: Serializer>(&self, serializer: &mut S) {
        self.value.serialize(serializer);
    }
}

impl sparse_hash_map::Deserialize for MoveOnly {
    fn deserialize<D: Deserializer>(deserializer: &mut D) -> Self {
        Self {
            value: String::deserialize(deserializer),
        }
    }
}

/// A copy-only style value backed by a string. Distinct type for the type sweep.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CopyOnly {
    value: String,
}

impl CopyOnly {
    pub fn from_i64(v: i64) -> Self {
        Self {
            value: v.to_string(),
        }
    }
    pub fn value(&self) -> &str {
        &self.value
    }
}

/// Deterministic key and value generators keyed on a counter.
pub trait TestType: Sized + Eq + Hash + std::fmt::Debug {
    fn get_key(counter: usize) -> Self;
    fn get_value(counter: usize) -> Self;
}

impl TestType for i64 {
    fn get_key(counter: usize) -> Self {
        counter as i64
    }
    fn get_value(counter: usize) -> Self {
        (counter * 2) as i64
    }
}

impl TestType for String {
    fn get_key(counter: usize) -> Self {
        format!("Key {counter}")
    }
    fn get_value(counter: usize) -> Self {
        format!("Value {counter}")
    }
}

impl TestType for MoveOnly {
    fn get_key(counter: usize) -> Self {
        MoveOnly::from_i64(counter as i64)
    }
    fn get_value(counter: usize) -> Self {
        MoveOnly::from_i64((counter * 2) as i64)
    }
}

impl TestType for CopyOnly {
    fn get_key(counter: usize) -> Self {
        CopyOnly::from_i64(counter as i64)
    }
    fn get_value(counter: usize) -> Self {
        CopyOnly::from_i64((counter * 2) as i64)
    }
}

/// A binary writer for the test wire format.
///
/// Integers are little-endian. Strings are a `u64` length then the bytes.
#[derive(Default)]
pub struct VecSerializer {
    pub buf: Vec<u8>,
}

impl Serializer for VecSerializer {
    fn write_u64(&mut self, value: u64) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }
    fn write_f32(&mut self, value: f32) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }
    fn write_bytes(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }
}

/// The matching reader for [`VecSerializer`].
pub struct VecDeserializer<'a> {
    pub buf: &'a [u8],
    pub pos: usize,
}

impl<'a> VecDeserializer<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
}

impl<'a> Deserializer for VecDeserializer<'a> {
    fn read_u64(&mut self) -> u64 {
        let mut a = [0u8; 8];
        a.copy_from_slice(&self.buf[self.pos..self.pos + 8]);
        self.pos += 8;
        u64::from_le_bytes(a)
    }
    fn read_f32(&mut self) -> f32 {
        let mut a = [0u8; 4];
        a.copy_from_slice(&self.buf[self.pos..self.pos + 4]);
        self.pos += 4;
        f32::from_le_bytes(a)
    }
    fn read_bytes(&mut self, len: usize) -> Vec<u8> {
        let s = self.buf[self.pos..self.pos + len].to_vec();
        self.pos += len;
        s
    }
    fn remaining(&self) -> Option<usize> {
        Some(self.buf.len().saturating_sub(self.pos))
    }
}

/// Convenience aliases used across test files.
pub type DefaultHash = StdHash<std::collections::hash_map::RandomState>;
pub type DefaultEq = StdEq;

/// A helper so generic tests can build default hashers of any wrapper type.
pub trait MakeDefault {
    fn make() -> Self;
}

impl<B: BuildHasher + Default> MakeDefault for StdHash<B> {
    fn make() -> Self {
        StdHash::default()
    }
}
impl MakeDefault for StdEq {
    fn make() -> Self {
        StdEq
    }
}
impl<const MOD: usize> MakeDefault for ModHash<MOD> {
    fn make() -> Self {
        ModHash::default()
    }
}
impl MakeDefault for IdentityHash {
    fn make() -> Self {
        IdentityHash
    }
}

/// Assert that `hash` and `eq` are usable together for key `K`.
///
/// A no-op marker to keep unused-import lints quiet in generic tests.
pub fn assert_usable<K, H, E>()
where
    H: HashKey<K>,
    E: EqKey<K, K>,
{
}
