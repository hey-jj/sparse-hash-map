//! A memory-efficient open-addressing hash set.
//!
//! [`SparseSet`] stores each key inside sparse arrays, sharing the engine with
//! [`crate::SparseMap`]. It uses far less memory than a flat table at low load
//! factor. The API mirrors the map minus the value-side operations.

use core::borrow::Borrow;
use core::hash::BuildHasher;
use core::marker::PhantomData;

use crate::growth_policy::{GrowthPolicy, LengthError, PowerOfTwo};
use crate::hasher::{EqKey, HashKey, StdEq, StdHash};
use crate::serialize::{Deserialize, DeserializeError, Deserializer, Serialize, Serializer};
use crate::sparse_hash::{
    KeySelect, SparseHash, DEFAULT_INIT_BUCKET_COUNT, DEFAULT_MAX_LOAD_FACTOR,
};
use crate::sparsity::{Medium, Sparsity};

/// Reads the key from a stored key, which is the key itself.
pub struct IdentityKeySelect<K>(PhantomData<K>);

impl<K> KeySelect<K> for IdentityKeySelect<K> {
    type Key = K;
    #[inline]
    fn key(value: &K) -> &K {
        value
    }
}

/// A hash set that trades a little insert speed for low memory use.
///
/// Type parameters match [`crate::SparseMap`] without the value type.
pub struct SparseSet<K, H = StdHash, E = StdEq, P = PowerOfTwo<2>, S = Medium> {
    ht: SparseHash<K, IdentityKeySelect<K>, H, E, P, S>,
}

impl<K> SparseSet<K, StdHash, StdEq, PowerOfTwo<2>, Medium> {
    /// An empty set with no allocation.
    ///
    /// Construction places no bound on `K`. The `Hash` and `Eq` bounds belong
    /// to the operations that hash or compare a key.
    #[must_use]
    pub fn new() -> Self {
        Self::with_bucket_count(DEFAULT_INIT_BUCKET_COUNT)
    }

    /// An empty set sized for at least `bucket_count` buckets.
    ///
    /// # Panics
    ///
    /// Panics when `bucket_count` exceeds the policy maximum.
    #[must_use]
    pub fn with_bucket_count(bucket_count: usize) -> Self {
        Self::try_with_bucket_count(bucket_count).expect("bucket count within policy limit")
    }

    /// An empty set sized for at least `bucket_count` buckets, fallibly.
    pub fn try_with_bucket_count(bucket_count: usize) -> Result<Self, LengthError> {
        Ok(Self {
            ht: SparseHash::new(
                bucket_count,
                StdHash::default(),
                StdEq,
                DEFAULT_MAX_LOAD_FACTOR,
            )?,
        })
    }
}

impl<K, H, E, P, S> Default for SparseSet<K, H, E, P, S>
where
    H: Default,
    E: Default,
    P: GrowthPolicy,
{
    /// An empty set with default parts and no allocation.
    ///
    /// Available when the hasher and comparator are [`Default`] and the policy
    /// can build a zero-bucket table. Places no bound on `K`.
    fn default() -> Self {
        Self {
            ht: SparseHash::new(
                DEFAULT_INIT_BUCKET_COUNT,
                H::default(),
                E::default(),
                DEFAULT_MAX_LOAD_FACTOR,
            )
            .expect("zero bucket count is within every policy limit"),
        }
    }
}

impl<K, B, P, S> SparseSet<K, StdHash<B>, StdEq, P, S>
where
    K: Eq,
    B: BuildHasher + Default,
    P: GrowthPolicy,
    S: Sparsity,
    StdHash<B>: HashKey<K>,
{
    /// An empty set that hashes with `B` and uses policy `P` and sparsity `S`.
    ///
    /// # Panics
    ///
    /// Panics when `bucket_count` exceeds the policy maximum.
    #[must_use]
    pub fn with_hasher_and_bucket_count(bucket_count: usize) -> Self {
        Self {
            ht: SparseHash::new(
                bucket_count,
                StdHash::default(),
                StdEq,
                DEFAULT_MAX_LOAD_FACTOR,
            )
            .expect("bucket count within policy limit"),
        }
    }
}

impl<K, H, E, P, S> SparseSet<K, H, E, P, S>
where
    H: HashKey<K> + Clone,
    E: EqKey<K, K> + Clone,
    P: GrowthPolicy,
    S: Sparsity,
{
    /// Build a set from explicit hasher, comparator, policy, and sparsity.
    ///
    /// # Panics
    ///
    /// Panics when `bucket_count` exceeds the policy maximum.
    #[must_use]
    pub fn with_parts(bucket_count: usize, hash: H, key_eq: E) -> Self {
        Self {
            ht: SparseHash::new(bucket_count, hash, key_eq, DEFAULT_MAX_LOAD_FACTOR)
                .expect("bucket count within policy limit"),
        }
    }

    /// Number of keys.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.ht.len()
    }

    /// Whether the set holds no keys.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ht.is_empty()
    }

    /// The logical bucket count. Zero for a fresh set.
    #[inline]
    #[must_use]
    pub fn bucket_count(&self) -> usize {
        self.ht.bucket_count()
    }

    /// The largest bucket count the set can hold.
    #[inline]
    #[must_use]
    pub fn max_bucket_count(&self) -> usize {
        self.ht.max_bucket_count()
    }

    /// The largest number of keys the set can hold.
    #[inline]
    #[must_use]
    pub fn max_size(&self) -> usize {
        self.ht.max_size()
    }

    /// Ratio of keys to buckets. Zero for an empty set.
    #[inline]
    #[must_use]
    pub fn load_factor(&self) -> f32 {
        self.ht.load_factor()
    }

    /// The maximum load factor before a grow.
    #[inline]
    #[must_use]
    pub fn max_load_factor(&self) -> f32 {
        self.ht.max_load_factor()
    }

    /// Set the maximum load factor, clamped to `[0.1, 0.8]`.
    pub fn set_max_load_factor(&mut self, ml: f32) {
        self.ht.set_max_load_factor(ml);
    }

    /// The hasher.
    #[inline]
    #[must_use]
    pub fn hash_function(&self) -> &H {
        self.ht.hash_function()
    }

    /// The key comparator.
    #[inline]
    #[must_use]
    pub fn key_eq(&self) -> &E {
        self.ht.key_eq()
    }

    /// Remove every key. Keeps the bucket count.
    pub fn clear(&mut self) {
        self.ht.clear();
    }

    /// Grow so the set holds at least `count` buckets.
    pub fn rehash(&mut self, count: usize) {
        self.ht.rehash(count);
    }

    /// Reserve room for `count` keys without exceeding the load factor.
    pub fn reserve(&mut self, count: usize) {
        self.ht.reserve(count);
    }

    /// Insert `key`. Returns whether it was newly inserted.
    pub fn insert(&mut self, key: K) -> bool {
        self.ht.insert(key).1
    }

    /// Whether `key` is present.
    #[must_use]
    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: ?Sized,
        H: HashKey<Q>,
        E: EqKey<K, Q>,
    {
        let hash = self.ht.hash_function().hash_key(key);
        self.ht.contains(key, hash)
    }

    /// Whether `key` is present, using a precomputed hash.
    #[must_use]
    pub fn contains_precalc<Q>(&self, key: &Q, hash: usize) -> bool
    where
        K: Borrow<Q>,
        Q: ?Sized,
        H: HashKey<Q>,
        E: EqKey<K, Q>,
    {
        self.ht.contains(key, hash)
    }

    /// 1 when `key` is present, 0 otherwise.
    #[must_use]
    pub fn count<Q>(&self, key: &Q) -> usize
    where
        K: Borrow<Q>,
        Q: ?Sized,
        H: HashKey<Q>,
        E: EqKey<K, Q>,
    {
        usize::from(self.contains(key))
    }

    /// 1 when `key` is present, 0 otherwise, using a precomputed hash.
    #[must_use]
    pub fn count_precalc<Q>(&self, key: &Q, hash: usize) -> usize
    where
        K: Borrow<Q>,
        Q: ?Sized,
        H: HashKey<Q>,
        E: EqKey<K, Q>,
    {
        usize::from(self.contains_precalc(key, hash))
    }

    /// The range of keys equal to `key`.
    ///
    /// A set holds at most one key per value, so the range is empty or a single
    /// key. The returned iterator yields `&K` and has length 0 or 1.
    pub fn equal_range<Q>(&self, key: &Q) -> EqualRange<'_, K>
    where
        K: Borrow<Q>,
        Q: ?Sized,
        H: HashKey<Q>,
        E: EqKey<K, Q>,
    {
        EqualRange {
            item: self.get(key),
        }
    }

    /// The range of keys equal to `key`, using a precomputed hash.
    pub fn equal_range_precalc<Q>(&self, key: &Q, hash: usize) -> EqualRange<'_, K>
    where
        K: Borrow<Q>,
        Q: ?Sized,
        H: HashKey<Q>,
        E: EqKey<K, Q>,
    {
        EqualRange {
            item: self.ht.get(key, hash),
        }
    }

    /// A reference to the stored key equal to `key`.
    #[must_use]
    pub fn get<Q>(&self, key: &Q) -> Option<&K>
    where
        K: Borrow<Q>,
        Q: ?Sized,
        H: HashKey<Q>,
        E: EqKey<K, Q>,
    {
        let hash = self.ht.hash_function().hash_key(key);
        self.ht.get(key, hash)
    }

    /// Remove `key`. Returns 1 when erased, 0 otherwise.
    pub fn erase<Q>(&mut self, key: &Q) -> usize
    where
        K: Borrow<Q>,
        Q: ?Sized,
        H: HashKey<Q>,
        E: EqKey<K, Q>,
    {
        let hash = self.ht.hash_function().hash_key(key);
        self.ht.erase(key, hash)
    }

    /// Remove and return the stored key equal to `key`.
    pub fn take<Q>(&mut self, key: &Q) -> Option<K>
    where
        K: Borrow<Q>,
        Q: ?Sized,
        H: HashKey<Q>,
        E: EqKey<K, Q>,
    {
        let hash = self.ht.hash_function().hash_key(key);
        self.ht.remove(key, hash)
    }

    /// Remove and return the first key in iteration order.
    pub fn pop_front(&mut self) -> Option<K> {
        self.ht.remove_nth(0)
    }

    /// Remove `count` keys starting at iteration index `skip`.
    ///
    /// Erasing leaves a tombstone. The walk is a single forward pass.
    pub fn erase_range(&mut self, skip: usize, count: usize) {
        self.ht.erase_range(skip, count);
    }

    /// Remove every key, leaving tombstones. Keeps the bucket count.
    ///
    /// Use [`SparseSet::clear`] to also reset the tombstones and counters.
    pub fn erase_all(&mut self) {
        self.ht.erase_all();
    }

    /// Keep only the keys for which `keep` returns true.
    ///
    /// Removed keys become tombstones. Each sparse array is scanned once.
    pub fn retain<F>(&mut self, mut keep: F)
    where
        F: FnMut(&K) -> bool,
    {
        self.ht.retain(|k| keep(k));
    }
}

// Equality. Order-independent membership check.

impl<K, H, E, P, S> PartialEq for SparseSet<K, H, E, P, S>
where
    H: HashKey<K> + Clone,
    E: EqKey<K, K> + Clone,
    P: GrowthPolicy,
    S: Sparsity,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().all(|k| other.contains(k))
    }
}

impl<K, H, E, P, S> Eq for SparseSet<K, H, E, P, S>
where
    H: HashKey<K> + Clone,
    E: EqKey<K, K> + Clone,
    P: GrowthPolicy,
    S: Sparsity,
{
}

impl<K, H, E, P, S> core::fmt::Debug for SparseSet<K, H, E, P, S>
where
    K: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<K, H, E, P, S> Clone for SparseSet<K, H, E, P, S>
where
    K: Clone,
    H: Clone,
    E: Clone,
    P: Clone,
{
    fn clone(&self) -> Self {
        Self {
            ht: self.ht.clone(),
        }
    }
}

// Iteration.

impl<K, H, E, P, S> SparseSet<K, H, E, P, S> {
    /// A forward iterator over keys.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, K> {
        Iter {
            inner: self.ht.iter(),
        }
    }
}

/// Range of keys equal to a lookup key. Length 0 or 1.
///
/// A [`SparseSet`] holds at most one matching key. The range from
/// [`SparseSet::equal_range`] yields that key once, or nothing.
pub struct EqualRange<'a, K> {
    item: Option<&'a K>,
}

impl<'a, K> Iterator for EqualRange<'a, K> {
    type Item = &'a K;
    fn next(&mut self) -> Option<&'a K> {
        self.item.take()
    }
}

impl<K> ExactSizeIterator for EqualRange<'_, K> {
    fn len(&self) -> usize {
        usize::from(self.item.is_some())
    }
}

/// Iterator over keys of a [`SparseSet`].
pub struct Iter<'a, K> {
    inner: crate::sparse_hash::Iter<'a, K>,
}

impl<'a, K> Iterator for Iter<'a, K> {
    type Item = &'a K;
    fn next(&mut self) -> Option<&'a K> {
        self.inner.next()
    }
}

impl<'a, K, H, E, P, S> IntoIterator for &'a SparseSet<K, H, E, P, S> {
    type Item = &'a K;
    type IntoIter = Iter<'a, K>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Owning iterator over keys of a [`SparseSet`].
pub struct IntoIter<K> {
    inner: crate::sparse_hash::IntoIter<K>,
}

impl<K> Iterator for IntoIter<K> {
    type Item = K;
    fn next(&mut self) -> Option<K> {
        self.inner.next()
    }
}

impl<K, H, E, P, S> IntoIterator for SparseSet<K, H, E, P, S> {
    type Item = K;
    type IntoIter = IntoIter<K>;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.ht.into_values(),
        }
    }
}

impl<K, H, E, P, S> Extend<K> for SparseSet<K, H, E, P, S>
where
    H: HashKey<K> + Clone,
    E: EqKey<K, K> + Clone,
    P: GrowthPolicy,
    S: Sparsity,
{
    /// Insert every key from `iter`. Keys already present are ignored.
    fn extend<I: IntoIterator<Item = K>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        self.reserve(self.len() + lower);
        for k in iter {
            self.insert(k);
        }
    }
}

// Serialization.

impl<K, H, E, P, S> SparseSet<K, H, E, P, S>
where
    K: Serialize,
{
    /// Write the set through `serializer` in protocol order.
    pub fn serialize<Sz: Serializer>(&self, serializer: &mut Sz) {
        self.ht.serialize(serializer);
    }
}

impl<K, H, E, P, S> SparseSet<K, H, E, P, S>
where
    H: HashKey<K> + Clone,
    E: EqKey<K, K> + Clone,
    P: GrowthPolicy,
    S: Sparsity,
    K: Serialize + Deserialize,
{
    /// Read a set written by [`SparseSet::serialize`].
    pub fn deserialize_with<D: Deserializer>(
        deserializer: &mut D,
        hash_compatible: bool,
        hash: H,
        key_eq: E,
    ) -> Result<Self, DeserializeError> {
        Ok(Self {
            ht: SparseHash::deserialize(deserializer, hash_compatible, hash, key_eq)?,
        })
    }
}
