//! A memory-efficient open-addressing hash map.
//!
//! [`SparseMap`] stores each entry as a `(K, V)` pair inside sparse arrays. At
//! low load factor it uses far less memory than a flat table because empty
//! buckets cost about one bit each. Lookup stays fast.
//!
//! # Iterator value semantics
//!
//! Iteration yields `(&K, &V)`. The pair is never handed out mutably, so a key
//! cannot change under the map. To mutate a value in place, use
//! [`SparseMap::get_mut`], [`SparseMap::iter_mut`], or index assignment through
//! [`SparseMap::get_mut`].
//!
//! # Iterator invalidation
//!
//! `clear`, `rehash`, and `reserve` invalidate outstanding references. `insert`
//! and its relatives invalidate them only when an element is actually inserted.
//! `remove` and `pop_front` always invalidate them.

use core::borrow::Borrow;
use core::hash::{BuildHasher, Hash};
use core::marker::PhantomData;

use crate::growth_policy::{GrowthPolicy, LengthError, PowerOfTwo};
use crate::hasher::{EqKey, HashKey, StdEq, StdHash};
use crate::serialize::{Deserialize, DeserializeError, Deserializer, Serialize, Serializer};
use crate::sparse_hash::{
    KeySelect, SparseHash, DEFAULT_INIT_BUCKET_COUNT, DEFAULT_MAX_LOAD_FACTOR,
};
use crate::sparsity::{Medium, Sparsity};

/// Reads the key from a stored `(K, V)` pair.
pub struct PairKeySelect<K, V>(PhantomData<(K, V)>);

impl<K, V> KeySelect<(K, V)> for PairKeySelect<K, V> {
    type Key = K;
    #[inline]
    fn key(value: &(K, V)) -> &K {
        &value.0
    }
}

/// A hash map that trades a little insert speed for low memory use.
///
/// Type parameters:
/// - `K`, `V`: key and value types.
/// - `H`: the hasher. Defaults to [`StdHash`], which uses the standard
///   [`BuildHasher`]. A hasher yields a `usize` directly.
/// - `E`: the key comparator. Defaults to [`StdEq`].
/// - `P`: the growth policy. Defaults to [`PowerOfTwo`] with factor 2.
/// - `S`: the sparsity level. Defaults to [`Medium`].
pub struct SparseMap<K, V, H = StdHash, E = StdEq, P = PowerOfTwo<2>, S = Medium> {
    ht: SparseHash<(K, V), PairKeySelect<K, V>, H, E, P, S>,
}

impl<K, V> SparseMap<K, V, StdHash, StdEq, PowerOfTwo<2>, Medium>
where
    K: Hash + Eq,
{
    /// An empty map with no allocation.
    pub fn new() -> Self {
        Self::with_bucket_count(DEFAULT_INIT_BUCKET_COUNT)
    }

    /// An empty map sized for at least `bucket_count` buckets.
    ///
    /// # Panics
    ///
    /// Panics when `bucket_count` exceeds the policy maximum. Use
    /// [`SparseMap::try_with_bucket_count`] for the fallible form.
    pub fn with_bucket_count(bucket_count: usize) -> Self {
        Self::try_with_bucket_count(bucket_count).expect("bucket count within policy limit")
    }

    /// An empty map sized for at least `bucket_count` buckets, fallibly.
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

impl<K, V> Default for SparseMap<K, V, StdHash, StdEq, PowerOfTwo<2>, Medium>
where
    K: Hash + Eq,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, B, P, S> SparseMap<K, V, StdHash<B>, StdEq, P, S>
where
    K: Eq,
    B: BuildHasher + Default,
    P: GrowthPolicy,
    S: Sparsity,
    StdHash<B>: HashKey<K>,
{
    /// An empty map that hashes with `B` and uses policy `P` and sparsity `S`.
    ///
    /// # Panics
    ///
    /// Panics when `bucket_count` exceeds the policy maximum.
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

impl<K, V, H, E, P, S> SparseMap<K, V, H, E, P, S>
where
    H: HashKey<K> + Clone,
    E: EqKey<K, K> + Clone,
    P: GrowthPolicy,
    S: Sparsity,
{
    /// Build a map from explicit hasher, comparator, policy, and sparsity.
    ///
    /// # Panics
    ///
    /// Panics when `bucket_count` exceeds the policy maximum.
    pub fn with_parts(bucket_count: usize, hash: H, key_eq: E) -> Self {
        Self {
            ht: SparseHash::new(bucket_count, hash, key_eq, DEFAULT_MAX_LOAD_FACTOR)
                .expect("bucket count within policy limit"),
        }
    }

    /// Build a map fallibly from explicit parts.
    pub fn try_with_parts(bucket_count: usize, hash: H, key_eq: E) -> Result<Self, LengthError> {
        Ok(Self {
            ht: SparseHash::new(bucket_count, hash, key_eq, DEFAULT_MAX_LOAD_FACTOR)?,
        })
    }

    /// Number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.ht.len()
    }

    /// Whether the map holds no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ht.is_empty()
    }

    /// The largest number of entries the map can hold.
    #[inline]
    pub fn max_size(&self) -> usize {
        self.ht.max_size()
    }

    /// The logical bucket count. Zero for a fresh map.
    #[inline]
    pub fn bucket_count(&self) -> usize {
        self.ht.bucket_count()
    }

    /// The largest bucket count the map can hold.
    #[inline]
    pub fn max_bucket_count(&self) -> usize {
        self.ht.max_bucket_count()
    }

    /// Ratio of entries to buckets. Zero for an empty map.
    #[inline]
    pub fn load_factor(&self) -> f32 {
        self.ht.load_factor()
    }

    /// The maximum load factor before a grow.
    #[inline]
    pub fn max_load_factor(&self) -> f32 {
        self.ht.max_load_factor()
    }

    /// Set the maximum load factor, clamped to `[0.1, 0.8]`.
    pub fn set_max_load_factor(&mut self, ml: f32) {
        self.ht.set_max_load_factor(ml);
    }

    /// The hasher.
    #[inline]
    pub fn hash_function(&self) -> &H {
        self.ht.hash_function()
    }

    /// The key comparator.
    #[inline]
    pub fn key_eq(&self) -> &E {
        self.ht.key_eq()
    }

    /// Remove every entry. Keeps the bucket count.
    pub fn clear(&mut self) {
        self.ht.clear();
    }

    /// Grow so the map holds at least `count` buckets.
    pub fn rehash(&mut self, count: usize) {
        self.ht.rehash(count);
    }

    /// Reserve room for `count` entries without exceeding the load factor.
    pub fn reserve(&mut self, count: usize) {
        self.ht.reserve(count);
    }

    /// Insert `key` with `value`, keeping the existing value on a collision.
    ///
    /// Returns whether a new entry was created. On a collision the passed value
    /// is dropped and the stored value is left as it was. This matches the
    /// container contract where insert never overwrites.
    pub fn insert(&mut self, key: K, value: V) -> bool {
        self.ht.insert((key, value)).1
    }

    /// A reference to the value at `key`.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: ?Sized,
        H: HashKey<Q>,
        E: EqKey<K, Q>,
    {
        let hash = self.ht.hash_function().hash_key(key);
        self.ht.get(key, hash).map(|(_, v)| v)
    }

    /// A reference to the value at `key`, using a precomputed hash.
    ///
    /// The hash must equal `hash_function().hash_key(key)` or the result is
    /// unspecified.
    pub fn get_precalc<Q>(&self, key: &Q, hash: usize) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: ?Sized,
        H: HashKey<Q>,
        E: EqKey<K, Q>,
    {
        self.ht.get(key, hash).map(|(_, v)| v)
    }

    /// A mutable reference to the value at `key`.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: ?Sized,
        H: HashKey<Q>,
        E: EqKey<K, Q>,
    {
        let hash = self.ht.hash_function().hash_key(key);
        self.ht.get_mut(key, hash).map(|(_, v)| v)
    }

    /// A reference to the key-value pair at `key`.
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: ?Sized,
        H: HashKey<Q>,
        E: EqKey<K, Q>,
    {
        let hash = self.ht.hash_function().hash_key(key);
        self.ht.get(key, hash).map(|(k, v)| (k, v))
    }

    /// Whether `key` is present.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
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
    pub fn contains_key_precalc<Q>(&self, key: &Q, hash: usize) -> bool
    where
        K: Borrow<Q>,
        Q: ?Sized,
        H: HashKey<Q>,
        E: EqKey<K, Q>,
    {
        self.ht.contains(key, hash)
    }

    /// 1 when `key` is present, 0 otherwise.
    pub fn count<Q>(&self, key: &Q) -> usize
    where
        K: Borrow<Q>,
        Q: ?Sized,
        H: HashKey<Q>,
        E: EqKey<K, Q>,
    {
        usize::from(self.contains_key(key))
    }

    /// A reference to the value at `key`, or a panic when absent.
    ///
    /// Use [`SparseMap::get`] to handle a missing key without panicking.
    ///
    /// # Panics
    ///
    /// Panics when `key` is not present.
    pub fn at<Q>(&self, key: &Q) -> &V
    where
        K: Borrow<Q>,
        Q: ?Sized,
        H: HashKey<Q>,
        E: EqKey<K, Q>,
    {
        self.get(key).expect("couldn't find key")
    }

    /// A reference to the value at `key`, using a precomputed hash.
    ///
    /// # Panics
    ///
    /// Panics when `key` is not present.
    pub fn at_precalc<Q>(&self, key: &Q, hash: usize) -> &V
    where
        K: Borrow<Q>,
        Q: ?Sized,
        H: HashKey<Q>,
        E: EqKey<K, Q>,
    {
        self.get_precalc(key, hash).expect("couldn't find key")
    }

    /// Remove `key` and return its value.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: ?Sized,
        H: HashKey<Q>,
        E: EqKey<K, Q>,
    {
        let hash = self.ht.hash_function().hash_key(key);
        self.ht.remove(key, hash).map(|(_, v)| v)
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

    /// Remove `key` using a precomputed hash. Returns 1 when erased, 0 otherwise.
    pub fn erase_precalc<Q>(&mut self, key: &Q, hash: usize) -> usize
    where
        K: Borrow<Q>,
        Q: ?Sized,
        H: HashKey<Q>,
        E: EqKey<K, Q>,
    {
        self.ht.erase(key, hash)
    }

    /// Remove and return the first entry in iteration order.
    pub fn pop_front(&mut self) -> Option<(K, V)> {
        self.ht.remove_nth(0)
    }

    /// Remove `count` entries starting at iteration index `skip`.
    ///
    /// Entries are erased in iteration order. Erasing shifts the tail forward,
    /// so the same index is erased `count` times.
    pub fn erase_range(&mut self, skip: usize, count: usize) {
        for _ in 0..count {
            self.ht.remove_nth(skip);
        }
    }

    /// Remove every entry by draining in iteration order.
    pub fn erase_all(&mut self) {
        while self.ht.remove_nth(0).is_some() {}
    }
}

impl<K, V, H, E, P, S> SparseMap<K, V, H, E, P, S>
where
    H: HashKey<K> + Clone,
    E: EqKey<K, K> + Clone,
    P: GrowthPolicy,
    S: Sparsity,
{
    /// The value at `key`, inserting `V::default()` when absent.
    ///
    /// This is the index-access behavior of a map that default-inserts.
    pub fn entry_or_default(&mut self, key: K) -> &mut V
    where
        V: Default,
    {
        self.try_emplace(key, V::default).0
    }

    /// Insert only when `key` is absent, building the value on demand.
    ///
    /// Returns a reference to the value and whether it was newly inserted. The
    /// value closure runs only when the key is absent, so a redundant call does
    /// not build or consume a value.
    pub fn try_emplace<F>(&mut self, key: K, make: F) -> (&mut V, bool)
    where
        F: FnOnce() -> V,
    {
        let hash = self.ht.hash_function().hash_key(&key);
        if let Some(pos) = self.ht.find_position(&key, hash) {
            let (_k, v) = self.ht.value_at_mut(pos);
            return (v, false);
        }
        let value = make();
        let (pos, _inserted) = self.ht.insert((key, value));
        let (_k, v) = self.ht.value_at_mut(pos);
        (v, true)
    }

    /// Insert `value` at `key`, or overwrite the existing value.
    ///
    /// Returns a reference to the value and whether it was newly inserted.
    pub fn insert_or_assign(&mut self, key: K, value: V) -> (&mut V, bool) {
        let hash = self.ht.hash_function().hash_key(&key);
        if let Some(pos) = self.ht.find_position(&key, hash) {
            let (_k, v) = self.ht.value_at_mut(pos);
            *v = value;
            return (v, false);
        }
        let (pos, _inserted) = self.ht.insert((key, value));
        let (_k, v) = self.ht.value_at_mut(pos);
        (v, true)
    }
}

// Iteration.

impl<K, V, H, E, P, S> SparseMap<K, V, H, E, P, S> {
    /// A forward iterator over `(&K, &V)` pairs.
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            inner: self.ht.iter(),
        }
    }

    /// A forward iterator over `(&K, &mut V)` pairs.
    ///
    /// The key stays shared so it cannot change under the map.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            inner: self.ht.iter_mut(),
        }
    }

    /// A forward iterator over keys.
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys {
            inner: self.ht.iter(),
        }
    }

    /// A forward iterator over shared value references.
    pub fn values(&self) -> Values<'_, K, V> {
        Values {
            inner: self.ht.iter(),
        }
    }
}

/// Iterator over `(&K, &V)` pairs of a [`SparseMap`].
pub struct Iter<'a, K, V> {
    inner: crate::sparse_hash::Iter<'a, (K, V)>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| (k, v))
    }
}

/// Iterator over `(&K, &mut V)` pairs of a [`SparseMap`].
pub struct IterMut<'a, K, V> {
    inner: crate::sparse_hash::IterMut<'a, (K, V)>,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| (&*k, v))
    }
}

/// Iterator over keys of a [`SparseMap`].
pub struct Keys<'a, K, V> {
    inner: crate::sparse_hash::Iter<'a, (K, V)>,
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, _)| k)
    }
}

/// Iterator over values of a [`SparseMap`].
pub struct Values<'a, K, V> {
    inner: crate::sparse_hash::Iter<'a, (K, V)>,
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }
}

impl<'a, K, V, H, E, P, S> IntoIterator for &'a SparseMap<K, V, H, E, P, S> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Equality. Order-independent. Compares keys through lookup and values with `==`.

impl<K, V, H, E, P, S> PartialEq for SparseMap<K, V, H, E, P, S>
where
    V: PartialEq,
    H: HashKey<K> + Clone,
    E: EqKey<K, K> + Clone,
    P: GrowthPolicy,
    S: Sparsity,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        for (k, v) in self.iter() {
            match other.get(k) {
                Some(ov) if ov == v => {}
                _ => return false,
            }
        }
        true
    }
}

impl<K, V, H, E, P, S> Eq for SparseMap<K, V, H, E, P, S>
where
    V: Eq,
    H: HashKey<K> + Clone,
    E: EqKey<K, K> + Clone,
    P: GrowthPolicy,
    S: Sparsity,
{
}

impl<K, V, H, E, P, S> core::fmt::Debug for SparseMap<K, V, H, E, P, S>
where
    K: core::fmt::Debug,
    V: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V, H, E, P, S> Clone for SparseMap<K, V, H, E, P, S>
where
    (K, V): Clone,
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

// Serialization.

impl<K, V, H, E, P, S> SparseMap<K, V, H, E, P, S>
where
    (K, V): Serialize,
{
    /// Write the map through `serializer` in protocol order.
    pub fn serialize<Sz: Serializer>(&self, serializer: &mut Sz) {
        self.ht.serialize(serializer);
    }
}

impl<K, V, H, E, P, S> SparseMap<K, V, H, E, P, S>
where
    H: HashKey<K> + Clone,
    E: EqKey<K, K> + Clone,
    P: GrowthPolicy,
    S: Sparsity,
    (K, V): Serialize + Deserialize,
{
    /// Read a map written by [`SparseMap::serialize`].
    ///
    /// See the engine docs for the meaning of `hash_compatible`.
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
