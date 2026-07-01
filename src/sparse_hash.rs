//! The shared open-addressing engine behind the map and the set.
//!
//! Values are stored in sparse arrays. A hash picks a starting bucket, and
//! quadratic probing walks forward on collision. Erase leaves a tombstone rather
//! than a hole, so probing must continue past tombstones to decide a key is
//! absent. The first tombstone seen during an insert probe is reused as the
//! insertion site so deletions are reclaimed.
//!
//! The engine is generic over how a stored value exposes its key. The map stores
//! `(K, V)` and reads the key from the first field. The set stores `K` and reads
//! it directly. This mirrors the key-select idea without duplicating the engine.

use core::marker::PhantomData;

use crate::growth_policy::{GrowthPolicy, LengthError};
use crate::hasher::{EqKey, HashKey};
use crate::serialize::{
    Deserialize, DeserializeError, Deserializer, Serialize, Serializer,
    SERIALIZATION_PROTOCOL_VERSION,
};
use crate::sparse_array::{
    index_in_sparse_bucket, nb_sparse_buckets, popcount_bitmap, sparse_ibucket, Bitmap,
    SparseArray, BITMAP_NB_BITS,
};
use crate::sparsity::Sparsity;

/// Default bucket count for a fresh container. No allocation until first insert.
pub const DEFAULT_INIT_BUCKET_COUNT: usize = 0;
/// Default maximum load factor before a grow.
pub const DEFAULT_MAX_LOAD_FACTOR: f32 = 0.5;

/// The largest logical bucket count the bucket vector can hold.
///
/// A request above this is a length error regardless of the growth policy.
const MAX_BUCKET_COUNT: usize = isize::MAX as usize;

/// How a stored value exposes its lookup key.
///
/// The map implements this to read `pair.0`. The set implements it as identity.
pub trait KeySelect<T> {
    /// The key type used for hashing and equality.
    type Key;
    /// Borrow the key out of a stored value.
    fn key(value: &T) -> &Self::Key;
}

/// The core engine. Holds sparse arrays plus the hasher, comparator, and policy.
pub struct SparseHash<T, K, H, E, P, S> {
    sparse_buckets: Vec<SparseArray<T>>,
    bucket_count: usize,
    nb_elements: usize,
    nb_deleted_buckets: usize,
    load_threshold_rehash: usize,
    load_threshold_clear_deleted: usize,
    max_load_factor: f32,
    hash: H,
    key_eq: E,
    policy: P,
    _key: PhantomData<K>,
    _sparsity: PhantomData<S>,
}

impl<T, K, H, E, P, S> SparseHash<T, K, H, E, P, S>
where
    P: GrowthPolicy,
{
    /// Build an engine sized for at least `bucket_count` buckets.
    ///
    /// Returns [`LengthError`] when the request exceeds the policy limit.
    pub fn new(
        bucket_count: usize,
        hash: H,
        key_eq: E,
        max_load_factor: f32,
    ) -> Result<Self, LengthError> {
        let (policy, settled) = P::new(bucket_count)?;

        // The bucket vector cannot exceed this many entries. A request past it
        // is a length error even when the policy would allow it.
        if settled > MAX_BUCKET_COUNT {
            return Err(LengthError);
        }

        let mut sparse_buckets = Vec::new();
        if settled > 0 {
            let n = nb_sparse_buckets(settled);
            sparse_buckets.reserve_exact(n);
            for _ in 0..n {
                sparse_buckets.push(SparseArray::new());
            }
            sparse_buckets
                .last_mut()
                .expect("non-empty by construction")
                .set_as_last();
        }

        let mut this = Self {
            sparse_buckets,
            bucket_count: settled,
            nb_elements: 0,
            nb_deleted_buckets: 0,
            load_threshold_rehash: 0,
            load_threshold_clear_deleted: 0,
            max_load_factor: DEFAULT_MAX_LOAD_FACTOR,
            hash,
            key_eq,
            policy,
            _key: PhantomData,
            _sparsity: PhantomData,
        };
        this.set_max_load_factor(max_load_factor);
        Ok(this)
    }

    /// Number of stored elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.nb_elements
    }

    /// Whether the container holds no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nb_elements == 0
    }

    /// The logical bucket count.
    #[inline]
    pub fn bucket_count(&self) -> usize {
        self.bucket_count
    }

    /// The largest bucket count the bucket vector can hold.
    #[inline]
    pub fn max_bucket_count(&self) -> usize {
        MAX_BUCKET_COUNT
    }

    /// The largest number of elements the container can hold.
    #[inline]
    pub fn max_size(&self) -> usize {
        MAX_BUCKET_COUNT
    }

    /// Ratio of elements to buckets. Zero for an empty table.
    #[inline]
    pub fn load_factor(&self) -> f32 {
        if self.bucket_count == 0 {
            return 0.0;
        }
        self.nb_elements as f32 / self.bucket_count as f32
    }

    /// The current maximum load factor.
    #[inline]
    pub fn max_load_factor(&self) -> f32 {
        self.max_load_factor
    }

    /// Set the maximum load factor, clamped to `[0.1, 0.8]`.
    ///
    /// Recomputes the grow threshold and the tombstone-cleanup threshold.
    /// Thresholds truncate toward zero.
    pub fn set_max_load_factor(&mut self, ml: f32) {
        self.max_load_factor = 0.1_f32.max(ml.min(0.8));
        self.load_threshold_rehash = (self.bucket_count as f32 * self.max_load_factor) as usize;
        let mlf_with_deleted = self.max_load_factor + 0.5 * (1.0 - self.max_load_factor);
        self.load_threshold_clear_deleted = (self.bucket_count as f32 * mlf_with_deleted) as usize;
    }

    /// The hasher.
    #[inline]
    pub fn hash_function(&self) -> &H {
        &self.hash
    }

    /// The key comparator.
    #[inline]
    pub fn key_eq(&self) -> &E {
        &self.key_eq
    }

    #[inline]
    fn hash_key<Q>(&self, key: &Q) -> usize
    where
        H: HashKey<Q>,
        Q: ?Sized,
    {
        self.hash.hash_key(key)
    }

    #[inline]
    fn bucket_for_hash(&self, hash: usize) -> usize {
        self.policy.bucket_for_hash(hash)
    }

    #[inline]
    fn next_bucket(&self, ibucket: usize, iprobe: usize) -> usize {
        if P::is_power_of_two() {
            // Mask arithmetic. bucket_count is a power of two here.
            (ibucket.wrapping_add(iprobe)) & (self.bucket_count - 1)
        } else {
            let next = ibucket + iprobe;
            if next < self.bucket_count {
                next
            } else {
                next % self.bucket_count
            }
        }
    }

    /// Remove every element. Keeps the bucket count.
    pub fn clear(&mut self) {
        for bucket in &mut self.sparse_buckets {
            let last = bucket.last();
            *bucket = SparseArray::new();
            if last {
                bucket.set_as_last();
            }
        }
        self.nb_elements = 0;
        self.nb_deleted_buckets = 0;
    }
}

/// Where an element lives inside the bucket vector.
#[derive(Clone, Copy)]
pub struct Position {
    /// Index into the bucket vector.
    pub sparse_ibucket: usize,
    /// Logical index within that sparse array.
    pub index: u8,
}

impl<T, K, H, E, P, S> SparseHash<T, K, H, E, P, S>
where
    K: KeySelect<T>,
    H: HashKey<K::Key> + Clone,
    E: EqKey<K::Key, K::Key> + Clone,
    P: GrowthPolicy,
    S: Sparsity,
{
    /// Find the position of `key`, or `None`.
    pub fn find_position<Q>(&self, key: &Q, hash: usize) -> Option<Position>
    where
        H: HashKey<Q>,
        E: EqKey<K::Key, Q>,
        Q: ?Sized,
    {
        if self.bucket_count == 0 {
            return None;
        }
        let mut ibucket = self.bucket_for_hash(hash);
        let mut probe = 0usize;
        loop {
            let sib = sparse_ibucket(ibucket);
            let idx = index_in_sparse_bucket(ibucket);
            let bucket = &self.sparse_buckets[sib];

            if bucket.has_value(idx) {
                if self.key_eq.eq_key(K::key(bucket.value(idx)), key) {
                    return Some(Position {
                        sparse_ibucket: sib,
                        index: idx,
                    });
                }
            } else if !bucket.has_deleted_value(idx) || probe >= self.bucket_count {
                return None;
            }

            probe += 1;
            ibucket = self.next_bucket(ibucket, probe);
        }
    }

    /// A shared reference to the value at `key`.
    pub fn get<Q>(&self, key: &Q, hash: usize) -> Option<&T>
    where
        H: HashKey<Q>,
        E: EqKey<K::Key, Q>,
        Q: ?Sized,
    {
        self.find_position(key, hash)
            .map(|p| self.sparse_buckets[p.sparse_ibucket].value(p.index))
    }

    /// A mutable reference to the value at `key`.
    pub fn get_mut<Q>(&mut self, key: &Q, hash: usize) -> Option<&mut T>
    where
        H: HashKey<Q>,
        E: EqKey<K::Key, Q>,
        Q: ?Sized,
    {
        let pos = self.find_position(key, hash)?;
        Some(self.sparse_buckets[pos.sparse_ibucket].value_mut(pos.index))
    }

    /// Whether `key` is present.
    pub fn contains<Q>(&self, key: &Q, hash: usize) -> bool
    where
        H: HashKey<Q>,
        E: EqKey<K::Key, Q>,
        Q: ?Sized,
    {
        self.find_position(key, hash).is_some()
    }

    /// Insert `value`. Returns the position and whether it was newly inserted.
    ///
    /// The key is read from `value`. When the key already exists, `value` is
    /// dropped and the existing element is returned with `false`.
    pub fn insert(&mut self, value: T) -> (Position, bool)
    where
        K::Key: Sized,
    {
        let hash = self.hash_key(K::key(&value));
        self.insert_with_hash(value, hash)
    }

    /// Insert `value` with a precomputed hash of its key.
    ///
    /// The hash must equal `hash_function().hash_key(K::key(&value))`. Callers
    /// that already computed the hash for a lookup pass it here to avoid a second
    /// hash on the insert path.
    pub fn insert_with_hash(&mut self, value: T, hash: usize) -> (Position, bool) {
        loop {
            // A fresh table has no sparse arrays. The first probe would land on
            // the shared empty slot, which confirms the key is absent and, with
            // size at the rehash threshold, grows the table. Reproduce that here
            // so the probe below always has an array to read.
            if self.sparse_buckets.is_empty() {
                let count = self
                    .policy
                    .next_bucket_count()
                    .expect("grow within policy limit");
                self.rehash_impl(count);
                continue;
            }

            let mut found_deleted: Option<(usize, u8)> = None;
            let mut ibucket = self.bucket_for_hash(hash);
            let mut probe = 0usize;

            loop {
                let sib = sparse_ibucket(ibucket);
                let idx = index_in_sparse_bucket(ibucket);
                let bucket = &self.sparse_buckets[sib];

                if bucket.has_value(idx) {
                    if self
                        .key_eq
                        .eq_key(K::key(bucket.value(idx)), K::key(&value))
                    {
                        return (
                            Position {
                                sparse_ibucket: sib,
                                index: idx,
                            },
                            false,
                        );
                    }
                } else if bucket.has_deleted_value(idx) && probe < self.bucket_count {
                    if found_deleted.is_none() {
                        found_deleted = Some((sib, idx));
                    }
                } else {
                    // Empty slot: the key is absent. Check thresholds first.
                    if self.nb_elements >= self.load_threshold_rehash {
                        let count = self
                            .policy
                            .next_bucket_count()
                            .expect("grow within policy limit");
                        self.rehash_impl(count);
                        break;
                    } else if self.nb_elements + self.nb_deleted_buckets
                        >= self.load_threshold_clear_deleted
                    {
                        self.clear_deleted_buckets();
                        break;
                    }

                    if let Some((dsib, didx)) = found_deleted {
                        let pos = self.insert_in_bucket(dsib, didx, value);
                        self.nb_deleted_buckets -= 1;
                        return (pos, true);
                    }
                    let pos = self.insert_in_bucket(sib, idx, value);
                    return (pos, true);
                }

                probe += 1;
                ibucket = self.next_bucket(ibucket, probe);
            }
        }
    }

    fn insert_in_bucket(&mut self, sib: usize, index: u8, value: T) -> Position {
        self.sparse_buckets[sib].set(index, value, S::STEP as usize);
        self.nb_elements += 1;
        Position {
            sparse_ibucket: sib,
            index,
        }
    }

    /// Remove `key` and return the removed value.
    pub fn remove<Q>(&mut self, key: &Q, hash: usize) -> Option<T>
    where
        H: HashKey<Q>,
        E: EqKey<K::Key, Q>,
        Q: ?Sized,
    {
        let pos = self.find_position(key, hash)?;
        let sib = pos.sparse_ibucket;
        let offset = self.sparse_buckets[sib].index_to_offset(pos.index);
        let value = self.sparse_buckets[sib].remove_value(offset, pos.index);
        self.nb_elements -= 1;
        self.nb_deleted_buckets += 1;
        Some(value)
    }

    /// Erase `key`. Returns 1 when found and erased, 0 otherwise.
    pub fn erase<Q>(&mut self, key: &Q, hash: usize) -> usize
    where
        H: HashKey<Q>,
        E: EqKey<K::Key, Q>,
        Q: ?Sized,
    {
        if self.bucket_count == 0 {
            return 0;
        }
        let mut ibucket = self.bucket_for_hash(hash);
        let mut probe = 0usize;
        loop {
            let sib = sparse_ibucket(ibucket);
            let idx = index_in_sparse_bucket(ibucket);
            let bucket = &self.sparse_buckets[sib];

            if bucket.has_value(idx) {
                if self.key_eq.eq_key(K::key(bucket.value(idx)), key) {
                    let offset = self.sparse_buckets[sib].index_to_offset(idx);
                    self.sparse_buckets[sib].remove_value(offset, idx);
                    self.nb_elements -= 1;
                    self.nb_deleted_buckets += 1;
                    return 1;
                }
            } else if !bucket.has_deleted_value(idx) || probe >= self.bucket_count {
                return 0;
            }

            probe += 1;
            ibucket = self.next_bucket(ibucket, probe);
        }
    }

    fn clear_deleted_buckets(&mut self) {
        self.rehash_impl(self.bucket_count);
        debug_assert_eq!(self.nb_deleted_buckets, 0);
    }

    /// Remove and return the value at iteration position `n`, counting from the
    /// first live element. Leaves a tombstone. Returns `None` when out of range.
    ///
    /// This backs positional erase used by range and loop erasure.
    pub fn remove_nth(&mut self, n: usize) -> Option<T> {
        let (mut sib, mut offset) = self.first_position()?;
        for _ in 0..n {
            let (s, o) = self.next_position(sib, offset)?;
            sib = s;
            offset = o;
        }
        let index = self.sparse_buckets[sib].offset_to_index(offset);
        let value = self.sparse_buckets[sib].remove_value(offset, index);
        self.nb_elements -= 1;
        self.nb_deleted_buckets += 1;
        Some(value)
    }

    /// Erase every element in one pass, leaving tombstones.
    ///
    /// Walks each sparse array once and tombstones its present slots, so the
    /// cost is linear in the array count plus the element count. Draining with
    /// `remove_nth(0)` in a loop would rescan from the front each time.
    pub fn erase_all(&mut self) {
        for bucket in &mut self.sparse_buckets {
            let removed = bucket.erase_all();
            self.nb_deleted_buckets += removed;
        }
        self.nb_elements = 0;
    }

    /// Erase `count` elements starting at iteration index `skip`, leaving
    /// tombstones.
    ///
    /// Advances one cursor to `skip`, then erases consecutive live slots while
    /// walking forward. Erasing at a dense offset shifts the tail of that array
    /// down by one, so the next live slot sits at the same offset until the
    /// array runs out. Stops early when the table ends.
    pub fn erase_range(&mut self, skip: usize, count: usize) {
        if count == 0 {
            return;
        }
        let Some((mut sib, mut offset)) = self.first_position() else {
            return;
        };
        for _ in 0..skip {
            match self.next_position(sib, offset) {
                Some((s, o)) => {
                    sib = s;
                    offset = o;
                }
                None => return,
            }
        }
        for _ in 0..count {
            while offset >= self.sparse_buckets[sib].len() {
                let mut n = sib + 1;
                while n < self.sparse_buckets.len() && self.sparse_buckets[n].is_empty() {
                    n += 1;
                }
                if n >= self.sparse_buckets.len() {
                    return;
                }
                sib = n;
                offset = 0;
            }
            let index = self.sparse_buckets[sib].offset_to_index(offset);
            self.sparse_buckets[sib].remove_value(offset, index);
            self.nb_elements -= 1;
            self.nb_deleted_buckets += 1;
        }
    }

    /// Rebuild the table with `count` buckets and re-insert every element.
    fn rehash_impl(&mut self, count: usize) {
        let mut new_table = SparseHash::<T, K, H, E, P, S>::new(
            count,
            self.hash.clone(),
            self.key_eq.clone(),
            self.max_load_factor,
        )
        .expect("rehash target within policy limit");

        let old = core::mem::take(&mut self.sparse_buckets);
        for bucket in old {
            for value in bucket.into_values() {
                new_table.insert_on_rehash(value);
            }
        }

        core::mem::swap(self, &mut new_table);
    }

    fn insert_on_rehash(&mut self, value: T) {
        let hash = self.hash_key(K::key(&value));
        let mut ibucket = self.bucket_for_hash(hash);
        let mut probe = 0usize;
        loop {
            let sib = sparse_ibucket(ibucket);
            let idx = index_in_sparse_bucket(ibucket);
            if !self.sparse_buckets[sib].has_value(idx) {
                self.sparse_buckets[sib].set(idx, value, S::STEP as usize);
                self.nb_elements += 1;
                return;
            }
            probe += 1;
            ibucket = self.next_bucket(ibucket, probe);
        }
    }

    /// Rehash so the table holds at least `count` buckets.
    pub fn rehash(&mut self, count: usize) {
        let needed = (self.len() as f32 / self.max_load_factor()).ceil() as usize;
        let count = count.max(needed);
        self.rehash_impl(count);
    }

    /// Reserve room for `count` elements without exceeding the load factor.
    pub fn reserve(&mut self, count: usize) {
        let buckets = (count as f32 / self.max_load_factor()).ceil() as usize;
        self.rehash(buckets);
    }
}

impl<T, K, H, E, P, S> SparseHash<T, K, H, E, P, S> {
    /// A mutable reference to the value at a known position.
    #[inline]
    pub fn value_at_mut(&mut self, pos: Position) -> &mut T {
        self.sparse_buckets[pos.sparse_ibucket].value_mut(pos.index)
    }

    /// Keep only the values for which `keep` returns true.
    ///
    /// Removed values become tombstones, reclaimed on the next grow or cleanup.
    /// Each array is scanned once. Within an array the dense block shifts as
    /// slots are removed, so the scan tracks the current offset explicitly.
    pub fn retain<F>(&mut self, mut keep: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        for array in &mut self.sparse_buckets {
            let mut offset = 0;
            while offset < array.len() {
                let index = array.offset_to_index(offset);
                if keep(&mut array.values_mut()[offset]) {
                    offset += 1;
                } else {
                    array.remove_value(offset, index);
                    self.nb_elements -= 1;
                    self.nb_deleted_buckets += 1;
                }
            }
        }
    }
}

// Iteration support. A forward cursor over (sparse array, dense offset).

impl<T, K, H, E, P, S> SparseHash<T, K, H, E, P, S> {
    /// The first occupied position, or `None` when empty.
    pub fn first_position(&self) -> Option<(usize, usize)> {
        let mut sib = 0;
        while sib < self.sparse_buckets.len() {
            if !self.sparse_buckets[sib].is_empty() {
                return Some((sib, 0));
            }
            sib += 1;
        }
        None
    }

    /// The position after `(sib, offset)`, or `None` at the end.
    pub fn next_position(&self, sib: usize, offset: usize) -> Option<(usize, usize)> {
        let next_offset = offset + 1;
        if next_offset < self.sparse_buckets[sib].len() {
            return Some((sib, next_offset));
        }
        let mut n = sib + 1;
        while n < self.sparse_buckets.len() && self.sparse_buckets[n].is_empty() {
            n += 1;
        }
        if n < self.sparse_buckets.len() {
            Some((n, 0))
        } else {
            None
        }
    }

    /// A forward iterator over shared references to every value.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            buckets: &self.sparse_buckets,
            pos: self.first_position(),
        }
    }

    /// Consume the table and yield every value by move in iteration order.
    pub fn into_values(self) -> IntoIter<T> {
        let mut arrays = self.sparse_buckets.into_iter();
        let inner = match arrays.next() {
            Some(a) => a.into_values().into_iter(),
            None => Vec::new().into_iter(),
        };
        IntoIter {
            inner,
            remaining_arrays: arrays,
        }
    }
}

/// A forward iterator over shared references to stored values.
pub struct Iter<'a, T> {
    buckets: &'a [SparseArray<T>],
    pos: Option<(usize, usize)>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        let (sib, offset) = self.pos?;
        let value = &self.buckets[sib].values()[offset];
        // Advance.
        let next_offset = offset + 1;
        self.pos = if next_offset < self.buckets[sib].len() {
            Some((sib, next_offset))
        } else {
            let mut n = sib + 1;
            while n < self.buckets.len() && self.buckets[n].is_empty() {
                n += 1;
            }
            if n < self.buckets.len() {
                Some((n, 0))
            } else {
                None
            }
        };
        Some(value)
    }
}

/// A forward iterator that moves every value out of a consumed table.
///
/// The cursor drains each sparse array's dense block in index order, then moves
/// to the next non-empty array.
pub struct IntoIter<T> {
    inner: std::vec::IntoIter<T>,
    remaining_arrays: std::vec::IntoIter<SparseArray<T>>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        loop {
            if let Some(v) = self.inner.next() {
                return Some(v);
            }
            let array = self.remaining_arrays.next()?;
            self.inner = array.into_values().into_iter();
        }
    }
}

/// A forward iterator over mutable references to stored values.
///
/// The cursor is a flat scan over the dense storage of each array in order.
pub struct IterMut<'a, T> {
    inner: std::slice::IterMut<'a, T>,
    remaining_buckets: std::slice::IterMut<'a, SparseArray<T>>,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        loop {
            if let Some(v) = self.inner.next() {
                return Some(v);
            }
            let bucket = self.remaining_buckets.next()?;
            self.inner = bucket.values_mut().iter_mut();
        }
    }
}

impl<T, K, H, E, P, S> SparseHash<T, K, H, E, P, S> {
    /// A forward iterator over mutable references to every value.
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        let mut buckets = self.sparse_buckets.iter_mut();
        let inner = match buckets.next() {
            Some(b) => b.values_mut().iter_mut(),
            None => [].iter_mut(),
        };
        IterMut {
            inner,
            remaining_buckets: buckets,
        }
    }
}

// Serialization.

impl<T, K, H, E, P, S> SparseHash<T, K, H, E, P, S>
where
    T: Serialize,
{
    /// Write the table through `serializer` in protocol order.
    pub fn serialize<Sz: Serializer>(&self, serializer: &mut Sz) {
        serializer.write_u64(SERIALIZATION_PROTOCOL_VERSION);
        serializer.write_u64(self.bucket_count as u64);
        serializer.write_u64(self.sparse_buckets.len() as u64);
        serializer.write_u64(self.nb_elements as u64);
        serializer.write_u64(self.nb_deleted_buckets as u64);
        serializer.write_f32(self.max_load_factor);

        for bucket in &self.sparse_buckets {
            serializer.write_u64(bucket.len() as u64);
            // Bitmap is u64 on 64-bit and u32 on 32-bit. The conversion widens
            // it to the fixed wire width on 32-bit targets.
            #[allow(clippy::useless_conversion)]
            serializer.write_u64(u64::from(bucket.bitmap_vals()));
            #[allow(clippy::useless_conversion)]
            serializer.write_u64(u64::from(bucket.bitmap_deleted_vals()));
            for value in bucket.values() {
                value.serialize(serializer);
            }
        }
    }
}

impl<T, K, H, E, P, S> SparseHash<T, K, H, E, P, S>
where
    K: KeySelect<T>,
    H: HashKey<K::Key> + Clone,
    E: EqKey<K::Key, K::Key> + Clone,
    P: GrowthPolicy,
    S: Sparsity,
    T: Deserialize + Serialize,
    K::Key: Sized,
{
    /// Read a table written by [`Self::serialize`].
    ///
    /// With `hash_compatible` false the values are re-hashed and re-inserted,
    /// which is safe across differing hashers and policies. With it true the
    /// bitmaps and dense values are restored directly, which requires the same
    /// hasher and policy that wrote the file.
    pub fn deserialize<D: Deserializer>(
        deserializer: &mut D,
        hash_compatible: bool,
        hash: H,
        key_eq: E,
    ) -> Result<Self, DeserializeError> {
        let version = deserializer.read_u64();
        if version != SERIALIZATION_PROTOCOL_VERSION {
            return Err(DeserializeError(
                "the serialization protocol version header is invalid",
            ));
        }

        let bucket_count_ds = deserializer.read_u64() as usize;
        let nb_sparse = deserializer.read_u64() as usize;
        let nb_elements = deserializer.read_u64() as usize;
        let nb_deleted = deserializer.read_u64() as usize;
        let max_load_factor = deserializer.read_f32();

        if !hash_compatible {
            let mut table = Self::new(0, hash, key_eq, DEFAULT_MAX_LOAD_FACTOR)
                .map_err(|_| DeserializeError("bucket count too big"))?;
            table.set_max_load_factor(max_load_factor);
            table.reserve(nb_elements);
            for _ in 0..nb_sparse {
                let sparse_bucket_size = deserializer.read_u64() as usize;
                let _bitmap_vals = deserializer.read_u64();
                let _bitmap_deleted = deserializer.read_u64();
                for _ in 0..sparse_bucket_size {
                    let value = T::deserialize(deserializer);
                    table.insert(value);
                }
            }
            Ok(table)
        } else {
            let (policy, settled) =
                P::new(bucket_count_ds).map_err(|_| DeserializeError("bucket count too big"))?;
            if settled != bucket_count_ds {
                return Err(DeserializeError(
                    "the growth policy is not the same even though hash_compatible is true",
                ));
            }
            if bucket_count_ds > MAX_BUCKET_COUNT {
                return Err(DeserializeError("bucket count too big"));
            }
            if nb_sparse != nb_sparse_buckets(bucket_count_ds) {
                return Err(DeserializeError(
                    "deserialized nb_sparse_buckets is invalid",
                ));
            }

            // Grow the vector as arrays are read. Do not pre-allocate from
            // `nb_sparse`, which is derived from an untrusted bucket count and
            // could request an allocation that aborts the process.
            let mut sparse_buckets = Vec::new();
            for _ in 0..nb_sparse {
                let sparse_bucket_size = deserializer.read_u64() as usize;
                let bitmap_vals = deserializer.read_u64();
                let bitmap_deleted = deserializer.read_u64();
                if sparse_bucket_size > BITMAP_NB_BITS {
                    return Err(DeserializeError(
                        "deserialized sparse_bucket_size is too big for the platform",
                    ));
                }
                let bitmap_vals: Bitmap = numeric_cast_bitmap(bitmap_vals)
                    .ok_or(DeserializeError("deserialized bitmap_vals is too big"))?;
                let bitmap_deleted: Bitmap = numeric_cast_bitmap(bitmap_deleted).ok_or(
                    DeserializeError("deserialized bitmap_deleted_vals is too big"),
                )?;
                // The bitmap and the value count are independent fields. Reject a
                // file where the popcount disagrees with the value count, or
                // where a slot is both present and a tombstone. Later lookups
                // derive dense offsets from the bitmap, so a mismatch would index
                // past the value block.
                if popcount_bitmap(bitmap_vals) as usize != sparse_bucket_size {
                    return Err(DeserializeError(
                        "deserialized bitmap_vals popcount does not match the value count",
                    ));
                }
                if bitmap_vals & bitmap_deleted != 0 {
                    return Err(DeserializeError(
                        "a deserialized slot is both present and a tombstone",
                    ));
                }
                let mut values = Vec::with_capacity(sparse_bucket_size);
                for _ in 0..sparse_bucket_size {
                    values.push(T::deserialize(deserializer));
                }
                sparse_buckets.push(SparseArray::from_parts(bitmap_vals, bitmap_deleted, values));
            }
            if let Some(last) = sparse_buckets.last_mut() {
                last.set_as_last();
            }

            let mut table = Self {
                sparse_buckets,
                bucket_count: bucket_count_ds,
                nb_elements,
                nb_deleted_buckets: nb_deleted,
                load_threshold_rehash: 0,
                load_threshold_clear_deleted: 0,
                max_load_factor: DEFAULT_MAX_LOAD_FACTOR,
                hash,
                key_eq,
                policy,
                _key: PhantomData,
                _sparsity: PhantomData,
            };
            table.set_max_load_factor(max_load_factor);
            if table.load_factor() > table.max_load_factor() {
                return Err(DeserializeError(
                    "invalid max_load_factor after deserialization",
                ));
            }
            Ok(table)
        }
    }
}

#[inline]
fn numeric_cast_bitmap(value: u64) -> Option<Bitmap> {
    let cast = value as Bitmap;
    // The widening is a no-op on 64-bit and meaningful on 32-bit.
    #[allow(clippy::useless_conversion)]
    let round_trip = u64::from(cast);
    if round_trip == value {
        Some(cast)
    } else {
        None
    }
}

impl<T, K, H, E, P, S> Clone for SparseHash<T, K, H, E, P, S>
where
    T: Clone,
    H: Clone,
    E: Clone,
    P: Clone,
{
    fn clone(&self) -> Self {
        Self {
            sparse_buckets: self.sparse_buckets.clone(),
            bucket_count: self.bucket_count,
            nb_elements: self.nb_elements,
            nb_deleted_buckets: self.nb_deleted_buckets,
            load_threshold_rehash: self.load_threshold_rehash,
            load_threshold_clear_deleted: self.load_threshold_clear_deleted,
            max_load_factor: self.max_load_factor,
            hash: self.hash.clone(),
            key_eq: self.key_eq.clone(),
            policy: self.policy.clone(),
            _key: PhantomData,
            _sparsity: PhantomData,
        }
    }
}
