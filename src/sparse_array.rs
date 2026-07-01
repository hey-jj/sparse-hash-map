//! One sparse array: a group of up to 64 logical buckets backed by a bitmap.
//!
//! Instead of one flat slot per bucket, buckets are grouped into arrays of
//! [`BITMAP_NB_BITS`] logical indices. An array stores only the present values,
//! packed contiguously, plus a bitmap of which indices hold a value and a bitmap
//! of which indices are tombstones. The dense offset of a logical index is the
//! number of value bits below it. An empty logical bucket costs about one bit,
//! not the size of a value.

#[cfg(not(target_pointer_width = "64"))]
use crate::popcount::popcount;
#[cfg(target_pointer_width = "64")]
use crate::popcount::popcountll;

/// Number of logical buckets per sparse array.
///
/// 64 on 64-bit targets, 32 on 32-bit targets. Popcount on 64-bit words is slow
/// on 32-bit machines, so a narrower bitmap is used there.
#[cfg(target_pointer_width = "64")]
pub const BITMAP_NB_BITS: usize = 64;
/// Number of logical buckets per sparse array.
#[cfg(not(target_pointer_width = "64"))]
pub const BITMAP_NB_BITS: usize = 32;

#[cfg(target_pointer_width = "64")]
const BUCKET_SHIFT: usize = 6;
#[cfg(not(target_pointer_width = "64"))]
const BUCKET_SHIFT: usize = 5;

const BUCKET_MASK: usize = BITMAP_NB_BITS - 1;

/// The bitmap word. Wide enough to hold [`BITMAP_NB_BITS`] bits.
#[cfg(target_pointer_width = "64")]
pub type Bitmap = u64;
/// The bitmap word.
#[cfg(not(target_pointer_width = "64"))]
pub type Bitmap = u32;

/// Which sparse array holds a global bucket index.
#[inline]
pub fn sparse_ibucket(ibucket: usize) -> usize {
    ibucket >> BUCKET_SHIFT
}

/// The logical index within its sparse array for a global bucket index.
#[inline]
pub fn index_in_sparse_bucket(ibucket: usize) -> u8 {
    (ibucket & BUCKET_MASK) as u8
}

/// Number of sparse arrays needed for `bucket_count` logical buckets.
#[inline]
pub fn nb_sparse_buckets(bucket_count: usize) -> usize {
    if bucket_count == 0 {
        return 0;
    }
    let rounded = crate::util::round_up_to_power_of_two(bucket_count);
    core::cmp::max(1, sparse_ibucket(rounded))
}

/// Count the occupied bits in a bitmap word.
#[inline]
pub(crate) fn popcount_bitmap(val: Bitmap) -> u8 {
    #[cfg(target_pointer_width = "64")]
    {
        popcountll(val) as u8
    }
    #[cfg(not(target_pointer_width = "64"))]
    {
        popcount(val) as u8
    }
}

/// A group of logical buckets stored densely with occupancy bitmaps.
///
/// Present values live in `values` in ascending index order. `bitmap_vals` marks
/// occupied indices. `bitmap_deleted_vals` marks tombstones left by erase.
pub struct SparseArray<T> {
    values: Vec<T>,
    bitmap_vals: Bitmap,
    bitmap_deleted_vals: Bitmap,
    last_array: bool,
}

impl<T> SparseArray<T> {
    /// An empty array that is not the last in a table.
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            bitmap_vals: 0,
            bitmap_deleted_vals: 0,
            last_array: false,
        }
    }

    /// Whether this array holds no values.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Number of values held.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// The heap capacity of the dense value block.
    ///
    /// The block over-allocates in fixed steps set by the sparsity level, so a
    /// full array reserves exactly `growth_step` more slots rather than doubling.
    #[cfg(test)]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.values.capacity()
    }

    /// Whether this is the last array in its table.
    #[inline]
    pub fn last(&self) -> bool {
        self.last_array
    }

    /// Flag this array as the last in its table.
    #[inline]
    pub fn set_as_last(&mut self) {
        self.last_array = true;
    }

    /// The occupancy bitmap. Bit `i` set means index `i` holds a value.
    #[inline]
    pub fn bitmap_vals(&self) -> Bitmap {
        self.bitmap_vals
    }

    /// The tombstone bitmap. Bit `i` set means index `i` was erased.
    #[inline]
    pub fn bitmap_deleted_vals(&self) -> Bitmap {
        self.bitmap_deleted_vals
    }

    /// The dense value slice in ascending index order.
    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// The dense value slice as mutable references.
    #[inline]
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }

    /// Whether logical index `index` holds a value.
    #[inline]
    pub fn has_value(&self, index: u8) -> bool {
        (self.bitmap_vals & (1 as Bitmap) << index) != 0
    }

    /// Whether logical index `index` is a tombstone.
    #[inline]
    pub fn has_deleted_value(&self, index: u8) -> bool {
        (self.bitmap_deleted_vals & (1 as Bitmap) << index) != 0
    }

    /// The dense offset of logical `index`: the count of value bits below it.
    #[inline]
    pub fn index_to_offset(&self, index: u8) -> usize {
        let mask = ((1 as Bitmap) << index).wrapping_sub(1);
        popcount_bitmap(self.bitmap_vals & mask) as usize
    }

    /// The logical index of the value at dense `offset`.
    pub fn offset_to_index(&self, offset: usize) -> u8 {
        let mut bitmap = self.bitmap_vals;
        let mut index: u8 = 0;
        let mut nb_ones = 0;
        while bitmap != 0 {
            if bitmap & 0x1 == 1 {
                if nb_ones == offset {
                    break;
                }
                nb_ones += 1;
            }
            index += 1;
            bitmap >>= 1;
        }
        index
    }

    /// A shared reference to the value at logical `index`.
    ///
    /// The caller must ensure `index` holds a value.
    #[inline]
    pub fn value(&self, index: u8) -> &T {
        &self.values[self.index_to_offset(index)]
    }

    /// A mutable reference to the value at logical `index`.
    #[inline]
    pub fn value_mut(&mut self, index: u8) -> &mut T {
        let offset = self.index_to_offset(index);
        &mut self.values[offset]
    }

    /// Insert `value` at logical `index`, which must be empty.
    ///
    /// `growth_step` is the sparsity capacity step. When the block is full it
    /// grows by exactly `growth_step` slots, not by `Vec` doubling, so the heap
    /// slack tracks the sparsity level. Returns the dense offset where the value
    /// landed.
    pub fn set(&mut self, index: u8, value: T, growth_step: usize) -> usize {
        debug_assert!(!self.has_value(index));
        let offset = self.index_to_offset(index);
        if self.values.len() == self.values.capacity() {
            self.values.reserve_exact(growth_step.max(1));
        }
        self.values.insert(offset, value);
        self.bitmap_vals |= (1 as Bitmap) << index;
        self.bitmap_deleted_vals &= !((1 as Bitmap) << index);
        offset
    }

    /// Erase the value at dense `offset` and return it.
    ///
    /// Marks `index` as a tombstone.
    pub fn remove_value(&mut self, offset: usize, index: u8) -> T {
        debug_assert!(self.has_value(index));
        let value = self.values.remove(offset);
        self.bitmap_vals &= !((1 as Bitmap) << index);
        self.bitmap_deleted_vals |= (1 as Bitmap) << index;
        value
    }

    /// Tombstone every present value and drop the dense block.
    ///
    /// Moves each occupied bit into the tombstone bitmap and clears the value
    /// bitmap, matching a slot-by-slot erase but in one pass. Returns how many
    /// values were dropped so the caller can update its counters.
    pub fn erase_all(&mut self) -> usize {
        let removed = self.values.len();
        self.values.clear();
        self.bitmap_deleted_vals |= self.bitmap_vals;
        self.bitmap_vals = 0;
        removed
    }

    /// Consume the array and yield its dense values in index order.
    #[inline]
    pub fn into_values(self) -> Vec<T> {
        self.values
    }

    /// Rebuild an array from decoded bitmaps and dense values.
    ///
    /// Used by hash-compatible deserialization, which restores slots directly
    /// without re-hashing.
    pub fn from_parts(bitmap_vals: Bitmap, bitmap_deleted_vals: Bitmap, values: Vec<T>) -> Self {
        debug_assert_eq!(popcount_bitmap(bitmap_vals) as usize, values.len());
        Self {
            values,
            bitmap_vals,
            bitmap_deleted_vals,
            last_array: false,
        }
    }
}

impl<T: Clone> Clone for SparseArray<T> {
    fn clone(&self) -> Self {
        Self {
            values: self.values.clone(),
            bitmap_vals: self.bitmap_vals,
            bitmap_deleted_vals: self.bitmap_deleted_vals,
            last_array: self.last_array,
        }
    }
}

impl<T> Default for SparseArray<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Capacity grows by the fixed step, not by Vec doubling. Fill the low bits
    // of one array in index order and check the block reserves in steps.
    #[test]
    fn capacity_grows_by_fixed_step() {
        const STEP: usize = 4;
        let mut array: SparseArray<i32> = SparseArray::new();
        for index in 0..8u8 {
            array.set(index, index as i32, STEP);
            // After each insert the capacity is the value count rounded up to a
            // multiple of the step. Doubling would jump past these bounds.
            let expected = array.len().div_ceil(STEP) * STEP;
            assert_eq!(array.capacity(), expected, "at len {}", array.len());
        }
    }

    #[test]
    fn a_larger_step_reserves_more_slack() {
        let mut high: SparseArray<i32> = SparseArray::new();
        let mut low: SparseArray<i32> = SparseArray::new();
        // One value each. The larger step over-allocates more.
        high.set(0, 0, 2);
        low.set(0, 0, 8);
        assert_eq!(high.capacity(), 2);
        assert_eq!(low.capacity(), 8);
    }
}
