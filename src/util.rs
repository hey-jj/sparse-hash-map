//! Small internal helpers shared across modules.

/// Whether `value` is a power of two. Zero is not a power of two.
#[inline]
pub(crate) fn is_power_of_two(value: usize) -> bool {
    value != 0 && (value & (value - 1)) == 0
}

/// The smallest power of two greater than or equal to `value`.
///
/// Returns 1 for an input of 0. The caller must keep `value` below the largest
/// representable power of two, since the result would otherwise overflow.
#[inline]
pub(crate) fn round_up_to_power_of_two(mut value: usize) -> usize {
    if is_power_of_two(value) {
        return value;
    }
    if value == 0 {
        return 1;
    }
    value -= 1;
    let mut i = 1;
    while i < usize::BITS as usize {
        value |= value >> i;
        i *= 2;
    }
    value + 1
}
