//! Serialization protocol for maps and sets.
//!
//! The container writes a fixed sequence of fields through a [`Serializer`] and
//! reads them back through a [`Deserializer`]. The protocol defines the order
//! and logical type of each field. Binary layout, endianness, and float
//! representation are the caller's responsibility, chosen by the serializer and
//! deserializer they supply.
//!
//! Integer header fields are always [`u64`] so a file written on one platform
//! can be read on another. The value type is serialized through
//! [`Serialize`]/[`Deserialize`] implementations.

/// The serialization protocol version. Bumped on any wire-format change.
pub const SERIALIZATION_PROTOCOL_VERSION: u64 = 1;

/// A sink for the primitive field types the container emits.
///
/// Implement this to control the binary layout. The container calls these in a
/// fixed order.
pub trait Serializer {
    /// Write a 64-bit integer field.
    fn write_u64(&mut self, value: u64);
    /// Write a 32-bit float field.
    fn write_f32(&mut self, value: f32);
    /// Write raw bytes, such as string contents.
    fn write_bytes(&mut self, bytes: &[u8]);
}

/// A source for the primitive field types the container reads back.
///
/// The order of reads matches the order of writes in [`Serializer`].
pub trait Deserializer {
    /// Read a 64-bit integer field.
    fn read_u64(&mut self) -> u64;
    /// Read a 32-bit float field.
    fn read_f32(&mut self) -> f32;
    /// Read `len` raw bytes.
    fn read_bytes(&mut self, len: usize) -> Vec<u8>;

    /// Bytes still available to read, when the source knows its length.
    ///
    /// A variable-length [`Deserialize`] impl checks this before allocating a
    /// buffer sized by a length read from the input. A source that cannot report
    /// its length returns `None`, and the impl allocates as before. The default
    /// returns `None`.
    fn remaining(&self) -> Option<usize> {
        None
    }
}

/// A value type that can be written to a [`Serializer`].
pub trait Serialize {
    /// Emit this value's fields.
    fn serialize<S: Serializer>(&self, serializer: &mut S);
}

/// A value type that can be read from a [`Deserializer`].
pub trait Deserialize: Sized {
    /// Read one value.
    fn deserialize<D: Deserializer>(deserializer: &mut D) -> Self;
}

macro_rules! impl_int_serialize {
    ($($t:ty),*) => {$(
        impl Serialize for $t {
            fn serialize<S: Serializer>(&self, serializer: &mut S) {
                serializer.write_u64(*self as u64);
            }
        }
        impl Deserialize for $t {
            fn deserialize<D: Deserializer>(deserializer: &mut D) -> Self {
                deserializer.read_u64() as $t
            }
        }
    )*};
}

impl_int_serialize!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

impl Serialize for String {
    fn serialize<S: Serializer>(&self, serializer: &mut S) {
        serializer.write_u64(self.len() as u64);
        serializer.write_bytes(self.as_bytes());
    }
}

impl Deserialize for String {
    fn deserialize<D: Deserializer>(deserializer: &mut D) -> Self {
        let len = deserializer.read_u64() as usize;
        if let Some(remaining) = deserializer.remaining() {
            assert!(
                len <= remaining,
                "deserialized string length {len} exceeds the {remaining} bytes left in the input"
            );
        }
        let bytes = deserializer.read_bytes(len);
        String::from_utf8(bytes).expect("deserialized string is not valid utf-8")
    }
}

impl<A: Serialize, B: Serialize> Serialize for (A, B) {
    fn serialize<S: Serializer>(&self, serializer: &mut S) {
        self.0.serialize(serializer);
        self.1.serialize(serializer);
    }
}

impl<A: Deserialize, B: Deserialize> Deserialize for (A, B) {
    fn deserialize<D: Deserializer>(deserializer: &mut D) -> Self {
        let a = A::deserialize(deserializer);
        let b = B::deserialize(deserializer);
        (a, b)
    }
}

/// A deserialize failure. Carries a static reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeserializeError(pub &'static str);

impl core::fmt::Display for DeserializeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for DeserializeError {}
