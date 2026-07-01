//! Hashing and equality traits for keys.
//!
//! A hash here yields a `usize` directly, matching a hash functor that returns a
//! machine word. This lets an identity hash place a key in a known bucket, which
//! the collision and precalculated-hash tests rely on.
//!
//! [`StdHash`] is the default. It feeds the key to a [`core::hash::Hasher`] built
//! by a [`core::hash::BuildHasher`] and truncates the 64-bit result to `usize`.

use core::hash::{BuildHasher, Hash};
use core::marker::PhantomData;

/// Produce a `usize` hash for a borrowed key `Q`.
pub trait HashKey<Q: ?Sized> {
    /// Hash `key`.
    fn hash_key(&self, key: &Q) -> usize;
}

/// Test two borrowed keys for equality.
pub trait EqKey<A: ?Sized, B: ?Sized> {
    /// Whether `a` and `b` are the same key.
    fn eq_key(&self, a: &A, b: &B) -> bool;
}

/// A hasher built on the standard [`BuildHasher`] machinery.
///
/// The default `B` is [`std::collections::hash_map::RandomState`], the same
/// builder that backs [`std::collections::HashMap`].
#[derive(Clone, Default)]
pub struct StdHash<B = std::collections::hash_map::RandomState> {
    build: B,
}

impl<B: BuildHasher> StdHash<B> {
    /// Wrap an existing [`BuildHasher`].
    pub fn with_hasher(build: B) -> Self {
        Self { build }
    }
}

impl<B: BuildHasher, Q: Hash + ?Sized> HashKey<Q> for StdHash<B> {
    #[inline]
    fn hash_key(&self, key: &Q) -> usize {
        self.build.hash_one(key) as usize
    }
}

/// Equality by the standard [`PartialEq`] relation.
#[derive(Clone, Default)]
pub struct StdEq;

impl<A, B> EqKey<A, B> for StdEq
where
    A: PartialEq<B> + ?Sized,
    B: ?Sized,
{
    #[inline]
    fn eq_key(&self, a: &A, b: &B) -> bool {
        a == b
    }
}

/// Adapt a plain closure `Fn(&Q) -> usize` into a [`HashKey`].
///
/// Useful for identity and modulo hashers in tests and simple cases.
#[derive(Clone)]
pub struct FnHash<F> {
    f: F,
    _marker: PhantomData<fn()>,
}

impl<F> FnHash<F> {
    /// Wrap a hashing closure.
    pub fn new(f: F) -> Self {
        Self {
            f,
            _marker: PhantomData,
        }
    }
}

impl<F, Q> HashKey<Q> for FnHash<F>
where
    F: Fn(&Q) -> usize,
    Q: ?Sized,
{
    #[inline]
    fn hash_key(&self, key: &Q) -> usize {
        (self.f)(key)
    }
}
