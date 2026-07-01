//! Heterogeneous lookup: a key type that is looked up through a borrowed form.
//!
//! The canonical case keys on owned pointers and looks up by the raw address.
//! The Rust analog keys on `Box<i32>` and looks up by the integer inside, using
//! a hasher and comparator that accept both the stored key and the lookup key.

mod common;

use sparse_hash_map::{EqKey, HashKey, SparseMap, StdEq};

// A key wrapping a boxed integer. Its identity is the integer value.
#[derive(Debug)]
struct BoxedKey(Box<i32>);

// A hasher that hashes both the stored key and a bare integer to the same value.
#[derive(Clone, Default)]
struct IntHash;

impl HashKey<BoxedKey> for IntHash {
    fn hash_key(&self, key: &BoxedKey) -> usize {
        *key.0 as usize
    }
}
impl HashKey<i32> for IntHash {
    fn hash_key(&self, key: &i32) -> usize {
        *key as usize
    }
}

// A comparator that compares a stored key against another key or a bare integer.
#[derive(Clone, Default)]
struct IntEq;

impl EqKey<BoxedKey, BoxedKey> for IntEq {
    fn eq_key(&self, a: &BoxedKey, b: &BoxedKey) -> bool {
        *a.0 == *b.0
    }
}
impl EqKey<BoxedKey, i32> for IntEq {
    fn eq_key(&self, a: &BoxedKey, b: &i32) -> bool {
        *a.0 == *b
    }
}

impl std::borrow::Borrow<i32> for BoxedKey {
    fn borrow(&self) -> &i32 {
        &self.0
    }
}

#[test]
fn test_heterogeneous_lookups() {
    let mut map: SparseMap<BoxedKey, i32, IntHash, IntEq> =
        SparseMap::with_parts(0, IntHash, IntEq);
    map.insert(BoxedKey(Box::new(1)), 4);
    map.insert(BoxedKey(Box::new(2)), 5);
    map.insert(BoxedKey(Box::new(3)), 6);
    assert_eq!(map.len(), 3);

    // Look up by the bare integer, not the boxed key.
    assert_eq!(map.at(&1), &4);
    assert_eq!(map.at(&2), &5);
    assert!(map.get(&99).is_none());

    assert!(map.get(&1).is_some());
    assert!(map.get(&2).is_some());
    assert!(map.get(&99).is_none());

    assert_eq!(map.count(&1), 1);
    assert_eq!(map.count(&2), 1);
    assert_eq!(map.count(&99), 0);

    assert_eq!(map.erase(&1), 1);
    assert_eq!(map.erase(&2), 1);
    assert_eq!(map.erase(&99), 0);
    assert_eq!(map.len(), 1);
}

#[test]
fn test_string_key_str_lookup() {
    // The common heterogeneous case: String keys, &str lookups through Borrow.
    let mut map: SparseMap<String, i32> = SparseMap::new();
    map.insert("alpha".to_string(), 1);
    map.insert("beta".to_string(), 2);

    assert_eq!(map.get("alpha"), Some(&1));
    assert_eq!(map.get("beta"), Some(&2));
    assert!(map.get("gamma").is_none());
    assert_eq!(map.erase("alpha"), 1);
    assert_eq!(map.len(), 1);

    // Prove the borrow-based comparator is StdEq under the hood.
    let _ = StdEq;
}
