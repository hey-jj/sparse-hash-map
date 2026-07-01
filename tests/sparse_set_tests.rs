//! Set behavior across the type and policy sweep.

mod common;

use common::{MoveOnly, TestType, VecDeserializer, VecSerializer};
use sparse_hash_map::{
    EqKey, GrowthPolicy, HashKey, Mod, SparsePgSet, SparseSet, Sparsity, StdEq, StdHash,
};

// insert 1000, re-insert reports no new key, find each.
fn body_insert<K, H, E, P, S>(mut set: SparseSet<K, H, E, P, S>)
where
    K: TestType,
    H: HashKey<K> + Clone,
    E: EqKey<K, K> + Clone,
    P: GrowthPolicy,
    S: Sparsity,
{
    let nb = 1000usize;
    for i in 0..nb {
        assert!(set.insert(K::get_key(i)));
        assert_eq!(set.get(&K::get_key(i)), Some(&K::get_key(i)));
    }
    assert_eq!(set.len(), nb);

    for i in 0..nb {
        assert!(!set.insert(K::get_key(i)));
        assert_eq!(set.get(&K::get_key(i)), Some(&K::get_key(i)));
    }

    for i in 0..nb {
        assert!(set.contains(&K::get_key(i)));
    }
}

macro_rules! set_case {
    ($name:ident, $K:ty, $H:ty, $P:ty, $S:ty) => {
        #[test]
        fn $name() {
            let set: SparseSet<$K, $H, StdEq, $P, $S> =
                SparseSet::with_parts(0, <$H>::default(), StdEq);
            body_insert(set);
        }
    };
}

mod sweep {
    use super::*;

    set_case!(
        insert_i64,
        i64,
        StdHash,
        sparse_hash_map::PowerOfTwo<2>,
        sparse_hash_map::Medium
    );
    set_case!(
        insert_str,
        String,
        StdHash,
        sparse_hash_map::PowerOfTwo<2>,
        sparse_hash_map::Medium
    );
    set_case!(
        insert_move,
        MoveOnly,
        StdHash,
        sparse_hash_map::PowerOfTwo<2>,
        sparse_hash_map::Medium
    );
    set_case!(
        insert_prime,
        i64,
        StdHash,
        sparse_hash_map::Prime,
        sparse_hash_map::Medium
    );
    set_case!(insert_mod, i64, StdHash, Mod, sparse_hash_map::Medium);
    set_case!(
        insert_move_prime,
        MoveOnly,
        StdHash,
        sparse_hash_map::Prime,
        sparse_hash_map::Medium
    );
    set_case!(
        insert_move_mod,
        MoveOnly,
        StdHash,
        Mod,
        sparse_hash_map::Medium
    );
}

#[test]
fn test_compare() {
    fn build(items: &[&str]) -> SparseSet<String> {
        let mut s = SparseSet::new();
        for i in items {
            s.insert(i.to_string());
        }
        s
    }

    let set1 = build(&["a", "e", "d", "c", "b"]);
    let set1_copy = build(&["e", "c", "b", "a", "d"]);
    let set2 = build(&["e", "c", "b", "a", "d", "f"]);
    let set3 = build(&["e", "c", "b", "a"]);
    let set4 = build(&["a", "e", "d", "c", "z"]);

    assert!(set1 == set1_copy);
    assert!(set1_copy == set1);
    for other in [&set2, &set3, &set4] {
        assert!(set1 != *other);
        assert!(*other != set1);
    }
    assert!(set2 != set3);
    assert!(set2 != set4);
    assert!(set3 != set4);
}

#[test]
fn test_insert_pointer_like() {
    // A single value inserted twice stays one element.
    let mut set: SparseSet<String> = SparseSet::new();
    set.insert("x".to_string());
    set.insert("x".to_string());
    assert_eq!(set.len(), 1);
    assert_eq!(set.iter().next().map(String::as_str), Some("x"));
}

fn serialize_bytes<K, H, E, P, S>(set: &SparseSet<K, H, E, P, S>) -> Vec<u8>
where
    K: sparse_hash_map::Serialize,
{
    let mut w = VecSerializer::default();
    set.serialize(&mut w);
    w.buf
}

#[test]
fn test_serialize_deserialize_reserve() {
    for nb_values in [0usize, 1, 3, 17, 1000] {
        let mut set: SparseSet<MoveOnly> = SparseSet::new();
        set.reserve(nb_values);
        for i in 0..nb_values {
            set.insert(MoveOnly::from_i64(i as i64));
        }

        let bytes = serialize_bytes(&set);
        for hash_compatible in [true, false] {
            let mut r = VecDeserializer::new(&bytes);
            let out = SparseSet::<MoveOnly>::deserialize_with(
                &mut r,
                hash_compatible,
                StdHash::default(),
                StdEq,
            )
            .unwrap();
            assert!(out == set);
        }
    }
}

#[test]
fn test_serialize_deserialize() {
    for nb_values in [0usize, 1, 3, 17, 1000] {
        let mut set: SparseSet<MoveOnly> = SparseSet::new();
        for i in 0..nb_values + 40 {
            set.insert(MoveOnly::from_i64(i as i64));
        }
        for i in nb_values..nb_values + 40 {
            set.erase(&MoveOnly::from_i64(i as i64));
        }
        assert_eq!(set.len(), nb_values);

        let bytes = serialize_bytes(&set);
        for hash_compatible in [true, false] {
            let mut r = VecDeserializer::new(&bytes);
            let out = SparseSet::<MoveOnly>::deserialize_with(
                &mut r,
                hash_compatible,
                StdHash::default(),
                StdEq,
            )
            .unwrap();
            assert!(out == set);
        }
    }
}

#[test]
fn test_prime_growth_set() {
    let mut set: SparsePgSet<i64> = SparsePgSet::with_parts(0, StdHash::default(), StdEq);
    for i in 0..500i64 {
        assert!(set.insert(i));
    }
    assert_eq!(set.len(), 500);
    for i in 0..500i64 {
        assert!(set.contains(&i));
    }
}
