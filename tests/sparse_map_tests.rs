//! Map behavior across the type and policy sweep.

mod common;

use common::{CopyOnly, IdentityHash, ModHash, MoveOnly, TestType, VecDeserializer, VecSerializer};
use sparse_hash_map::{
    EqKey, GrowthPolicy, HashKey, Mod, PowerOfTwo, SparseMap, SparsePgMap, Sparsity, StdEq, StdHash,
};

// A generic filled map builder. Reserves then inserts 0..n.
fn fill_map<K, V, H, E, P, S>(map: &mut SparseMap<K, V, H, E, P, S>, n: usize)
where
    K: TestType,
    V: TestType,
    H: HashKey<K> + Clone,
    E: EqKey<K, K> + Clone,
    P: GrowthPolicy,
    S: Sparsity,
{
    map.reserve(n);
    for i in 0..n {
        map.insert(K::get_key(i), V::get_value(i));
    }
}

// insert 1000, re-insert keeps original, find each. bucket_count(0) starts at 0.
fn body_insert<K, V, H, E, P, S>(mut map: SparseMap<K, V, H, E, P, S>)
where
    K: TestType,
    V: TestType,
    H: HashKey<K> + Clone,
    E: EqKey<K, K> + Clone,
    P: GrowthPolicy,
    S: Sparsity,
{
    assert_eq!(map.bucket_count(), 0);
    let nb = 1000usize;
    for i in 0..nb {
        assert!(map.insert(K::get_key(i), V::get_value(i)));
        assert_eq!(map.get(&K::get_key(i)), Some(&V::get_value(i)));
    }
    assert_eq!(map.len(), nb);

    for i in 0..nb {
        // A redundant insert keeps the stored value and reports no new entry.
        assert!(!map.insert(K::get_key(i), V::get_value(i + 1)));
        assert_eq!(map.get(&K::get_key(i)), Some(&V::get_value(i)));
    }

    for i in 0..nb {
        assert_eq!(map.get(&K::get_key(i)), Some(&V::get_value(i)));
    }
}

// Erase all one by one via pop_front, checking size and absence.
fn body_erase_loop<K, V, H, E, P, S>(mut map: SparseMap<K, V, H, E, P, S>)
where
    K: TestType + Clone,
    V: TestType,
    H: HashKey<K> + Clone,
    E: EqKey<K, K> + Clone,
    P: GrowthPolicy,
    S: Sparsity,
{
    let mut nb = 1000usize;
    fill_map(&mut map, nb);

    while let Some((k, _v)) = map.pop_front() {
        nb -= 1;
        assert_eq!(map.count(&k), 0);
        assert_eq!(map.len(), nb);
    }
    assert!(map.is_empty());
}

// Erase five at a time via a range erase; size drops by five each pass.
fn body_erase_loop_range<K, V, H, E, P, S>(mut map: SparseMap<K, V, H, E, P, S>)
where
    K: TestType,
    V: TestType,
    H: HashKey<K> + Clone,
    E: EqKey<K, K> + Clone,
    P: GrowthPolicy,
    S: Sparsity,
{
    let hop = 5usize;
    let mut nb = 1000usize;
    fill_map(&mut map, nb);

    while !map.is_empty() {
        map.erase_range(0, hop);
        nb -= hop;
        assert_eq!(map.len(), nb);
    }
    assert!(map.is_empty());
}

// Insert 1000, erase even keys, insert 1000 more, then verify presence.
fn body_insert_erase_insert<K, V, H, E, P, S>(mut map: SparseMap<K, V, H, E, P, S>)
where
    K: TestType,
    V: TestType,
    H: HashKey<K> + Clone,
    E: EqKey<K, K> + Clone,
    P: GrowthPolicy,
    S: Sparsity,
{
    let nb = 2000usize;
    for i in 0..nb / 2 {
        assert!(map.insert(K::get_key(i), V::get_value(i)));
    }
    assert_eq!(map.len(), nb / 2);

    for i in 0..nb / 2 {
        if i % 2 == 0 {
            assert_eq!(map.erase(&K::get_key(i)), 1);
        }
    }
    assert_eq!(map.len(), nb / 4);

    for i in nb / 2..nb {
        assert!(map.insert(K::get_key(i), V::get_value(i)));
    }
    assert_eq!(map.len(), nb - nb / 4);

    for i in 0..nb {
        if i % 2 == 0 && i < nb / 2 {
            assert!(map.get(&K::get_key(i)).is_none());
        } else {
            assert_eq!(map.get(&K::get_key(i)), Some(&V::get_value(i)));
        }
    }
}

// compare over string->int maps with same and different size, key, value.
fn body_compare<H, E, P, S>()
where
    H: HashKey<String> + Clone + Default,
    E: EqKey<String, String> + Clone + Default,
    P: GrowthPolicy,
    S: Sparsity,
{
    fn build<H, E, P, S>(pairs: &[(&str, i64)]) -> SparseMap<String, i64, H, E, P, S>
    where
        H: HashKey<String> + Clone + Default,
        E: EqKey<String, String> + Clone + Default,
        P: GrowthPolicy,
        S: Sparsity,
    {
        let mut m = SparseMap::with_parts(0, H::default(), E::default());
        for (k, v) in pairs {
            m.insert(k.to_string(), *v);
        }
        m
    }

    let map1 = build::<H, E, P, S>(&[("a", 1), ("e", 5), ("d", 4), ("c", 3), ("b", 2)]);
    let map1_copy = build::<H, E, P, S>(&[("e", 5), ("c", 3), ("b", 2), ("a", 1), ("d", 4)]);
    let map2 = build::<H, E, P, S>(&[("e", 5), ("c", 3), ("b", 2), ("a", 1), ("d", 4), ("f", 6)]);
    let map3 = build::<H, E, P, S>(&[("e", 5), ("c", 3), ("b", 2), ("a", 1)]);
    let map4 = build::<H, E, P, S>(&[("a", 1), ("e", 5), ("d", 4), ("c", 3), ("b", 26)]);
    let map5 = build::<H, E, P, S>(&[("a", 1), ("e", 5), ("d", 4), ("c", 3), ("z", 2)]);

    assert!(map1 == map1_copy);
    assert!(map1_copy == map1);
    for other in [&map2, &map3, &map4, &map5] {
        assert!(map1 != *other);
        assert!(*other != map1);
    }
    assert!(map2 != map3);
    assert!(map2 != map4);
    assert!(map2 != map5);
    assert!(map3 != map4);
    assert!(map3 != map5);
    assert!(map4 != map5);
}

// The sweep macro instantiates each templated case for a map type.
macro_rules! sweep_case {
    ($name:ident, $body:ident, $K:ty, $V:ty, $H:ty, $P:ty, $S:ty) => {
        #[test]
        fn $name() {
            let map: SparseMap<$K, $V, $H, StdEq, $P, $S> =
                SparseMap::with_parts(0, <$H>::default(), StdEq);
            $body(map);
        }
    };
}

// Instantiate the four templated bodies over the representative type matrix.
mod sweep {
    use super::*;

    // insert
    sweep_case!(
        insert_i64,
        body_insert,
        i64,
        i64,
        StdHash,
        PowerOfTwo<2>,
        sparse_hash_map::Medium
    );
    sweep_case!(
        insert_str,
        body_insert,
        String,
        String,
        StdHash,
        PowerOfTwo<2>,
        sparse_hash_map::Medium
    );
    sweep_case!(
        insert_modhash_i64,
        body_insert,
        i64,
        i64,
        ModHash<9>,
        PowerOfTwo<2>,
        sparse_hash_map::Medium
    );
    sweep_case!(
        insert_pow4,
        body_insert,
        i64,
        i64,
        ModHash<9>,
        PowerOfTwo<4>,
        sparse_hash_map::Medium
    );
    sweep_case!(
        insert_mod,
        body_insert,
        i64,
        i64,
        ModHash<9>,
        Mod,
        sparse_hash_map::Medium
    );
    sweep_case!(
        insert_high,
        body_insert,
        String,
        String,
        ModHash<9>,
        PowerOfTwo<2>,
        sparse_hash_map::High
    );
    sweep_case!(
        insert_low,
        body_insert,
        String,
        String,
        ModHash<9>,
        PowerOfTwo<2>,
        sparse_hash_map::Low
    );

    // erase_loop
    sweep_case!(
        erase_loop_i64,
        body_erase_loop,
        i64,
        i64,
        StdHash,
        PowerOfTwo<2>,
        sparse_hash_map::Medium
    );
    sweep_case!(
        erase_loop_str,
        body_erase_loop,
        String,
        String,
        ModHash<9>,
        PowerOfTwo<2>,
        sparse_hash_map::Medium
    );
    sweep_case!(
        erase_loop_pow4,
        body_erase_loop,
        i64,
        i64,
        ModHash<9>,
        PowerOfTwo<4>,
        sparse_hash_map::Medium
    );
    sweep_case!(
        erase_loop_mod,
        body_erase_loop,
        i64,
        i64,
        ModHash<9>,
        Mod,
        sparse_hash_map::Medium
    );

    // erase_loop_range
    sweep_case!(
        erase_range_i64,
        body_erase_loop_range,
        i64,
        i64,
        StdHash,
        PowerOfTwo<2>,
        sparse_hash_map::Medium
    );
    sweep_case!(
        erase_range_modhash,
        body_erase_loop_range,
        i64,
        i64,
        ModHash<9>,
        PowerOfTwo<2>,
        sparse_hash_map::Medium
    );

    // insert_erase_insert
    sweep_case!(
        ie_insert_i64,
        body_insert_erase_insert,
        i64,
        i64,
        StdHash,
        PowerOfTwo<2>,
        sparse_hash_map::Medium
    );
    sweep_case!(
        ie_insert_str,
        body_insert_erase_insert,
        String,
        String,
        ModHash<9>,
        PowerOfTwo<2>,
        sparse_hash_map::Medium
    );
    sweep_case!(
        ie_insert_pow4,
        body_insert_erase_insert,
        i64,
        i64,
        ModHash<9>,
        PowerOfTwo<4>,
        sparse_hash_map::Medium
    );
    sweep_case!(
        ie_insert_mod,
        body_insert_erase_insert,
        i64,
        i64,
        ModHash<9>,
        Mod,
        sparse_hash_map::Medium
    );
    sweep_case!(
        ie_insert_high,
        body_insert_erase_insert,
        String,
        String,
        ModHash<9>,
        PowerOfTwo<2>,
        sparse_hash_map::High
    );
    sweep_case!(
        ie_insert_low,
        body_insert_erase_insert,
        String,
        String,
        ModHash<9>,
        PowerOfTwo<2>,
        sparse_hash_map::Low
    );

    // Copy-only value type. This value type has no move-only marker, so it
    // drives the copying capacity path in the sparse array. Sweeping it across
    // the policies checks that path under collisions and rehashes.
    sweep_case!(
        insert_copyonly,
        body_insert,
        CopyOnly,
        CopyOnly,
        ModHash<9>,
        PowerOfTwo<2>,
        sparse_hash_map::Medium
    );
    sweep_case!(
        insert_copyonly_pow4,
        body_insert,
        CopyOnly,
        CopyOnly,
        ModHash<9>,
        PowerOfTwo<4>,
        sparse_hash_map::Medium
    );
    sweep_case!(
        insert_copyonly_mod,
        body_insert,
        CopyOnly,
        CopyOnly,
        ModHash<9>,
        Mod,
        sparse_hash_map::Medium
    );
    sweep_case!(
        erase_loop_copyonly,
        body_erase_loop,
        CopyOnly,
        CopyOnly,
        ModHash<9>,
        PowerOfTwo<2>,
        sparse_hash_map::Medium
    );
    sweep_case!(
        ie_insert_copyonly,
        body_insert_erase_insert,
        CopyOnly,
        CopyOnly,
        ModHash<9>,
        PowerOfTwo<2>,
        sparse_hash_map::Medium
    );
}

#[test]
fn compare_default() {
    body_compare::<StdHash, StdEq, PowerOfTwo<2>, sparse_hash_map::Medium>();
    body_compare::<ModHash<9>, StdEq, PowerOfTwo<2>, sparse_hash_map::Medium>();
    body_compare::<ModHash<9>, StdEq, Mod, sparse_hash_map::Medium>();
}

// Single-instance cases.

#[test]
fn test_range_insert() {
    let nb = 1000i32;
    let values: Vec<(i32, i32)> = (0..nb).map(|i| (i, i + 1)).collect();

    let mut map: SparseMap<i32, i32> = SparseMap::from([(-1, 1), (-2, 2)]);
    // Insert the sub-range [10, nb - 5).
    for &(k, v) in &values[10..(nb as usize - 5)] {
        map.insert(k, v);
    }

    assert_eq!(map.len(), 987);
    assert_eq!(map[&-1], 1);
    assert_eq!(map[&-2], 2);
    for i in 10..nb - 5 {
        assert_eq!(map[&i], i + 1);
    }
}

#[test]
fn test_insert_with_hint_equivalent() {
    // Without C++ hints, the observable result is the plain insert semantics.
    let mut map: SparseMap<i32, i32> = SparseMap::from([(1, 0), (2, 1), (3, 2)]);

    // Redundant key keeps the value.
    assert!(!map.insert(3, 4));
    assert_eq!(map[&3], 2);
    assert!(!map.insert(2, 4));
    assert_eq!(map[&2], 1);
    assert_eq!(map.len(), 3);

    assert!(map.insert(4, 3));
    assert!(map.insert(5, 4));
    assert_eq!(map.len(), 5);
}

#[test]
fn test_emplace_keeps_original() {
    let mut map: SparseMap<i64, MoveOnly> = SparseMap::new();
    let (_, inserted) = map.try_emplace(10, || MoveOnly::from_i64(1));
    assert!(inserted);
    assert_eq!(map.get(&10).unwrap().value(), "1");

    let (v, inserted) = map.try_emplace(10, || MoveOnly::from_i64(3));
    assert!(!inserted);
    assert_eq!(v.value(), "1");
}

#[test]
fn test_try_emplace() {
    let mut map: SparseMap<i64, MoveOnly> = SparseMap::new();
    let (_, inserted) = map.try_emplace(10, || MoveOnly::from_i64(1));
    assert!(inserted);
    assert_eq!(map.get(&10).unwrap().value(), "1");

    let (_, inserted) = map.try_emplace(10, || MoveOnly::from_i64(3));
    assert!(!inserted);
    assert_eq!(map.get(&10).unwrap().value(), "1");
}

#[test]
fn test_try_emplace_2() {
    let mut map: SparseMap<String, MoveOnly> = SparseMap::new();
    let nb = 1000usize;
    for i in 0..nb {
        let (_, inserted) = map.try_emplace(format!("Key {i}"), || MoveOnly::from_i64(i as i64));
        assert!(inserted);
        assert_eq!(map.get(&format!("Key {i}")).unwrap().value(), i.to_string());
    }
    assert_eq!(map.len(), nb);

    for i in 0..nb {
        let (v, inserted) =
            map.try_emplace(format!("Key {i}"), || MoveOnly::from_i64((i + 1) as i64));
        assert!(!inserted);
        assert_eq!(v.value(), i.to_string());
    }

    for i in 0..nb {
        assert_eq!(map.get(&format!("Key {i}")).unwrap().value(), i.to_string());
    }
}

#[test]
fn test_try_emplace_does_not_consume_on_occupied() {
    let mut map: SparseMap<i64, i64> = SparseMap::new();
    map.insert(10, 1);

    let spare = MoveOnly::from_i64(42);
    // The closure is not called on the occupied path, so `spare` stays intact.
    let (_v, inserted) = map.try_emplace(10, || {
        let _ = &spare;
        99
    });
    assert!(!inserted);
    assert_eq!(spare.value(), "42");
}

#[test]
fn test_insert_or_assign() {
    let mut map: SparseMap<i64, MoveOnly> = SparseMap::new();
    let (_, inserted) = map.insert_or_assign(10, MoveOnly::from_i64(1));
    assert!(inserted);
    assert_eq!(map.get(&10).unwrap().value(), "1");

    let (v, inserted) = map.insert_or_assign(10, MoveOnly::from_i64(3));
    assert!(!inserted);
    assert_eq!(v.value(), "3");
}

#[test]
fn test_range_erase_all() {
    let mut map: SparseMap<String, i64> = SparseMap::new();
    fill_map(&mut map, 1000);
    map.erase_all();
    assert!(map.is_empty());
}

#[test]
fn test_range_erase() {
    let mut map: SparseMap<String, i64> = SparseMap::new();
    fill_map(&mut map, 1000);
    // Erase the range [10, 220): 210 entries.
    map.erase_range(10, 210);
    assert_eq!(map.len(), 790);
    let live: Vec<String> = map.keys().cloned().collect();
    assert_eq!(live.len(), 790);
    for k in &live {
        assert_eq!(map.count(k), 1);
    }
}

#[test]
fn test_rehash_empty() {
    let mut map: SparseMap<i64, i64> = SparseMap::new();
    fill_map(&mut map, 100);
    let bucket_count = map.bucket_count();
    assert!(bucket_count >= 100);

    map.clear();
    assert_eq!(map.bucket_count(), bucket_count);
    assert!(map.is_empty());

    map.rehash(0);
    assert_eq!(map.bucket_count(), 0);
    assert!(map.is_empty());

    assert!(map.get(&1).is_none());
    assert_eq!(map.erase(&1), 0);
    assert!(map.insert(1, 10));
    assert_eq!(map[&1], 10);
}

#[test]
fn test_clear() {
    let mut map: SparseMap<i64, i64> = SparseMap::new();
    fill_map(&mut map, 1000);
    map.clear();
    assert_eq!(map.len(), 0);
    assert_eq!(map.iter().count(), 0);

    map.insert(5, -5);
    for (k, v) in [(1, -1), (2, -1), (4, -4), (3, -3)] {
        map.insert(k, v);
    }
    let expected: SparseMap<i64, i64> =
        SparseMap::from([(5, -5), (1, -1), (2, -1), (4, -4), (3, -3)]);
    assert!(map == expected);
}

#[test]
fn test_modify_value_through_iterator() {
    let mut map: SparseMap<i64, i64> = SparseMap::new();
    fill_map(&mut map, 100);

    for (k, v) in map.iter_mut() {
        if k % 2 == 0 {
            *v = -1;
        }
    }

    for (k, v) in &map {
        if k % 2 == 0 {
            assert_eq!(*v, -1);
        } else {
            assert_ne!(*v, -1);
        }
    }
}

#[test]
fn test_extreme_bucket_count_value_construction() {
    assert!(SparseMap::<i32, i32>::try_with_bucket_count(usize::MAX).is_err());
    assert!(SparseMap::<i32, i32>::try_with_bucket_count(usize::MAX / 2 + 1).is_err());

    assert!(
        SparsePgMap::<i32, i32>::try_with_parts(usize::MAX, StdHash::default(), StdEq).is_err()
    );
    assert!(SparseMap::<i32, i32, StdHash, StdEq, Mod>::try_with_parts(
        usize::MAX,
        StdHash::default(),
        StdEq
    )
    .is_err());
}

#[test]
fn test_assign_operator() {
    let mut map: SparseMap<i64, i64> = SparseMap::from([(0, 10), (-2, 20)]);
    assert_eq!(map.len(), 2);

    map = SparseMap::from([(1, 3), (2, 4)]);
    assert_eq!(map.len(), 2);
    assert_eq!(map.at(&1), &3);
    assert_eq!(map.at(&2), &4);
    assert!(map.get(&0).is_none());

    map = SparseMap::from([]);
    assert!(map.is_empty());
}

#[test]
fn test_move_and_reassign() {
    let map: SparseMap<String, String> = SparseMap::from([
        ("Key1".to_string(), "Value1".to_string()),
        ("Key2".to_string(), "Value2".to_string()),
        ("Key3".to_string(), "Value3".to_string()),
    ]);
    let map_move = map;
    assert_eq!(map_move.len(), 3);
}

#[test]
fn test_copy_constructor_and_operator() {
    let mut map: SparseMap<String, String, ModHash<9>> =
        SparseMap::with_parts(0, ModHash::default(), StdEq);
    for i in 0..100usize {
        map.insert(format!("Key {i}"), format!("Value {i}"));
    }

    let map_copy = map.clone();
    let map_copy2 = map.clone();
    assert!(map == map_copy);
    map.clear();
    assert!(map_copy == map_copy2);
}

#[test]
fn test_at() {
    let map: SparseMap<i64, i64> = SparseMap::from([(0, 10), (-2, 20)]);
    assert_eq!(map.at(&0), &10);
    assert_eq!(map.at(&-2), &20);
    assert!(map.get(&1).is_none());
}

#[test]
#[should_panic(expected = "couldn't find key")]
fn test_at_missing_panics() {
    let map: SparseMap<i64, i64> = SparseMap::from([(0, 10)]);
    let _ = map.at(&1);
}

#[test]
fn test_contains() {
    let map: SparseMap<i64, i64> = SparseMap::from([(0, 10), (-2, 20)]);
    assert!(map.contains_key(&0));
    assert!(map.contains_key(&-2));
    assert!(!map.contains_key(&-3));
}

#[test]
fn test_access_operator() {
    let mut map: SparseMap<i64, i64> = SparseMap::from([(0, 10), (-2, 20)]);
    assert_eq!(map[&0], 10);
    assert_eq!(map[&-2], 20);
    // Reading an absent key default-inserts.
    assert_eq!(*map.entry_or_default(2), 0);
    assert_eq!(map.len(), 3);
}

#[test]
fn test_swap() {
    let mut map: SparseMap<i64, i64> = SparseMap::from([(1, 10), (8, 80), (3, 30)]);
    let mut map2: SparseMap<i64, i64> = SparseMap::from([(4, 40), (5, 50)]);
    std::mem::swap(&mut map, &mut map2);

    assert!(map == SparseMap::from([(4, 40), (5, 50)]));
    assert!(map2 == SparseMap::from([(1, 10), (8, 80), (3, 30)]));

    map.insert(6, 60);
    map2.insert(4, 40);
    assert!(map == SparseMap::from([(4, 40), (5, 50), (6, 60)]));
    assert!(map2 == SparseMap::from([(1, 10), (8, 80), (3, 30), (4, 40)]));
}

#[test]
fn test_swap_empty() {
    let mut map: SparseMap<i64, i64> = SparseMap::from([(1, 10), (8, 80), (3, 30)]);
    let mut map2: SparseMap<i64, i64> = SparseMap::new();
    std::mem::swap(&mut map, &mut map2);

    assert!(map == SparseMap::from([]));
    assert!(map2 == SparseMap::from([(1, 10), (8, 80), (3, 30)]));

    map.insert(6, 60);
    map2.insert(4, 40);
    assert!(map == SparseMap::from([(6, 60)]));
    assert!(map2 == SparseMap::from([(1, 10), (8, 80), (3, 30), (4, 40)]));
}

#[test]
fn test_key_equal() {
    // A hasher and comparator where any odd x equals x - 1.
    #[derive(Clone, Default)]
    struct OddEvenHash;
    impl HashKey<u64> for OddEvenHash {
        fn hash_key(&self, key: &u64) -> usize {
            let base = if key % 2 == 1 { key - 1 } else { *key };
            base as usize
        }
    }
    #[derive(Clone, Default)]
    struct OddEvenEq;
    impl EqKey<u64, u64> for OddEvenEq {
        fn eq_key(&self, a: &u64, b: &u64) -> bool {
            let na = if a % 2 == 1 { a - 1 } else { *a };
            let nb = if b % 2 == 1 { b - 1 } else { *b };
            na == nb
        }
    }

    let mut map: SparseMap<u64, u64, OddEvenHash, OddEvenEq> =
        SparseMap::with_parts(0, OddEvenHash, OddEvenEq);
    assert!(map.insert(2, 10));
    assert_eq!(map.at(&2), &10);
    assert_eq!(map.at(&3), &10);
    assert!(!map.insert(3, 10));
    assert_eq!(map.len(), 1);
}

#[test]
fn test_all_buckets_marked_as_deleted_or_with_a_value() {
    // Intrusive edge case. identity hash, load factor 0.8, 64 buckets.
    let mut map: SparseMap<u32, u32, IdentityHash> = SparseMap::with_parts(0, IdentityHash, StdEq);
    map.set_max_load_factor(0.8);
    map.rehash(64);

    assert_eq!(map.bucket_count(), 64);
    assert_eq!(map.max_load_factor(), 0.8);

    for i in 0..51u32 {
        assert!(map.insert(i, i));
    }
    for i in 0..14u32 {
        assert_eq!(map.erase(&i), 1);
    }
    for i in 51..64u32 {
        assert!(map.insert(i, i));
    }

    assert_eq!(map.len(), 50);
    assert_eq!(map.bucket_count(), 64);

    for i in 0..14u32 {
        assert!(map.get(&i).is_none());
    }
    for i in 0..14u32 {
        assert_eq!(map.erase(&i), 0);
    }
    assert_eq!(map.len(), 50);
    assert_eq!(map.bucket_count(), 64);

    for i in 14..64u32 {
        assert!(!map.insert(i, i));
    }
    assert_eq!(map.len(), 50);
    assert_eq!(map.bucket_count(), 64);

    for i in 0..14u32 {
        assert!(map.insert(i, i));
    }
    assert_eq!(map.len(), 64);
    assert_eq!(map.bucket_count(), 128);
}

#[test]
fn test_empty_map() {
    let mut map: SparseMap<String, i32> = SparseMap::with_bucket_count(0);

    assert_eq!(map.bucket_count(), 0);
    assert_eq!(map.len(), 0);
    assert_eq!(map.load_factor(), 0.0);
    assert!(map.is_empty());

    assert_eq!(map.iter().count(), 0);
    assert!(map.get("").is_none());
    assert!(map.get("test").is_none());
    assert_eq!(map.count(""), 0);
    assert_eq!(map.count("test"), 0);
    assert!(!map.contains_key(""));
    assert!(!map.contains_key("test"));

    let range: Vec<_> = map.iter().collect();
    assert!(range.is_empty());

    assert_eq!(map.erase("test"), 0);
    assert_eq!(*map.entry_or_default("new value".to_string()), 0);
}

#[test]
fn test_precalculated_hash() {
    let map: SparseMap<i32, i32, IdentityHash> = {
        let mut m = SparseMap::with_parts(0, IdentityHash, StdEq);
        for (k, v) in [(1, -1), (2, -2), (3, -3), (4, -4), (5, -5), (6, -6)] {
            m.insert(k, v);
        }
        m
    };

    let h = map.hash_function().hash_key(&3);
    assert_eq!(map.get_precalc(&3, h), Some(&-3));
    assert_eq!(map.at_precalc(&3, h), &-3);
    assert!(map.contains_key_precalc(&3, h));
    assert_eq!(map.count_precalc(&3, h), 1);

    let absent = map.hash_function().hash_key(&99);
    assert_eq!(map.count_precalc(&99, absent), 0);

    let mut map = map;
    assert_eq!(map.erase_precalc(&3, h), 1);
    assert!(map.get(&3).is_none());
}

#[test]
fn test_equal_range() {
    let map: SparseMap<i32, i32> = SparseMap::from([(0, 10), (-2, 20)]);

    // A present key yields a range of length 1 with the stored pair.
    let mut range = map.equal_range(&0);
    assert_eq!(range.len(), 1);
    assert_eq!(range.next(), Some((&0, &10)));
    assert_eq!(range.next(), None);

    // An absent key yields an empty range.
    let mut empty = map.equal_range(&1);
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.next(), None);
}

#[test]
fn test_equal_range_precalc() {
    let map: SparseMap<i32, i32, IdentityHash> = {
        let mut m = SparseMap::with_parts(0, IdentityHash, StdEq);
        m.insert(0, 10);
        m.insert(2, 20);
        m
    };

    let h = map.hash_function().hash_key(&2);
    let found: Vec<_> = map.equal_range_precalc(&2, h).collect();
    assert_eq!(found, vec![(&2, &20)]);

    let absent = map.hash_function().hash_key(&5);
    assert_eq!(map.equal_range_precalc(&5, absent).count(), 0);
}

#[test]
fn test_owning_into_iter_and_collect() {
    let map: SparseMap<i64, i64> = SparseMap::from([(1, 10), (2, 20), (3, 30)]);

    // Move entries out and rebuild an ordinary collection.
    let mut pairs: Vec<(i64, i64)> = map.into_iter().collect();
    pairs.sort_unstable();
    assert_eq!(pairs, vec![(1, 10), (2, 20), (3, 30)]);
}

#[test]
fn test_into_iter_moves_move_only_values() {
    let mut map: SparseMap<i64, MoveOnly> = SparseMap::new();
    map.insert(1, MoveOnly::from_i64(11));
    map.insert(2, MoveOnly::from_i64(22));

    let mut got: Vec<(i64, String)> = map
        .into_iter()
        .map(|(k, v)| (k, v.value().to_string()))
        .collect();
    got.sort();
    assert_eq!(got, vec![(1, "11".to_string()), (2, "22".to_string())]);
}

#[test]
fn test_iter_mut_over_ref() {
    let mut map: SparseMap<i64, i64> = SparseMap::from([(1, 1), (2, 2), (3, 3)]);
    for (_, v) in &mut map {
        *v *= 10;
    }
    let mut vals: Vec<i64> = map.values().copied().collect();
    vals.sort_unstable();
    assert_eq!(vals, vec![10, 20, 30]);
}

#[test]
fn test_extend() {
    let mut map: SparseMap<i64, i64> = SparseMap::new();
    map.extend([(1, 1), (2, 2)]);
    map.extend(vec![(3, 3), (2, 99)]); // present key keeps its value
    assert_eq!(map.len(), 3);
    assert_eq!(map[&1], 1);
    assert_eq!(map[&2], 2);
    assert_eq!(map[&3], 3);
}

#[test]
fn test_retain() {
    let mut map: SparseMap<i64, i64> = (0..100).map(|i| (i, i)).collect();
    map.retain(|k, _| k % 2 == 0);
    assert_eq!(map.len(), 50);
    for i in 0..100 {
        assert_eq!(map.contains_key(&i), i % 2 == 0);
    }

    // The retained entries still round-trip through lookup after the shifts.
    for i in (0..100).step_by(2) {
        assert_eq!(map[&i], i);
    }
}

#[test]
fn test_retain_can_mutate_values() {
    let mut map: SparseMap<i64, i64> = (0..10).map(|i| (i, i)).collect();
    map.retain(|_, v| {
        *v += 1;
        true
    });
    for i in 0..10 {
        assert_eq!(map[&i], i + 1);
    }
}

#[test]
fn test_try_insert_returns_rejected_value() {
    let mut map: SparseMap<i64, MoveOnly> = SparseMap::new();
    assert!(map.try_insert(1, MoveOnly::from_i64(10)).is_ok());

    // A collision hands the value back instead of dropping it.
    let rejected = map.try_insert(1, MoveOnly::from_i64(99));
    let (k, v) = rejected.expect_err("key already present");
    assert_eq!(k, 1);
    assert_eq!(v.value(), "99");
    // The stored value is untouched.
    assert_eq!(map.get(&1).unwrap().value(), "10");
}

#[test]
fn test_load_factor_value() {
    let mut map: SparseMap<i64, i64> = SparseMap::new();
    for i in 0..10i64 {
        map.insert(i, i);
    }
    let expected = map.len() as f32 / map.bucket_count() as f32;
    assert!((map.load_factor() - expected).abs() < 1e-6);
}

#[test]
fn test_max_load_factor_clamp() {
    let mut map: SparseMap<i64, i64> = SparseMap::new();
    map.set_max_load_factor(0.05);
    assert_eq!(map.max_load_factor(), 0.1);
    map.set_max_load_factor(2.0);
    assert_eq!(map.max_load_factor(), 0.8);
}

#[test]
fn test_iterator_stable_across_rehash() {
    let mut map: SparseMap<i64, i64> = SparseMap::with_bucket_count(2);
    for i in 0..500i64 {
        map.insert(i, i * 2);
    }
    for i in 0..500i64 {
        assert_eq!(map.get(&i), Some(&(i * 2)));
    }
}

// Serialization round trips.

fn serialize_bytes<K, V, H, E, P, S>(map: &SparseMap<K, V, H, E, P, S>) -> Vec<u8>
where
    (K, V): sparse_hash_map::Serialize,
{
    let mut w = VecSerializer::default();
    map.serialize(&mut w);
    w.buf
}

#[test]
fn test_serialize_deserialize_empty() {
    let map: SparseMap<String, i64> = SparseMap::with_bucket_count(0);
    let bytes = serialize_bytes(&map);

    for hash_compatible in [true, false] {
        let mut r = VecDeserializer::new(&bytes);
        let out = SparseMap::<String, i64>::deserialize_with(
            &mut r,
            hash_compatible,
            StdHash::default(),
            StdEq,
        )
        .unwrap();
        assert!(out == map);
    }
}

#[test]
fn test_serialize_deserialize_few() {
    let map: SparseMap<i64, i64> = SparseMap::from([(10, 100), (4, 14), (9, 201)]);
    let bytes = serialize_bytes(&map);

    for hash_compatible in [true, false] {
        let mut r = VecDeserializer::new(&bytes);
        let out = SparseMap::<i64, i64>::deserialize_with(
            &mut r,
            hash_compatible,
            StdHash::default(),
            StdEq,
        )
        .unwrap();
        assert!(out == map);
    }
}

#[test]
fn test_serialize_deserialize() {
    let mut map: SparseMap<i64, i64> = SparseMap::new();
    for i in 0..1040i64 {
        map.insert(i, i * 3);
    }
    for i in 1000..1040i64 {
        map.erase(&i);
    }
    assert_eq!(map.len(), 1000);

    let bytes = serialize_bytes(&map);
    for hash_compatible in [true, false] {
        let mut r = VecDeserializer::new(&bytes);
        let out = SparseMap::<i64, i64>::deserialize_with(
            &mut r,
            hash_compatible,
            StdHash::default(),
            StdEq,
        )
        .unwrap();
        assert!(out == map);
    }
}

#[test]
fn test_serialize_deserialize_error_paths() {
    let map: SparseMap<i64, i64> = SparseMap::from([(1, 1)]);
    let mut bytes = serialize_bytes(&map);

    // Corrupt the protocol version (first u64).
    bytes[0] = 9;
    let mut r = VecDeserializer::new(&bytes);
    let out = SparseMap::<i64, i64>::deserialize_with(&mut r, false, StdHash::default(), StdEq);
    assert!(out.is_err());
}

// A deterministic hasher that differs from the default. It offsets a stable
// hash by a constant so the same key hashes the same way on every call.
#[derive(Clone, Default)]
struct OffsetHash;

impl HashKey<String> for OffsetHash {
    fn hash_key(&self, key: &String) -> usize {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut h);
        (h.finish() as usize).wrapping_add(123)
    }
}

#[test]
fn test_serialize_deserialize_with_different_hash() {
    // Serialize with the default hasher, deserialize into a map with a different
    // hasher. With hash_compatible false the values are re-hashed, so every key
    // must still be found.
    let mut map: SparseMap<String, i64> = SparseMap::new();
    for i in 0..1000usize {
        map.insert(format!("Key {i}"), i as i64);
    }

    let bytes = serialize_bytes(&map);
    let mut r = VecDeserializer::new(&bytes);
    let out =
        SparseMap::<String, i64, OffsetHash>::deserialize_with(&mut r, false, OffsetHash, StdEq)
            .unwrap();

    assert_eq!(out.len(), map.len());
    for (k, v) in map.iter() {
        assert_eq!(out.get(k), Some(v));
    }
}

#[test]
fn test_deserialize_hash_compatible_error_paths() {
    // Header layout: version, bucket_count, nb_sparse, nb_elements, nb_deleted
    // as u64 each, then max_load_factor as f32.
    let mut map: SparseMap<i64, i64> = SparseMap::new();
    for i in 0..100i64 {
        map.insert(i, i);
    }
    let good = serialize_bytes(&map);

    // A sparse_bucket_size larger than the platform bitmap width is rejected.
    // The first per-array size field sits after the 5 u64 fields and the f32.
    {
        let mut bytes = serialize_bytes(&SparseMap::<i64, i64>::from([(1, 1)]));
        let off = 5 * 8 + 4;
        bytes[off..off + 8].copy_from_slice(&999u64.to_le_bytes());
        let mut r = VecDeserializer::new(&bytes);
        let out = SparseMap::<i64, i64>::deserialize_with(&mut r, true, StdHash::default(), StdEq);
        assert!(out.is_err(), "oversized sparse bucket must fail");
    }

    // A bucket_count that the growth policy would round to a different value is
    // rejected under hash_compatible. 300 is not a power of two.
    {
        let mut bytes = good.clone();
        bytes[8..16].copy_from_slice(&300u64.to_le_bytes());
        let mut r = VecDeserializer::new(&bytes);
        let out = SparseMap::<i64, i64>::deserialize_with(&mut r, true, StdHash::default(), StdEq);
        assert!(out.is_err(), "growth policy mismatch must fail");
    }

    // A max_load_factor below the actual load factor is rejected. Forcing it to
    // 0.1 while the table is fuller trips the guard.
    {
        let mut bytes = good.clone();
        bytes[40..44].copy_from_slice(&0.1f32.to_le_bytes());
        let mut r = VecDeserializer::new(&bytes);
        let out = SparseMap::<i64, i64>::deserialize_with(&mut r, true, StdHash::default(), StdEq);
        assert!(out.is_err(), "load factor guard must fail");
    }

    // A bitmap_vals whose popcount disagrees with the value count is rejected.
    // Set a bit in the first array's value bitmap while its size field stays 0.
    // The bitmap_vals field is the second u64 of the first per-array record.
    {
        let mut bytes = good.clone();
        let bitmap_off = 5 * 8 + 4 + 8;
        bytes[bitmap_off..bitmap_off + 8].copy_from_slice(&1u64.to_le_bytes());
        let mut r = VecDeserializer::new(&bytes);
        let out = SparseMap::<i64, i64>::deserialize_with(&mut r, true, StdHash::default(), StdEq);
        assert!(out.is_err(), "bitmap popcount mismatch must fail");
    }

    // A slot marked both present and deleted is rejected. Copy the value bitmap
    // of the first array into its deleted bitmap so the two overlap.
    {
        let mut bytes = good.clone();
        let vals_off = 5 * 8 + 4 + 8;
        let del_off = vals_off + 8;
        let vals = {
            let mut a = [0u8; 8];
            a.copy_from_slice(&bytes[vals_off..vals_off + 8]);
            a
        };
        bytes[del_off..del_off + 8].copy_from_slice(&vals);
        // Only meaningful when the first array holds a value. Skip the assert if
        // the first array is empty, since then the bitmaps are both zero.
        if u64::from_le_bytes(vals) != 0 {
            let mut r = VecDeserializer::new(&bytes);
            let out =
                SparseMap::<i64, i64>::deserialize_with(&mut r, true, StdHash::default(), StdEq);
            assert!(out.is_err(), "present-and-deleted slot must fail");
        }
    }
}

#[test]
fn test_unicode_and_boundary_string_keys() {
    // Empty, multibyte, and long keys must round-trip through insert, lookup,
    // and erase without truncation or collision.
    let mut map: SparseMap<String, i64> = SparseMap::new();
    let keys: Vec<String> = ["", "café", "naïve", "日本語", "🦀rust"]
        .iter()
        .map(|s| s.to_string())
        .chain(std::iter::once("a".repeat(1000)))
        .collect();

    for (i, k) in keys.iter().enumerate() {
        assert!(map.insert(k.clone(), i as i64));
    }
    assert_eq!(map.len(), keys.len());
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(map.get(k.as_str()), Some(&(i as i64)));
    }

    assert_eq!(map.get(""), Some(&0));
    assert_eq!(map.erase(""), 1);
    assert!(map.get("").is_none());
    assert_eq!(map.len(), keys.len() - 1);
}

#[test]
fn test_max_size_and_max_bucket_count() {
    let map: SparseMap<i64, i64> = SparseMap::new();
    // Both limits report a large capacity for a fresh map.
    assert!(map.max_size() > 1000);
    assert!(map.max_bucket_count() > 1000);
}
