# sparse-hash-map

Open-addressing hash map and set that stay small at low load factor.

Empty slots in a flat hash table each cost the full size of a bucket. This crate
groups buckets into sparse arrays of up to 64 logical indices. Each array stores
only its present values, packed together, plus a bitmap marking which indices are
occupied. The dense position of an index is the number of occupied bits below it,
found with a population count. An empty logical bucket costs about one bit.

The result uses far less memory than a flat table when the table is sparsely
filled, and keeps lookup fast.

## Install

```toml
[dependencies]
sparse-hash-map = "0.1"
```

## Map

```rust
use sparse_hash_map::SparseMap;

let mut map: SparseMap<String, i32> = SparseMap::new();
map.insert("a".to_string(), 1);
map.insert("b".to_string(), 2);

assert_eq!(map.get("a"), Some(&1));
assert_eq!(map.len(), 2);

*map.get_mut("a").unwrap() = 10;
assert_eq!(map.get("a"), Some(&10));

for (k, v) in &map {
    println!("{k} = {v}");
}
```

Iteration yields `(&K, &V)`. The key is never mutable through an iterator, so it
cannot change under the map. To mutate a value, use `get_mut` or `iter_mut`.

## Set

```rust
use sparse_hash_map::SparseSet;

let mut set = SparseSet::new();
set.insert(3);
set.insert(7);

assert!(set.contains(&3));
assert_eq!(set.len(), 2);
```

## Growth policies

The default policy keeps the bucket count a power of two and maps a hash with a
mask. Two others are available:

- `Mod` grows by a rational factor and maps with a modulo.
- `Prime` steps through a table of primes. It spreads values better when the
  hash function is poor, such as an identity hash of pointers.

`SparsePgMap` and `SparsePgSet` are the prime-growth variants.

## Serialization

`serialize` writes a map or set through a `Serializer` you supply.
`deserialize_with` reads it back. You control the binary layout. The library
defines only the order and logical type of each field. Round trips work in two
modes: a portable mode that re-hashes on load, and a fast mode that restores slots
directly when the hasher and policy match.

## License

Licensed under the [MIT license](LICENSE).
