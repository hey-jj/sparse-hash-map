//! Allocation routing. Rust stable has no per-collection allocator, so this
//! uses a counting global allocator and checks that inserts allocate and do not
//! explode per element.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use sparse_hash_map::SparseMap;

struct CountingAlloc;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

#[test]
fn test_allocations_routed_and_bounded() {
    let before = ALLOCS.load(Ordering::Relaxed);

    let mut map: SparseMap<i64, i64> = SparseMap::new();
    let nb = 1000i64;
    for i in 0..nb {
        map.insert(i, i * 2);
    }
    assert_eq!(map.len(), nb as usize);

    let after = ALLOCS.load(Ordering::Relaxed);
    let allocations = after - before;

    // The map allocated for its buffers.
    assert!(allocations > 0, "map should allocate");
    // Growth reallocates in bounded steps. A small constant per element bounds
    // the total, unlike a scheme that allocates fresh storage on every insert.
    assert!(
        allocations < nb as usize * 3,
        "allocations {allocations} should stay bounded relative to {nb}"
    );
}
