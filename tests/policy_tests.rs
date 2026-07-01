//! Growth-policy behavior: monotonic growth to a limit, zero-bucket handling,
//! and the maximum bucket count. Also asserts the exact deterministic sequences,
//! which is stronger than a monotonic-only check.

use sparse_hash_map::growth_policy::PRIMES_TABLE;
use sparse_hash_map::{GrowthPolicy, Mod, PowerOfTwo, Prime};

/// Iterate `next_bucket_count` from 0 until it errors. Each step must grow and
/// map hash 0 to bucket 0. The sequence must eventually error.
fn check_policy<P: GrowthPolicy>() {
    let (policy, mut bucket_count) = P::new(0).expect("zero bucket count is valid");
    assert_eq!(policy.bucket_for_hash(0), 0);
    assert_eq!(bucket_count, 0);

    let mut steps = 0;
    let errored = loop {
        let previous = bucket_count;
        let next = {
            let (p, _) = P::new(bucket_count).expect("valid bucket count");
            p.next_bucket_count()
        };
        match next {
            Ok(next) => {
                bucket_count = next;
                let (p, _settled) = P::new(bucket_count).expect("valid bucket count");
                assert_eq!(p.bucket_for_hash(0), 0);
                assert!(bucket_count > previous, "must strictly grow");
            }
            Err(_) => break true,
        }
        steps += 1;
        assert!(steps < 1000, "policy should reach its limit");
    };
    assert!(errored, "growth must eventually error");
}

fn check_min_bucket_count<P: GrowthPolicy>() {
    let (policy, _) = P::new(0).expect("zero bucket count is valid");
    assert_eq!(policy.bucket_for_hash(0), 0);
}

fn check_max_bucket_count<P: GrowthPolicy>() {
    let (policy, _) = P::new(0).expect("zero bucket count is valid");
    let max = policy.max_bucket_count();

    assert!(P::new(max).is_ok(), "max bucket count is valid");
    assert!(P::new(usize::MAX).is_err(), "usize::MAX must error");
    assert!(P::new(max + 1).is_err(), "max + 1 must error");
}

#[test]
fn policy_power_of_two_2() {
    check_policy::<PowerOfTwo<2>>();
    check_min_bucket_count::<PowerOfTwo<2>>();
    check_max_bucket_count::<PowerOfTwo<2>>();
}

#[test]
fn policy_power_of_two_4() {
    check_policy::<PowerOfTwo<4>>();
    check_min_bucket_count::<PowerOfTwo<4>>();
    check_max_bucket_count::<PowerOfTwo<4>>();
}

#[test]
fn policy_prime() {
    check_policy::<Prime>();
    check_min_bucket_count::<Prime>();
    check_max_bucket_count::<Prime>();
}

#[test]
fn policy_mod_default() {
    check_policy::<Mod>();
    check_min_bucket_count::<Mod>();
    check_max_bucket_count::<Mod>();
}

#[test]
fn policy_mod_7_2() {
    check_policy::<Mod<7, 2>>();
    check_min_bucket_count::<Mod<7, 2>>();
    check_max_bucket_count::<Mod<7, 2>>();
}

#[test]
fn power_of_two_exact_sequence() {
    // Starting from 1, the count doubles each growth.
    let (_, settled) = PowerOfTwo::<2>::new(1).unwrap();
    assert_eq!(settled, 1);
    let mut count = 1usize;
    let expected = [2, 4, 8, 16, 32, 64, 128, 256];
    for want in expected {
        let (p, _) = PowerOfTwo::<2>::new(count).unwrap();
        count = p.next_bucket_count().unwrap();
        assert_eq!(count, want);
    }
}

#[test]
fn power_of_two_4_exact_sequence() {
    let mut count = 1usize;
    let expected = [4, 16, 64, 256, 1024];
    for want in expected {
        let (p, _) = PowerOfTwo::<4>::new(count).unwrap();
        count = p.next_bucket_count().unwrap();
        assert_eq!(count, want);
    }
}

#[test]
fn prime_exact_sequence() {
    // next_bucket_count follows the prime table exactly.
    for i in 0..PRIMES_TABLE.len() - 1 {
        let (p, _) = Prime::new(PRIMES_TABLE[i]).unwrap();
        assert_eq!(p.next_bucket_count().unwrap(), PRIMES_TABLE[i + 1]);
    }
    // The last prime cannot grow.
    let (p, _) = Prime::new(PRIMES_TABLE[PRIMES_TABLE.len() - 1]).unwrap();
    assert!(p.next_bucket_count().is_err());
}

#[test]
fn prime_rounds_up_to_next_prime() {
    // A request between primes rounds up to the next one.
    let (_, settled) = Prime::new(6).unwrap();
    assert_eq!(settled, 17);
    let (_, settled) = Prime::new(100).unwrap();
    assert_eq!(settled, 131);
}

#[test]
fn mod_does_not_change_requested_count() {
    // Mod uses the requested count as the modulus, unchanged.
    let (_, settled) = Mod::<3, 2>::new(100).unwrap();
    assert_eq!(settled, 100);
    let (p, _) = Mod::<3, 2>::new(100).unwrap();
    assert_eq!(p.bucket_for_hash(250), 250 % 100);
}

#[test]
fn mod_growth_factor_3_2() {
    // Growth is ceil(count * 3 / 2).
    let (p, _) = Mod::<3, 2>::new(100).unwrap();
    assert_eq!(p.next_bucket_count().unwrap(), 150);
    let (p, _) = Mod::<3, 2>::new(150).unwrap();
    assert_eq!(p.next_bucket_count().unwrap(), 225);
}

#[test]
fn mod_growth_factor_7_2() {
    // Growth is ceil(count * 7 / 2).
    let (p, _) = Mod::<7, 2>::new(100).unwrap();
    assert_eq!(p.next_bucket_count().unwrap(), 350);
    let (p, _) = Mod::<7, 2>::new(10).unwrap();
    assert_eq!(p.next_bucket_count().unwrap(), 35);
}
