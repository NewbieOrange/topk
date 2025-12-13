#[cfg(not(feature = "serde"))]
use ahash::AHasher;
use ahash::RandomState;
#[cfg(feature = "serde")]
use rustc_stable_hash::StableSipHasher128;
use priority_queue::PriorityQueue;
use std::cmp::{Ordering, Reverse};
use std::fmt;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

/// Default hasher for alpha buckets. Uses stable hash when serde is enabled
/// for cross-platform compatibility; uses fast AHasher otherwise.
#[cfg(feature = "serde")]
pub type DefaultAlphaHasher = StableSipHasher128;

/// Default hasher for alpha buckets. Uses stable hash when serde is enabled
/// for cross-platform compatibility; uses fast AHasher otherwise.
#[cfg(not(feature = "serde"))]
pub type DefaultAlphaHasher = AHasher;

/// A counter for element occurrences with associated error (if present).
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ElementCounter {
    estimated_count: u64,
    associated_error: u64,
}

impl ElementCounter {
    fn new(estimated_count: u64, associated_error: u64) -> Self {
        ElementCounter {
            estimated_count,
            associated_error,
        }
    }

    /// Returns the estimated element occurrence count.
    pub fn estimated_count(&self) -> u64 {
        self.estimated_count
    }

    /// Returns the associated occurrence count error.
    pub fn associated_error(&self) -> u64 {
        self.associated_error
    }
}

impl Ord for ElementCounter {
    fn cmp(&self, other: &Self) -> Ordering {
        self.estimated_count
            .cmp(&other.estimated_count)
            .then(other.associated_error.cmp(&self.associated_error))
    }
}

impl PartialOrd for ElementCounter {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

type MonitoredList<T> = PriorityQueue<T, Reverse<ElementCounter>, RandomState>;

const ALPHAS_FACTOR: usize = 6;

/// A filtered space-saving structure containing the current Top-K elements.
///
/// The elements is of type T, which must implement `Eq` and `Hash`.
///
/// The hasher H is used for the alpha bucket hash function. The default is
/// `AHasher` for performance. When the `serde` feature is enabled, the default
/// is `StableSipHasher128` for cross-platform determinism. This reduces
/// performance of common operations by 20–25%.
///
/// The space-saving algorithm guarantees the following:
/// 1. `estimated_count` >= `exact_count`
/// 2. `estimated_count` - `associated_error` <= `exact_count`
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FilteredSpaceSaving<T: Eq + Hash, H: Hasher + Default = DefaultAlphaHasher> {
    k: usize,
    monitored_list: MonitoredList<T>,
    alphas: Vec<u64>,
    count: u64,
    #[cfg_attr(feature = "serde", serde(skip))]
    _hasher: PhantomData<H>,
}

// `new` implicitly forces the use of DefaultAlphaHasher for convenience and
// backwards compatibility.
impl<T: Eq + Hash> FilteredSpaceSaving<T, DefaultAlphaHasher> {
    /// Creates an empty filtered space-saving structure with pre-allocated space for `k` elements.
    ///
    /// Uses `DefaultAlphaHasher` for alpha bucket hashing, which is `AHasher` by default
    /// for performance, or `StableSipHasher128` when the `serde` feature is enabled for
    /// cross-platform determinism.
    pub fn new(k: usize) -> Self {
        Self::with_hasher(k)
    }
}

// Other methods inherit their default from the `FilteredSpaceSaving` struct or
// from context. We provide `with_hasher` to override the default hasher. This 
// is the same pattern used by Rust's `HashMap`.
impl<T: Eq + Hash, H: Hasher + Default> FilteredSpaceSaving<T, H> {
    /// Creates an empty filtered space-saving structure with a custom hasher.
    pub fn with_hasher(k: usize) -> Self {
        Self {
            k,
            monitored_list: MonitoredList::with_capacity_and_default_hasher(k),
            alphas: vec![0; ALPHAS_FACTOR * k],
            count: 0,
            _hasher: PhantomData,
        }
    }

    /// Inserts the item `x` for `count` times.
    ///
    /// Computes in **O(log(k))** time.
    pub fn insert(&mut self, x: T, count: u64) {
        self.count += count;

        if self.monitored_list.change_priority_by(&x, |e| e.0.estimated_count += count) {
            return;
        }
        if self.monitored_list.len() < self.k {
            self.monitored_list.push(x, Reverse(ElementCounter::new(count, 0)));
            return;
        }

        let x_hash = self.alpha_hash(&x);
        let (min_elem, min_counter) = self.monitored_list.peek().unwrap();
        if self.alphas[x_hash] + count < min_counter.0.estimated_count {
            self.alphas[x_hash] += count;
            return;
        }

        let m_hash = self.alpha_hash(min_elem);
        self.alphas[m_hash] = min_counter.0.estimated_count;

        self.monitored_list.pop();
        self.monitored_list.push(
            x,
            Reverse(ElementCounter::new(self.alphas[x_hash] + count, self.alphas[x_hash])),
        );
    }

    /// Estimates the occurrences of the item `x`.
    ///
    /// If the item is in the Top-K approximation, the approximation is returned.
    ///
    /// Otherwise, a rough approximation is returned (`error` == `estimate`).
    ///
    /// Computes in **O(1)** time.
    pub fn estimate(&self, x: &T) -> ElementCounter {
        self.monitored_list
            .get(x)
            .and_then(|(_, v)| Some(v.0))
            .unwrap_or_else(|| {
                let count = self.alphas[self.alpha_hash(&x)];
                ElementCounter::new(count, count)
            })
    }

    /// Estimates the occurrences of the item `x` if the item is in the Top-K approximation.
    /// Otherwise, `None` is returned.
    ///
    /// Computes in **O(1)** time.
    pub fn get(&self, x: &T) -> Option<ElementCounter> {
        self.monitored_list.get(x).and_then(|(_, v)| Some(v.0))
    }

    /// Merges with `other` filtered space-saving approximation.
    ///
    /// Require `T` to implement `Clone`.
    ///
    /// Merging of different `k` values will result in an `InvalidMergeError`.
    ///
    /// ref: <https://ieeexplore.ieee.org/document/8438445>
    ///
    /// Computes in **O(k*log(k))** time.
    pub fn merge(&mut self, other: &FilteredSpaceSaving<T, H>) -> Result<(), InvalidMergeError> where T: Clone {
        if self.k != other.k {
            return Err(InvalidMergeError {
                expect: self.k,
                actual: other.k,
            });
        }
        self.count += other.count;
        for (key, value) in self.monitored_list.iter_mut() {
            if let Some((_, e)) = other.monitored_list.get(key) {
                value.0.estimated_count += e.0.estimated_count;
                value.0.associated_error += e.0.associated_error;
            } else {
                let k_hash = other.alpha_hash(key);
                let a2 = other.alphas[k_hash];
                value.0.estimated_count += a2;
                value.0.associated_error += a2;
            }
        }
        for (key, value) in other.monitored_list.iter() {
            if self.monitored_list.get(key).is_some() {
                continue;
            }
            let k_hash = self.alpha_hash(key);
            let a1 = self.alphas[k_hash];
            let e = Reverse(ElementCounter::new(value.0.estimated_count + a1, value.0.associated_error + a1));
            if self.monitored_list.len() < self.k {
                // We have fewer than k items, so add a new item.
                self.monitored_list.push(key.clone(), e);
            } else if self.monitored_list.peek().map_or(true, |(_, m)| m.0 < e.0) {
                // We want to evict an item and replace it with this one, but we
                // need to update the error accordingly.
                let popped = self.monitored_list.pop().expect("monitored_list should not be empty");
                let p_hash = self.alpha_hash(&popped.0);
                self.alphas[p_hash] += popped.1.0.estimated_count;
                self.monitored_list.push(key.clone(), e);
            } else {
                // This item is not in the Top-K and cannot replace any existing
                // item, but we still need to track the error.
                self.alphas[k_hash] = self.alphas[k_hash].max(e.0.estimated_count);
            }
        }
        for (i, v) in other.alphas.iter().enumerate() {
            self.alphas[i] += v;
        }
        Ok(())
    }

    /// Decays the counters, equivalent to multiplying all counters by the factor.
    ///
    /// Computes in **O(k)** time.
    pub fn decay(&mut self, factor: f64) {
        for (_, value) in self.monitored_list.iter_mut() {
            value.0.estimated_count = (value.0.estimated_count as f64 * factor) as u64;
            value.0.associated_error = (value.0.associated_error as f64 * factor) as u64;
        }
        self.alphas.iter_mut().for_each(|x| *x = (*x as f64 * factor) as u64);
        self.count = (self.count as f64 * factor) as u64
    }

    /// Clears the counter, resetting count to 0.
    ///
    /// Note that this method has no effect on the `k` of the counter.
    pub fn clear(&mut self) {
        self.monitored_list.clear();
        self.alphas.iter_mut().for_each(|x| *x = 0);
        self.count = 0;
    }

    /// Returns an iterator in arbitrary order over the Top-K items.
    pub fn iter(&self) -> impl Iterator<Item=(&T, &ElementCounter)> {
        self.monitored_list.iter().map(|(k, v)| (k, &v.0))
    }

    /// Consumes the `FilteredSpaceSaving` and return an iterator in arbitrary order over the Top-K items.
    pub fn into_iter(self) -> impl Iterator<Item=(T, ElementCounter)> {
        self.monitored_list.into_iter().map(|(k, v)| (k, v.0))
    }

    /// Consumes the `FilteredSpaceSaving` and return a `Vec` with Top-K items and counters in descending order (top items first).
    ///
    /// Computes in **O(k*log(k))** time.
    pub fn into_sorted_vec(self) -> Vec<(T, ElementCounter)> {
        let mut result = Vec::with_capacity(self.monitored_list.len());
        result.extend(self.monitored_list.into_sorted_iter().map(|(k, v)| (k, v.0)));
        result.reverse();
        result
    }

    /// Consumes the `FilteredSpaceSaving` and return a `DoubleEndedIterator` with Top-K items and counters in descending order (top items first).
    ///
    /// Computes in **O(k*log(k))** time.
    pub fn into_sorted_iter(self) -> impl DoubleEndedIterator<Item=(T, ElementCounter)> {
        self.into_sorted_vec().into_iter()
    }

    /// Returns count of all seen items (sum of all inserted `count`).
    ///
    /// Computes in **O(1)** time.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns the `k` value of the counter.
    pub fn k(&self) -> usize {
        self.k
    }

    fn alpha_hash(&self, x: &T) -> usize {
        let mut hasher = H::default();
        x.hash(&mut hasher);
        (hasher.finish() as u128 * self.alphas.len() as u128 >> 64) as usize
    }
}

#[derive(Debug, Clone)]
pub struct InvalidMergeError {
    expect: usize,
    actual: usize,
}

impl Display for InvalidMergeError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "expected merge with same k {}, got {}",
            self.expect, self.actual
        )
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn test_topk() {
        let mut fss = FilteredSpaceSaving::new(3);
        fss.insert("1", 10);
        fss.insert("2", 20);
        fss.insert("3", 2);
        fss.insert("4", 1);
        fss.insert("4", 3);
        fss.insert("5", 5);
        let result = fss.into_sorted_vec();
        assert_eq!(result[0].0, "2");
        assert_eq!(result[0].1, ElementCounter::new(20, 0));
        assert_eq!(result[1].0, "1");
        assert_eq!(result[1].1, ElementCounter::new(10, 0));
        assert_eq!(result[2].0, "5");
        assert!(result[2].1.estimated_count >= 5);
    }

    #[test]
    fn test_merge() {
        let mut fss1 = FilteredSpaceSaving::new(3);
        fss1.insert("1", 10);
        fss1.insert("2", 20);
        fss1.insert("3", 2);
        fss1.insert("4", 1);
        fss1.insert("4", 3);
        fss1.insert("5", 5);
        let mut fss2 = FilteredSpaceSaving::new(3);
        fss2.insert("1", 10);
        fss2.insert("2", 20);
        fss2.insert("3", 20);
        fss2.insert("4", 10);
        let total_count = fss1.count + fss2.count;
        fss2.merge(&fss1).unwrap();
        assert_eq!(fss2.count, total_count);
        let result = fss2.into_sorted_vec();
        assert_eq!(result[0].0, "2");
        assert_eq!(result[0].1, ElementCounter::new(40, 0));
        assert_eq!(result[1].0, "3");
        assert!(result[1].1.estimated_count + result[1].1.associated_error >= 22);
        assert_eq!(result[2].0, "1");
        assert_eq!(result[2].1, ElementCounter::new(20, 10));
    }

    #[test]
    fn test_merge_bad() {
        let mut fss1 = FilteredSpaceSaving::new(1);
        fss1.insert(0, 0);
        let mut fss2 = FilteredSpaceSaving::new(2);
        fss2.insert(0, 0);
        assert!(fss1.merge(&fss2).is_err());
    }

    #[test]
    fn test_clear() {
        let mut fss = FilteredSpaceSaving::new(3);
        fss.insert("1", 10);
        fss.insert("2", 20);
        fss.insert("3", 2);
        fss.insert("4", 1);
        fss.insert("4", 3);
        fss.insert("5", 5);
        fss.clear();
        assert_eq!(fss.k, 3);
        assert_eq!(fss.count, 0);
        for x in fss.alphas {
            assert_eq!(x, 0);
        }
        assert!(fss.monitored_list.is_empty());
    }

    fn topk_of<'a>(fss: &'a FilteredSpaceSaving<&str>) -> Vec<(&'a str, u64)> {
        let mut result = Vec::with_capacity(fss.monitored_list.len());
        result.extend(fss.monitored_list.iter().map(|(k, v)| (k, &v.0)));
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result.iter().map(|(&name, &counter)| (name, counter.estimated_count())).collect::<Vec<_>>()
    }

    #[test]
    fn test_decay() {
        let mut fss = FilteredSpaceSaving::new(3);
        fss.insert("a", 10);
        fss.insert("b", 20);
        fss.insert("c", 6);
        fss.insert("d", 8);
        fss.insert("a", 2);

        assert_eq!(fss.count(), 46);
        assert_eq!(topk_of(&fss), vec![("b", 20), ("a", 12), ("d", 8)]);

        fss.decay(0.5);
        assert_eq!(fss.count(), 23);
        assert_eq!(topk_of(&fss), vec![("b", 10), ("a", 6), ("d", 4)]);

        // make sure we did not lose the alphas of items outside of the top-k
        assert_eq!(fss.estimate(&"c").estimated_count(), 3);
        fss.insert("c", 2);
        assert_eq!(fss.estimate(&"c").estimated_count(), 5);
        assert_eq!(topk_of(&fss), vec![("b", 10), ("a", 6), ("c", 5)]);
    }

    #[test]
    fn test_merge_into_empty() {
        let mut empty = FilteredSpaceSaving::new(100);
        let mut other = FilteredSpaceSaving::new(100);

        for i in 0..50 {
            other.insert(i, (i % 10 + 1) as u64);
        }

        empty.merge(&other).unwrap();
        assert_eq!(empty.iter().count(), 50);
    }

    #[test]
    fn test_merge_underfull() {
        let mut a = FilteredSpaceSaving::new(100);
        let mut b = FilteredSpaceSaving::new(100);

        for i in 0..30 { a.insert(i, 10); }
        for i in 30..60 { b.insert(i, 5); }  // Lower counts than a's minimum

        a.merge(&b).unwrap();
        assert_eq!(a.iter().count(), 60);  // Should have all 60 items
    }

    // =======================================================================
    // Property tests
    //
    // These tests generate hundreds of individual test cases, and check that
    // certain algorithmic invariants hold. If a failure is found, proptest
    // will try to "shrink" the failure to find a minimal example, which can
    // then be added as a regular test case.
    //
    // These tests verify the Lemma 3 invariants from the original
    // Space-Saving paper (Metwally et al., 2005).
    //
    // ### Lemma 3 Guarantees
    //
    // For any monitored element with true frequency `f` and estimated count
    // `c` with error `ε`:
    //
    // - `0 ≤ ε ≤ min` (error bounded by minimum count in structure)
    // - `f ≤ c ≤ f + min` (estimated count never underestimates)
    //
    // Testable properties:
    //
    // 1. `estimated_count >= exact_count` (never underestimates)
    // 2. `estimated_count - associated_error <= exact_count` (error bound is
    //     conservative)

    /// Weighted key distribution approximating Zipf's law (common English
    /// words)
    fn weighted_key() -> impl Strategy<Value = &'static str> {
        prop_oneof![
            100 => Just("the"),
            70 => Just("of"),
            60 => Just("and"),
            55 => Just("to"),
            50 => Just("a"),
            45 => Just("in"),
            35 => Just("is"),
            30 => Just("it"),
            25 => Just("for"),
            20 => Just("that"),
        ]
    }

    /// (key, count) pair for insertion
    fn weighted_insertion() -> impl Strategy<Value = (&'static str, u64)> {
        (weighted_key(), 0u64..=5)
    }

    proptest! {
        /// Verify Lemma 3 invariants hold after insert operations
        #[test]
        fn test_insert_lemma3_invariants(
            k in 1usize..=12,
            insertions in prop::collection::vec(weighted_insertion(), 0..25)
        ) {
            let mut fss = FilteredSpaceSaving::new(k);
            let mut exact_counts: HashMap<&str, u64> = HashMap::new();

            for (key, count) in insertions {
                fss.insert(key, count);
                *exact_counts.entry(key).or_default() += count;
            }

            // Check Lemma 3 for ALL items via estimate()
            for (key, &exact) in &exact_counts {
                let estimate = fss.estimate(key);
                let estimated = estimate.estimated_count();
                let error = estimate.associated_error();

                // Invariant 1: Never underestimates
                prop_assert!(estimated >= exact,
                    "Underestimate: key={}, est={}, err={}, exact={} ", key, estimated, error, exact);

                // Invariant 2: Error bound is conservative
                // For evicted items (estimated == error): 0 <= exact (trivially true)
                // For monitored items: meaningful bound
                prop_assert!(estimated - error <= exact,
                    "Error bound violated: key={}, est={}, err={}, exact={}",
                    key, estimated, error, exact);
            }
        }

        /// Verify Lemma 3 invariants hold after merge operations
        #[test]
        fn test_merge_lemma3_invariants(
            k in 1usize..=12,
            insertions_a in prop::collection::vec(weighted_insertion(), 0..25),
            insertions_b in prop::collection::vec(weighted_insertion(), 0..25),
        ) {
            let mut fss_a = FilteredSpaceSaving::new(k);
            let mut fss_b = FilteredSpaceSaving::new(k);
            let mut exact_counts: HashMap<&str, u64> = HashMap::new();

            for (key, count) in insertions_a {
                fss_a.insert(key, count);
                *exact_counts.entry(key).or_default() += count;
            }
            for (key, count) in insertions_b {
                fss_b.insert(key, count);
                *exact_counts.entry(key).or_default() += count;
            }

            fss_a.merge(&fss_b).unwrap();

            // Check Lemma 3 for ALL items via estimate()
            // This catches underfull merge bug: wrongly-dropped items
            // will have estimate() return low alpha, failing estimated >= exact
            for (key, &exact) in &exact_counts {
                let estimate = fss_a.estimate(key);
                let estimated = estimate.estimated_count();
                let error = estimate.associated_error();

                prop_assert!(estimated >= exact,
                    "Merge underestimate: key={}, est={}, err={}, exact={}", key, estimated, error, exact);

                prop_assert!(estimated - error <= exact,
                    "Merge error bound violated: key={}, est={}, err={}, exact={}",
                    key, estimated, error, exact);
            }
        }

        /// Verify Lemma 3 invariants hold after serde roundtrip
        #[cfg(feature = "serde")]
        #[test]
        fn test_serde_roundtrip_and_lemma3_invariants(
            k in 1usize..=12,
            insertions in prop::collection::vec(weighted_insertion(), 0..25)
        ) {
            let mut fss = FilteredSpaceSaving::new(k);
            let mut exact_counts: HashMap<&str, u64> = HashMap::new();

            for (key, count) in insertions {
                fss.insert(key, count);
                *exact_counts.entry(key).or_default() += count;
            }

            // Serialize and deserialize
            let serialized = serde_json::to_string(&fss).unwrap();
            let deserialized: FilteredSpaceSaving<&str> = serde_json::from_str(&serialized).unwrap();

            // Verify basic properties preserved
            prop_assert_eq!(fss.count(), deserialized.count());
            prop_assert_eq!(fss.k(), deserialized.k());

            // Check Lemma 3 for ALL items via estimate() after roundtrip
            for (key, &exact) in &exact_counts {
                let orig = fss.estimate(key);
                let restored = deserialized.estimate(key);

                // Estimates should match exactly before and after
                prop_assert_eq!(orig.estimated_count(), restored.estimated_count(),
                    "Serde roundtrip changed estimate: key={}", key);
                prop_assert_eq!(orig.associated_error(), restored.associated_error(),
                    "Serde roundtrip changed error: key={}", key);

                // Lemma 3 invariants should hold. This is almost certainly
                // reundant with the insert tests, and exact dump/load tests
                // above, but we write it out explicitly for documentation
                // purposes, and to be extremely careful.
                let estimated = restored.estimated_count();
                let error = restored.associated_error();

                prop_assert!(estimated >= exact,
                    "Serde roundtrip underestimate: key={}, est={}, err={}, exact={}",
                    key, estimated, error, exact);

                prop_assert!(estimated - error <= exact,
                    "Serde roundtrip error bound violated: key={}, est={}, err={}, exact={}",
                    key, estimated, error, exact);
            }
        }
    }
}
