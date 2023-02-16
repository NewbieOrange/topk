use ahash::{AHasher, RandomState};
use priority_queue::DoublePriorityQueue;
use std::cmp::Ordering;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};

/// A counter for element occurrences with associated error (if present).
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
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

type MonitoredList<T> = DoublePriorityQueue<T, ElementCounter, RandomState>;

/// A filtered space-saving structure containing the current Top-K elements.
///
/// The elements is of type T, which must implement `Eq` and `Hash`.
///
/// The space-saving algorithm guarantees the following:
/// 1. `estimated_count` >= `exact_count`
/// 2. `estimated_count` - `associated_error` <= `exact_count`
#[derive(Clone)]
pub struct FilteredSpaceSaving<T: Eq + Hash> {
    k: usize,
    monitored_list: MonitoredList<T>,
    alphas: Vec<u64>,
    count: u64,
}

impl<T: Eq + Hash> FilteredSpaceSaving<T> {
    /// Creates an empty filtered space-saving structure with pre-allocated space for `k` elements.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            monitored_list: MonitoredList::with_capacity_and_default_hasher(k),
            alphas: vec![0; 6 * k],
            count: 0,
        }
    }

    /// Inserts the item `x` for `count` times.
    ///
    /// Computes in **O(log(K))** time.
    pub fn insert(&mut self, x: T, count: u64) {
        self.count += count;

        if self.monitored_list.change_priority_by(&x, |e| {
            e.estimated_count += count;
        }) {
            return;
        }
        if self.monitored_list.len() < self.k {
            self.monitored_list.push(x, ElementCounter::new(count, 0));
            return;
        }

        let x_hash = Self::alpha_hash(&x, self.alphas.len());
        let (min_elem, min_counter) = self.monitored_list.peek_min().unwrap();
        if self.alphas[x_hash] + count < min_counter.estimated_count {
            self.alphas[x_hash] += count;
            return;
        }

        let m_hash = Self::alpha_hash(min_elem, self.alphas.len());
        self.alphas[m_hash] = min_counter.estimated_count;

        self.monitored_list.pop_min();
        self.monitored_list.push(
            x,
            ElementCounter::new(self.alphas[x_hash] + count, self.alphas[x_hash]),
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
            .and_then(|(_, v)| Some(*v))
            .unwrap_or_else(|| {
                let count = self.alphas[Self::alpha_hash(&x, self.alphas.len())];
                ElementCounter::new(count, count)
            })
    }

    /// Merges with `other` filtered space-saving approximation.
    ///
    /// Require `T` to implement `Clone`.
    ///
    /// Merging of different `k` values will result in an `InvalidMergeError`.
    ///
    /// ref: <https://ieeexplore.ieee.org/document/8438445>
    ///
    /// Computes in **O(K*log(K))** time.
    pub fn merge(&mut self, other: &FilteredSpaceSaving<T>) -> Result<(), InvalidMergeError> where T: Clone {
        if self.k != other.k {
            return Err(InvalidMergeError {
                expect: self.k,
                actual: other.k,
            });
        }
        self.count += other.count;
        for (key, value) in self.monitored_list.iter_mut() {
            if let Some((_, e)) = other.monitored_list.get(key) {
                value.estimated_count += e.estimated_count;
                value.associated_error += e.associated_error;
            } else {
                let k_hash = Self::alpha_hash(key, other.alphas.len());
                let a2 = other.alphas[k_hash];
                value.estimated_count += a2;
                value.associated_error += a2;
            }
        }
        for (key, value) in other.monitored_list.iter() {
            if self.monitored_list.get(key).is_some() {
                continue;
            }
            let k_hash = Self::alpha_hash(key, self.alphas.len());
            let a1 = self.alphas[k_hash];
            let e = ElementCounter::new(value.estimated_count + a1, value.associated_error + a1);
            let (_, min_counter) = self.monitored_list.peek_min().unzip();
            if min_counter.map_or(true, |m| m < &e) {
                if self.monitored_list.len() >= self.k {
                    self.monitored_list.pop_min();
                }
                self.monitored_list.push(key.clone(), e);
            }
        }
        for (i, v) in other.alphas.iter().enumerate() {
            self.alphas[i] += v;
        }
        Ok(())
    }

    /// Returns an iterator in arbitrary order over the Top-K items.
    pub fn iter(&self) -> impl Iterator<Item=(&T, &ElementCounter)> {
        self.monitored_list.iter()
    }

    /// Consumes the `FilteredSpaceSaving` and return an iterator in arbitrary order over the Top-K items.
    pub fn into_iter(self) -> impl Iterator<Item=(T, ElementCounter)> {
        self.monitored_list.into_iter()
    }

    /// Consumes the `FilteredSpaceSaving` and return a `Vec` with Top-K items and counters in descending order (top items first).
    ///
    /// Computes in **O(K*log(K))** time.
    pub fn into_sorted_vec(mut self) -> Vec<(T, ElementCounter)> {
        let mut result = Vec::with_capacity(self.monitored_list.len());
        while let Some(e) = self.monitored_list.pop_max() {
            result.push(e);
        }
        result
    }

    /// Consumes the `FilteredSpaceSaving` and return a `DoubleEndedIterator` with Top-K items and counters in descending order (top items first).
    ///
    /// Each consumption computes in **O(log(K))** time.
    pub fn into_sorted_iter(self) -> impl DoubleEndedIterator<Item=(T, ElementCounter)> {
        self.monitored_list.into_sorted_iter().rev()
    }

    /// Returns count of all seen items (sum of all inserted `count`).
    ///
    /// Computes in **O(1)** time.
    pub fn count(&self) -> u64 {
        self.count
    }

    fn reduce(x: u64, n: u64) -> usize {
        (x as u128 * n as u128 >> 64) as usize
    }

    fn alpha_hash(x: &T, n: usize) -> usize {
        let mut hasher = AHasher::default();
        x.hash(&mut hasher);
        Self::reduce(hasher.finish(), n as u64)
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
        fss2.merge(&fss1).unwrap();
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
}
