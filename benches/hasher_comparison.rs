//! Benchmarks comparing alpha hasher performance.
//!
//! # Running benchmarks
//!
//! ```sh
//! # Run with default hasher (AHasher)
//! cargo bench
//!
//! # Run with serde feature to compare AHasher vs StableSipHasher128
//! cargo bench --features serde
//! ```
//!
//! Results are saved to `target/criterion/` and can be viewed in a browser
//! by opening `target/criterion/report/index.html`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use topk::FilteredSpaceSaving;

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");

    for k in [100, 1000, 10000] {
        // Pre-generate keys outside the benchmark loop
        let keys: Vec<String> = (0..k * 2)
            .map(|i| format!("item_{}", i % (k * 3)))
            .collect();

        group.bench_with_input(BenchmarkId::new("ahash", k), &keys, |b, keys| {
            b.iter(|| {
                let mut fss: FilteredSpaceSaving<String, ahash::AHasher> =
                    FilteredSpaceSaving::with_hasher(k);
                for key in keys {
                    fss.insert(key.clone(), 1);
                }
                black_box(fss)
            })
        });

        #[cfg(feature = "serde")]
        group.bench_with_input(BenchmarkId::new("stable", k), &keys, |b, keys| {
            b.iter(|| {
                let mut fss: FilteredSpaceSaving<String, rustc_stable_hash::StableSipHasher128> =
                    FilteredSpaceSaving::with_hasher(k);
                for key in keys {
                    fss.insert(key.clone(), 1);
                }
                black_box(fss)
            })
        });
    }
    group.finish();
}

fn bench_estimate(c: &mut Criterion) {
    let mut group = c.benchmark_group("estimate");

    let k = 1000;
    // Pre-generate keys
    let setup_keys: Vec<String> = (0..k * 2).map(|i| format!("item_{}", i)).collect();
    let query_keys: Vec<String> = (0..1000).map(|i| format!("item_{}", i)).collect();

    // Setup: pre-populate with items
    let mut fss_ahash: FilteredSpaceSaving<String, ahash::AHasher> =
        FilteredSpaceSaving::with_hasher(k);
    for key in &setup_keys {
        fss_ahash.insert(key.clone(), 1);
    }

    group.bench_function("ahash", |b| {
        b.iter(|| {
            for key in &query_keys {
                black_box(fss_ahash.estimate(key));
            }
        })
    });

    #[cfg(feature = "serde")]
    {
        let mut fss_stable: FilteredSpaceSaving<String, rustc_stable_hash::StableSipHasher128> =
            FilteredSpaceSaving::with_hasher(k);
        for key in &setup_keys {
            fss_stable.insert(key.clone(), 1);
        }

        group.bench_function("stable", |b| {
            b.iter(|| {
                for key in &query_keys {
                    black_box(fss_stable.estimate(key));
                }
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_insert, bench_estimate);
criterion_main!(benches);
