# TopK

TopK algorithm implementation in Rust.

This crate currently provides the [Filtered Space-Saving algorithm](https://doi.org/10.1016/j.ins.2010.08.024).

### Usage

To use this crate, simply add this as dependency in your `Cargo.toml`:

```toml
topk = "0.1"
```

Version numbers follow the [semver](https://semver.org/) convention.

### Example

```rust
let mut topk = FilteredSpaceSaving::new(3);
topk.insert("1", 10);
topk.insert("2", 20);
topk.insert("3", 1);
topk.insert("4", 2);
let topk_result = topk.to_vec();
assert_eq!(topk_result.len(), 3);
assert_eq!(topk_result[0].0, "1");
```

merging space-saving results are supported:

```rust
let mut fss1 = FilteredSpaceSaving::new(3);
fss1.insert("1", 10);
fss1.insert("2", 20);
fss1.insert("3", 2);
fss1.insert("4", 1);
fss1.insert("4", 3);
fss1.insert("5", 3);
let mut fss2 = FilteredSpaceSaving::new(3);
fss1.insert("1", 10);
fss1.insert("2", 20);
fss1.insert("3", 20);
fss1.insert("4", 10);
fss1.merge( & fss2).unwrap();
let result = fss1.into_vec();
assert_eq!(result[0].0, "2");
```
