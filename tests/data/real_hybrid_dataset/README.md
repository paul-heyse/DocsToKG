# Real Hybrid Search Fixture

This directory contains a deterministic sample of chunk/vector artifacts used for
real-vector regression tests. The fixture was generated with the following parameters:

- Namespace: `real-fixture`
- Sample size: `3`
- Seed: `1337`

To regenerate the fixture, run:

```bash
python scripts/build_real_hybrid_fixture.py --seed 1337 --sample-size 3
```

Ensure the `Data/ChunkedDocTagFiles` and `Data/Vectors` directories are populated
before regenerating. The builder records source file hashes so changes can be
audited when refreshing the fixture.
