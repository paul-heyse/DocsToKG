# Real Hybrid Search Fixture

This directory contains a deterministic sample of chunk/vector artifacts used for
real-vector regression tests. The fixture was generated with the following parameters:

- Namespaces: `operations, research, support`
- Sample size: `128`
- Seed: `20241014`
- Max chunks per document: `0`

To regenerate the fixture, run:

```bash
python scripts/build_real_hybrid_fixture.py --seed 20241014 --sample-size 128 --namespaces operations,research,support --max-chunks-per-doc 0
```

Ensure the `Data/ChunkedDocTagFiles` and `Data/Vectors` directories are populated
before regenerating. The builder records source file hashes so changes can be
audited when refreshing the fixture.
