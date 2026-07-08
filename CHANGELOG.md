# Changelog

## [0.2.0] - 2026-07-07

### Changed
- Insert sequences that forced a full probe cycle during `Mod` rehash now grow the table and complete. (#23)
- Deserialization now rejects element counts that cannot fit or reserve on the current platform. (#24)
- `Mod` growth policies with a zero denominator now fail during construction instead of building an invalid policy. (#25)

### Documentation
- `Default` docs for maps and sets now state the actual hasher, comparator, and growth policy requirements. (#26)

## [0.2.0] - 2026-07-07

### Changed
- Insert sequences that forced a full probe cycle during `Mod` rehash now grow the table and complete. (#23)
- Deserialization now rejects element counts that cannot fit or reserve on the current platform. (#24)
- `Mod` growth policies with a zero denominator now fail during construction instead of building an invalid policy. (#25)

### Documentation
- `Default` docs for maps and sets now state the actual hasher, comparator, and growth policy requirements. (#26)
