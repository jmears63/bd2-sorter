# bd2-sorter

`bd2_sorter` scans a directory of `.wav` recordings, runs BatDetect2 on each file,
and writes:

- raw JSON output per recording in `raw`
- a row per file in `csv/raw_combined.csv`, listing probabilites by species
- sets of symlinks under the "matched" subdirectory for detections

## Usage

```bash
bd2_sorter /path/to/wav_directory
```

Options:

- `-v`, `--verbose`: print active BatDetect2 config and per-file matched class map
- `-t`, `--threshold <float>`: probability threshold for class matching (default `0.5`)
- `-j`, `--junk`: move files with no over-threshold detections into `junk/`

Examples:

```bash
bd2_sorter -v /path/to/wav_directory
bd2_sorter -t 0.7 /path/to/wav_directory
bd2_sorter -j -t 0.6 /path/to/wav_directory
```

## Output layout

For an input directory `INPUT_DIR`, the tool creates:

- `INPUT_DIR/raw/` with files named `<original.wav>.json`
- `INPUT_DIR/csv/raw_combined.csv`
- `INPUT_DIR/csv/summary.csv`
- `INPUT_DIR/matched/all/`
- `INPUT_DIR/matched/<class_key>/` for each class key that meets threshold

Class keys are generated from class names by taking the first 3 letters of each
word, lowercasing, and removing spaces. Example:

- `Barbastellus barbastellus` -> `barbar`

For each processed wav file and each matched class key, symlinks are created in:

- `matched/all/`
- each relevant `matched/<class_key>/`
