# bd2-sorter

`bd2_sorter` scans a directory of `.wav` recordings, runs BatDetect2 on each file,
and writes:

- raw JSON output per recording in `raw`
- threshold-filtered annotation rows in `csv/raw_combined.csv`
- per-file class summary in `csv/summary.csv`
- bucketed time profile in `csv/time_profile.csv`
- matched symlinks under `matched`

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
- `INPUT_DIR/csv/time_profile.csv`
- `INPUT_DIR/matched/all/`
- `INPUT_DIR/matched/<class_key>/` for each class key that meets threshold
- optionally `INPUT_DIR/junk/` when `-j` is enabled and a file has no matched classes

Class keys are generated from class names by taking the first 3 letters of each
word, lowercasing, and removing spaces. Example:

- `Barbastellus barbastellus` -> `barbar`

For each processed wav file and each matched class key, symlinks are created in:

- `matched/all/`
- each relevant `matched/<class_key>/`

`raw_combined.csv` columns are:

- `filename,time_exp,duration,class_name,start_time,end_time,low_freq,high_freq,class,class_prob,det_prob,individual,event`
