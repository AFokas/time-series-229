# Data Guide

This guide explains how to add time-series data to this repository.

## Directory Structure

Place data files under a `data/` directory at the root of the repository:

```
data/
  raw/        # Original, unmodified source data
  processed/  # Cleaned or transformed data ready for analysis
```

## Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| CSV    | `.csv`    | Preferred for tabular time-series data |
| JSON   | `.json`   | Useful for nested or metadata-rich data |
| Parquet | `.parquet` | Recommended for large datasets |

## CSV Format Requirements

Every CSV file must have a timestamp column as the first column:

```
timestamp,value1,value2,...
2024-01-01T00:00:00Z,1.23,4.56
2024-01-01T01:00:00Z,1.45,4.78
```

- **`timestamp`**: ISO 8601 format (`YYYY-MM-DDTHH:MM:SSZ`). Must be monotonically increasing.
- **Value columns**: Numeric values for each measured variable.

## Adding Data

1. **Create the data directories** if they don't exist yet:
   ```bash
   mkdir -p data/raw data/processed
   ```

2. **Copy your data file** into the appropriate subdirectory:
   ```bash
   cp your_data.csv data/raw/
   ```

3. **Avoid committing large files** directly. For files larger than ~50 MB, use [Git LFS](https://git-lfs.github.com/):
   ```bash
   git lfs track "data/**/*.csv"
   git add .gitattributes
   ```

4. **Add a `.gitignore`** entry if your raw data is sensitive or too large to track:
   ```
   data/raw/
   ```

5. **Stage and commit** your data:
   ```bash
   git add data/
   git commit -m "Add raw time-series data for <description>"
   ```

## Metadata

For each dataset, add a companion `<filename>.meta.json` file alongside the data file describing its contents:

```json
{
  "description": "Short description of the dataset",
  "source": "Where the data came from",
  "frequency": "1h",
  "start": "2024-01-01T00:00:00Z",
  "end": "2024-12-31T23:00:00Z",
  "columns": {
    "timestamp": "UTC timestamp",
    "value1": "Description of value1 (units)"
  }
}
```

## Naming Conventions

- Use lowercase letters, numbers, and hyphens only: `sensor-readings-2024.csv`
- Include a date or version suffix when applicable: `weather-data-2024-01.csv`
- Avoid spaces and special characters in filenames.
