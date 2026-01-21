# Using EDA Module in Jupyter Notebooks

The EDA module automatically detects when it's running in a Jupyter notebook and displays plots inline while still saving them to disk.

## Quick Start

Run all analyses with plots displayed inline:

```python
import pandas as pd
from src.renewable.eda import (
    analyze_seasonality,
    analyze_zero_inflation,
    analyze_coverage_gaps,
    analyze_negative_values,
    analyze_weather_alignment,
    display_section
)
from pathlib import Path

# Load data
generation_df = pd.read_parquet("data/renewable/generation.parquet")
weather_df = pd.read_parquet("data/renewable/weather.parquet")

# Create output directory
output_dir = Path("reports/renewable/eda/notebook_demo")
output_dir.mkdir(parents=True, exist_ok=True)

# Run analyses with displayed plots
display_section("Seasonality Analysis",
    "Checking for 24h and 168h seasonal patterns...")
seasonality_results = analyze_seasonality(generation_df, output_dir / "seasonality")

display_section("Zero-Inflation Analysis",
    "Analyzing zero values to justify MAE/RMSE over MAPE...")
zero_results = analyze_zero_inflation(generation_df, output_dir / "zero_inflation")

display_section("Coverage & Missing Data",
    "Examining data completeness and gap patterns...")
coverage_results = analyze_coverage_gaps(generation_df, output_dir / "coverage")

display_section("Negative Values Analysis",
    "Critical analysis for preprocessing policy decisions...")
negative_results = analyze_negative_values(generation_df, output_dir / "negative_values")

display_section("Weather Alignment",
    "Validating weather feature correlations with generation...")
weather_results = analyze_weather_alignment(
    generation_df, weather_df, output_dir / "weather_alignment"
)

# Summary of findings
display_section("Summary of Findings")
print(f"Seasonality strength: {list(seasonality_results['hourly_seasonality_strength'].values())}")
print(f"Solar zero ratio: {sum(info['zero_ratio'] for uid, info in zero_results['series_zero_ratios'].items() if 'SUN' in uid) / max(1, sum(1 for uid in zero_results['series_zero_ratios'] if 'SUN' in uid)):.2%}")
print(f"Negatives found: {negative_results['negative_count']}")
print(f"Weather merge success: {weather_results['merge_success_ratio']:.1%}")
```

## Features

### Automatic Plot Display
- Plots are displayed inline in notebook cells
- Plots are saved to disk for later reference
- No need to call `plt.show()` manually

### Section Headers
Use `display_section()` for formatted markdown headers in notebooks:

```python
display_section("Analysis Title", "Optional description")
# ... your analysis code ...
```

### File Output
All analyses save to the output directory:
- PNG files for visualizations
- JSON files for metrics and diagnostics
- CSV files for detailed data (coverage, negative samples)

## Advantages of Notebook Usage

1. **Interactive Exploration**: See results immediately as cells execute
2. **Documentation**: Markdown sections organize your analysis narrative
3. **Reproducibility**: Code and outputs are saved together
4. **File Backup**: All plots still saved to disk automatically
5. **Cross-Platform**: Works on Windows, macOS, Linux

## Example: Custom Analysis

```python
# Load and filter data for specific analysis
gen_subset = generation_df[
    generation_df['unique_id'].isin(['CALI_WND', 'CALI_SUN'])
]

# Run analysis on subset
display_section("California Region Analysis")
cali_results = analyze_seasonality(gen_subset, output_dir / "cali_seasonality")

# Inspect results directly
print(f"Series analyzed: {len(cali_results['series_analyzed'])}")
print(f"Seasonality strength: {cali_results['hourly_seasonality_strength']}")
```

## Command-Line Alternative

For non-interactive use, run from command line:

```bash
python -m src.renewable.eda
```

This generates the same output but without notebook display.

## Output Files

After running analyses, check:

```
reports/renewable/eda/YYYYMMDD_HHMMSS/
├── metadata.json
├── eda_summary.json
├── seasonality/
│   ├── acf_decomposition.png
│   ├── hourly_profiles.png
│   └── analysis.json
├── zero_inflation/
│   ├── zero_inflation_by_hour.png
│   ├── generation_distributions.png
│   └── analysis.json
├── coverage/
│   ├── coverage_by_series.png
│   ├── largest_missing_blocks.png
│   ├── coverage_table.csv
│   └── analysis.json
├── negative_values/
│   ├── negative_values_analysis.png
│   ├── negative_samples.csv
│   └── analysis.json
└── weather_alignment/
    ├── weather_correlation.png
    ├── scatter_wind_speed.png
    ├── scatter_solar_radiation.png
    └── analysis.json
```

## Troubleshooting

### Plots not displaying in Jupyter?
- Make sure you're running in a Jupyter notebook (not JupyterLab without proper matplotlib backend)
- Try: `%matplotlib inline` magic command at top of notebook

### Import errors?
```python
# Make sure you're in the project directory
import sys
sys.path.insert(0, '/path/to/atsaf')

from src.renewable.eda import analyze_seasonality
```

### Large plots cutting off?
```python
# Increase figure size in notebook
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (14, 8)

# Then run analysis
analyze_seasonality(...)
```

## Recommended Notebook Structure

```python
# Cell 1: Imports and setup
import pandas as pd
from pathlib import Path
from src.renewable.eda import *

# Cell 2: Load data
generation_df = pd.read_parquet("data/renewable/generation.parquet")
weather_df = pd.read_parquet("data/renewable/weather.parquet")
output_dir = Path("reports/renewable/eda/my_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

# Cell 3: Seasonality analysis
display_section("Seasonality Analysis")
seasonality_results = analyze_seasonality(generation_df, output_dir / "seasonality")

# Cell 4: Zero-inflation analysis
display_section("Zero-Inflation Analysis")
zero_results = analyze_zero_inflation(generation_df, output_dir / "zero_inflation")

# ... and so on
```

This structure keeps analysis organized and makes it easy to re-run specific analyses.
