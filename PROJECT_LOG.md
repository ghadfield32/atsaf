# ATSAF Project Log

## 2026-01-22: JSON Corruption Bug Fix & Pre-commit Hook Resolution

### Issues Addressed
1. **Pre-commit Hook**: Trailing whitespace auto-fix exit code 1 in CI (expected behavior, not a bug)
2. **Critical Bug**: JSONDecodeError when loading run_log.json - malformed JSON with Git merge conflicts
3. **Data Quality Bug**: NaN values written to JSON as invalid `NaN` literal instead of `null`

### Root Cause Analysis

#### Issue #1: Pre-commit Hook "Failure"
- **File**: pyproject.toml:65, modeling.py:512-513
- **Symptom**: CI shows "Error: Process completed with exit code 1" after pre-commit hook
- **Root Cause**: This is NOT a bug - exit code 1 signals files were modified by hooks
- **Expected Behavior**: Developer reviews auto-fixed files and commits them
- **Action**: No code changes needed - working as designed

#### Issue #2: Git Merge Conflicts in JSON File (CRITICAL)
- **File**: data/renewable/run_log.json
- **Symptom**: `JSONDecodeError: Expecting property name enclosed in double quotes: line 2 column 1 (char 2)`
- **Root Cause**: Commit ff9d3bd merged two branches but left unresolved conflict markers in JSON file
- **Impact**: Dashboard crashes on load at dashboard.py:1184 when calling `json.load()`
- **Conflicts Found**: 4 conflict sections (lines 2-6, 28-37, 60-74, 80-108)
  - `run_at_utc` timestamp conflict (HEAD: 21:10:55 vs MERGE: 21:53:07)
  - `weather_rows` and `eda` section conflict
  - `best_rmse` and `baseline` metrics conflict
  - `leaderboard` model results conflict

#### Issue #3: Invalid NaN in JSON (DATA QUALITY)
- **File**: data/renewable/run_log.json:139
- **Symptom**: `"coverage_95": NaN` in JSON file
- **Root Cause**: modeling.py:141-149 `compute_coverage()` returns `np.nan` when no valid data
  - "index" baseline model has no prediction intervals → returns `np.nan`
  - Python's `json.dumps()` has `allow_nan=True` by default → writes `NaN` literal
  - `NaN` is valid JavaScript but INVALID JSON per RFC 8259
- **Impact**: JSON file is technically malformed and fails strict parsers
- **Code Path**:
  ```
  compute_coverage() → np.nan → leaderboard dict →
  json.dumps(allow_nan=True) → "coverage_95": NaN →
  json.load() → JSONDecodeError
  ```

### Changes Made

#### Fix #1: Resolve JSON Merge Conflicts
- **Script**: scratchpad/fix_json.py (temporary debugging tool)
- **Action**: Resolved 4 merge conflicts by keeping newer version (2026-01-22T21:53:07 run)
- **Action**: Replaced 2 instances of `NaN` with `null`
- **Verification**: JSON now loads correctly with 6 root keys, 6 models in leaderboard

#### Fix #2: Prevent Future NaN Corruption
- **run_hourly.py:50-78** - Added `_sanitize_for_json()` function
  - Recursively walks data structures (dict, list, primitives)
  - Replaces `NaN`/`Infinity`/`-Infinity` with `null` before JSON serialization
  - Handles both Python floats and numpy scalars
  - **Rationale**: `json.dumps()` checks NaN before calling `default` function, so must pre-process
- **run_hourly.py:81-97** - Updated `_json_default()` function
  - Simplified to handle only type conversions (Timestamp → ISO, numpy → Python)
  - Removed NaN handling (now in `_sanitize_for_json()`)
  - Added comprehensive docstrings explaining RFC 8259 compliance
- **run_hourly.py:344-350** - Updated JSON write call
  - Added `sanitized_log = _sanitize_for_json(run_log)` before `json.dumps()`
  - Added `allow_nan=False` to catch any NaN that slips through
  - Added comment explaining NaN is not valid JSON per RFC 8259

#### Testing
- **scratchpad/test_json_fix.py** - Created comprehensive test suite
  - Test numpy NaN → null conversion
  - Test Python float NaN → null conversion
  - Test Infinity/-Infinity → null conversion
  - Test normal floats preserved
  - Test pandas Timestamp → ISO string
  - Test nested structures with NaN
  - Test JSON validity and parseability
  - **Result**: All 8 tests pass

### Verification Steps
1. ✅ Fixed JSON file loads without errors: `json.load(open('data/renewable/run_log.json'))`
2. ✅ Dashboard starts without crashes (run_log.json parsed successfully)
3. ✅ Test suite passes for NaN sanitization (8/8 tests)
4. ✅ Updated code verified: `_sanitize_for_json()` and `_json_default()` import correctly
5. ✅ Smoke test: NaN in dict → `{"test": null, "value": 42}` (valid JSON)

### Impact
- **Before**: Dashboard crashes on startup with JSONDecodeError
- **After**: Dashboard loads successfully, displays model metrics correctly
- **Prevention**: Future pipeline runs will never write invalid NaN/Infinity to JSON
- **Data Quality**: All numeric metrics now properly serialize as `null` when undefined

### Files Modified
- ✅ data/renewable/run_log.json - Resolved conflicts, replaced NaN with null
- ✅ src/renewable/jobs/run_hourly.py - Added `_sanitize_for_json()`, updated JSON serialization

### Next Steps
- Monitor next pipeline run to confirm NaN handling works in production
- Consider adding JSON schema validation in CI to catch malformed JSON early
- Add Git hook to prevent committing files with conflict markers
- Document JSON serialization standards in developer guide

---

## 2026-01-22: FutureWarning Fixes & Configuration Cleanup

### Issues Addressed
1. **FutureWarning**: Pandas `groupby().apply()` will stop passing grouping columns in future versions
2. **Config Bug**: `fetch_all_regions()` default included regions with `eia_respondent=None` causing guaranteed failures
3. **Build Config**: `pyproject.toml` had incorrect package path and mixed Black/Hatch settings

### Changes Made

#### Fix #1: groupby().apply() FutureWarning (4 instances)
- **eda.py:648-656** - Solar zero ratio: replaced `apply(lambda)` with vectorized `.assign().groupby()['col'].mean()`
- **eda.py:661-669** - Wind zero ratio: same vectorized approach
- **dashboard.py:1870-1877** - Zero by hour chart: same vectorized approach
- **modeling.py:457-459** - CV window RMSE: added `include_groups=False` (can't vectorize due to custom function)
- **Rationale**: Vectorized approach is faster, cleaner, and avoids FutureWarning entirely

#### Fix #2: fetch_all_regions() Default Regions
- **eia_renewable.py:321-326** - Changed default regions filter from `r != "US48"` to also exclude regions with `eia_respondent=None`
- **Before**: Included NW, SW, SE, FLA, CAR, TEN → all fail with ValueError
- **After**: Only CALI, ERCO, MISO, PJM, SWPP (5 regions with valid EIA respondents)

#### Fix #3: pyproject.toml Configuration
- **pyproject.toml:93-94** - Fixed `packages = ["eia_data.py"]` → `packages = ["src"]` for src/ layout
- **pyproject.toml:96-110** - Moved Black settings from `[tool.hatch.build]` to `[tool.black]` section
- **pyproject.toml:97** - Updated `target-version` from `['py38', ...]` to `['py311', 'py312', 'py313']` (matches `requires-python`)

### Verification
- All 4 modified Python files pass syntax check
- pyproject.toml validates as proper TOML
- Valid EIA regions reduced from 11 to 5 (correct behavior)

---

## 2026-01-22: EDA-Driven Architecture & Bug Fixes

### Issues Addressed
1. **Critical Bug**: Negative forecasts in renewable generation (CALI_WND: -574.35 MWh minimum value)
2. **Architecture**: Mixed investigation/building concerns in dataset_builder.py
3. **Config Warning**: False "insufficient data" warning (validation timing issue)
4. **Library Warnings**: Statsforecast deprecated escape sequences

### Changes Made

#### Bug Fixes (Phases 1-3)
- **tasks.py:36** - Added `enforce_physical_constraints` import from modeling module
- **tasks.py:711-717** - Applied physical constraints to final forecasts (clip to [0, ∞))
  - Matches constraint enforcement pattern used in cross-validation (modeling.py:301)
- **run_hourly.py:202-219** - Fixed cv_windows/cv_step_size timing (pass to constructor)
  - Prevents false "insufficient data" warning during config validation
- **run_hourly.py:17-20** - Suppressed statsforecast DeprecationWarnings
  - Third-party library issues we cannot fix directly

#### Bug Fixes (Post-Implementation)

**Bug #1: Missing NegativeValueReport Parameters**
- **dataset_builder.py:164-175** - Fixed `NegativeValueReport` constructor call in `_apply_negative_policy()`
  - **Issue**: Refactoring bug - forgot to pass `total_rows` and `negative_ratio` parameters
  - **Root Cause**: Calculated `total_rows` at line 138 but didn't pass to constructor at line 164
  - **Fix**: Added `negative_ratio` calculation and passed both missing parameters
  - **Impact**: Function now works correctly, no TypeError when creating reports

**Bug #2: Missing results["preprocessing"] Initialization**
- **tasks.py:899-904** - Fixed KeyError in `run_full_pipeline()` when accessing preprocessing results
  - **Issue**: Refactoring bug - removed old code that set `results["preprocessing"]` but forgot to add new initialization
  - **Root Cause**: At line 911, calling `results["preprocessing"].update()` before creating the key
  - **Fix**: Added initialization of `results["preprocessing"]` after combining fuel datasets (line 899-904)
  - **Added Debug**: Log messages at lines 897, 907, 913, 926 to trace execution flow
  - **Impact**: Pipeline now completes successfully, dashboard shows forecasts correctly

**Bug #3: Dashboard Pipeline Skip Issue**
- **dashboard.py:901-922** - Fixed pipeline skipping when "Run Pipeline" button clicked
  - **Issue**: Pipeline was checking data freshness and skipping execution even when user explicitly requested it
  - **Root Cause**: Dashboard wasn't setting `FORCE_RUN=true`, so freshness check caused skip
  - **Behavior**: User clicks "Run Pipeline" → freshness check runs → "SKIPPED: No new data found" → nothing happens
  - **Fix**: Set `FORCE_RUN=true` environment variable before calling pipeline, restore after
  - **run_hourly.py:153-156** - Added debug logging to show FORCE_RUN status
  - **run_hourly.py:207-208** - Improved message when force run is active
  - **Impact**: Dashboard "Run Pipeline" button now always runs pipeline, ignoring freshness check

**Bug #4: Pandas FutureWarning in EDA**
- **eda.py:651-653, 664-667** - Fixed FutureWarning about groupby().apply() behavior change
  - **Issue**: `groupby('hour').apply()` without explicit `include_groups` parameter
  - **Root Cause**: Pandas 2.2+ deprecates including grouping columns in apply operations by default
  - **Warning**: "DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated..."
  - **Fix**: Added `include_groups=False` to both apply() calls (lines 651 and 664)
  - **Impact**: Code now compatible with future pandas versions, warnings eliminated

#### Feature Enhancements

**Feature #1: Data & Insights Dashboard Tab (Enhanced)**
- **dashboard.py:91-96** - Added new "📚 Data & Insights" tab to main navigation
- **dashboard.py:116-118** - Added tab6 rendering call to render_insights_tab()
- **dashboard.py:905-1950** - Created comprehensive render_insights_tab() function with 1,045 lines of interactive content:

  **Original Sections**:
  - **Pipeline Architecture**: Mermaid diagram from README showing data flow
  - **Regional Markets**: Detailed explanations of ERCOT, MISO, CAISO with comparison table
  - **Model Details**: StatsForecast rationale, weather features, data sources

  **New Enhanced Sections (Interactive with Plotly)**:
  - **Model Performance Dashboard** (lines 1170-1285):
    - Live metrics from run_log.json (best model, RMSE, series count)
    - Interactive model leaderboard with formatting
    - Plotly bar charts comparing RMSE/MAE across models
    - Prediction interval coverage analysis with target lines

  - **Forecast Accuracy by Region** (lines 1287-1437):
    - Merges forecasts.parquet with generation.parquet for actual vs predicted
    - Calculates MAE, RMSE, Bias per region
    - Interactive regional comparison bar charts
    - Box plots showing error distribution by region
    - Heatmap of MAE by region and fuel type

  - **Data Quality & EDA History** (lines 1439-1620):
    - Lists all historical EDA runs with timestamps
    - Tracks negative value counts over time (line chart)
    - Allows selection of specific EDA runs to view details
    - Provides links to download visualization files

  - **Enhanced EDA Visualizations** (lines 1698-1946):
    - **Negative Values**: Interactive scatter plot of negatives over time (Plotly)
    - **Seasonality**: Hourly profiles by series + day-of-week patterns (Plotly)
    - **Zero Inflation**: Zero ratio by hour with series comparison (Plotly)
    - **Generation Profiles**: Full time series with range slider and date selectors (Plotly)
    - All include static EDA reports as expandable fallback
    - Summary statistics tables for each analysis

- **Purpose**: Complete self-documenting dashboard with interactive exploration capabilities
- **Impact**: Users can now explore data interactively, compare models, track quality over time, and understand regional differences without reading code

**Feature #2: Pipeline Dataset Persistence & Smoke Test Update**
- **tasks.py:900-903** - Added save step for combined modeling dataset in run_full_pipeline()
  - **Issue**: Smoke test in modeling.py expected `modeling_dataset.parquet` but pipeline didn't save it
  - **Root Cause**: After fuel-specific refactoring, combined dataset was only kept in memory
  - **Fix**: Save `modeling_df` to `data/renewable/modeling_dataset.parquet` after combining fuel datasets
  - **Impact**: Smoke test now works, dataset can be inspected/analyzed externally, enables model debugging
- **modeling.py:556-561** - Added save step in smoke test after building dataset
  - **Issue**: Smoke test built dataset but didn't save before trying to reload it
  - **Root Cause**: Missing `modeling_df.to_parquet()` call between build and load steps
  - **Fix**: Added save with size logging (TEST 3) before smoke test load
  - **Flow**: Build dataset → Save to disk → Reload for CV → Validate constraints
- **modeling.py:471-599** - Enhanced smoke test with comprehensive output and multi-step validation
  - **Improvements**: EIA fetcher test, weather API test, dataset builder test, CV model test, constraint validation
  - **Output**: Clear step-by-step progress with emojis, detailed metrics, pass/fail indicators
  - **Impact**: Complete end-to-end smoke test validates entire pipeline from API fetch through forecasting

**Feature #3: Data Cleanup Utility**
- **scripts/cleanup_old_data.py** - Created automated cleanup script for old/test data
  - **Purpose**: Remove outdated files, keep only current pipeline outputs and latest 3 EDA runs
  - **Targets**: Old corrected files (generation_corrected.parquet), demo preprocessing runs (demo_clamp, demo_hybrid, latest), old EDA runs (keeps 3 newest), temporary JSON files
  - **Safety**: Dry-run mode by default, requires --live flag to actually delete
  - **Usage**: `python scripts/cleanup_old_data.py` (dry-run) or `python scripts/cleanup_old_data.py --live`
  - **Bug Fix**: Fixed counter logic to properly count items in dry-run mode (lines 48-51, 65-68, 76-79, 94-97)
  - **First Run**: Cleaned up 5 old items, freed 38.6 KB
  - **Impact**: Keeps data directory clean and organized, prevents confusion from old files

**Current Data Files (After Cleanup)**:
- ✅ `generation.parquet` - Raw generation data from EIA
- ✅ `weather.parquet` - Weather data from Open-Meteo
- ✅ `forecasts.parquet` - Generated forecasts with intervals
- ✅ `run_log.json` - Pipeline execution log with metrics
- ✅ `skip_log.json` - Freshness check skip tracking
- ✅ `modeling_dataset.parquet` - Combined preprocessed dataset (NEW, created by pipeline)
- ✅ `eda/*/` - Latest 3 EDA runs with recommendations
- ✅ `preprocessing/sun/` - Solar preprocessing reports
- ✅ `preprocessing/wnd/` - Wind preprocessing reports

#### Architectural Refactoring (Phase 4)
- **eda.py:18-80** - Added `PreprocessingRecommendation` and `EDARecommendations` dataclasses
  - Structured format for EDA recommendations
  - Includes save/load methods for persistence
- **eda.py:391** - Modified `run_full_eda()` to return `EDARecommendations` object
  - Changed from returning Dict[str, Any] to structured dataclass
  - Generates structured preprocessing recommendations based on data analysis
- **dataset_builder.py:110-202** - Simplified `_handle_negative_values()` → `_apply_negative_policy()`
  - Removed investigation logic (moved to EDA)
  - Now only applies policies, no root cause analysis
- **dataset_builder.py:331** - Added `eda_recommendations` parameter to `build_modeling_dataset()`
  - Accepts recommendations from EDA
  - Uses EDA-recommended policies instead of hardcoded defaults
- **dataset_builder.py:472-573** - Added fuel-specific builders
  - `RenewableDatasetBuilder` (base class)
  - `SolarDatasetBuilder` (solar-specific handling)
  - `WindDatasetBuilder` (wind-specific handling)
  - `build_dataset_by_fuel_type()` factory function
- **tasks.py:796-802** - Added `skip_eda` parameter to `run_full_pipeline()`
  - Allows disabling EDA for fast iteration
- **tasks.py:845-900** - Integrated EDA step before dataset building
  - Runs EDA to generate recommendations
  - Builds separate datasets per fuel type using EDA recommendations
  - Combines datasets for training
- **run_hourly.py:221-229** - Added `SKIP_EDA` environment variable support
  - Passes skip_eda flag to pipeline
  - Allows backward compatibility during rollout

### Key Architectural Changes

**Before**:
```
Raw Data → Build Dataset (hardcoded policy) → Train → Forecast (missing constraints)
```

**After**:
```
Raw Data → EDA (investigate & recommend) → Build Dataset per Fuel Type (apply recommendations) → Train → Forecast (with constraints)
```

### Verification Steps
1. Run pipeline: `python -m src.renewable.jobs.run_hourly`
2. Check `data/renewable/run_log.json` - `negative_forecasts.negative_rows` should be 0
3. Check console - no "insufficient data" or statsforecast warnings
4. Check `data/renewable/eda/{timestamp}/recommendations.json` - structured output exists

### Expected Improvements
- **Zero negative forecasts** (previously had 7-8 negative values per run)
- **Clean console output** (no false warnings)
- **Data-driven preprocessing** (EDA recommendations used instead of hardcoded policies)
- **Fuel-specific handling** (solar and wind datasets built separately with appropriate configs)

### Testing Notes
- All changes are backward compatible
- Can disable EDA with `SKIP_EDA=true` environment variable
- Fuel-specific builders fall back to sensible defaults if EDA not run

### Next Steps
- Monitor forecasts for negative values in production runs
- Add integration tests for EDA → dataset builder flow
- Consider caching EDA results if data unchanged (checksums)
- Evaluate performance impact of EDA step (~10-30s overhead)

---

## Log Format
- **Date**: YYYY-MM-DD format
- **Topic**: Brief description of changes
- **Issues Addressed**: List of problems solved
- **Changes Made**: File-by-file summary with line numbers
- **Verification**: How to test the changes
- **Next Steps**: Follow-up work needed
