# Python Implementation of Lifespan Brain Development Analysis

This repository contains a Python implementation of the R-based lifespan brain development analysis, specifically porting the functionality from `example.r`.

## Overview

The Python script `lifespan_analysis.py` replicates the core functionality of the R analysis pipeline, including:

- **GAMLSS Model Implementation**: Simplified GAMLSS (Generalized Additive Models for Location, Scale and Shape) modeling
- **Population Curve Generation**: Creates age-sex grids for population-level predictions
- **Parameter Application**: Applies fitted model parameters to generate predictions (equivalent to R's `Apply.Param` function)
- **Visualization**: Generates plots corresponding to the R plotting commands
- **Statistical Analysis**: Provides summary statistics and developmental trajectory analysis

## Key Features

### Core Functions

1. **`GAMLSSModel`**: Simplified GAMLSS model class for generating quantile predictions
2. **`create_population_curve_grid()`**: Creates age-sex prediction grids matching R's `POP.CURVE.LIST`
3. **`apply_fitted_parameters()`**: Core function equivalent to R's `Apply.Param()`
4. **Visualization functions**: Generate plots matching the R `plot()` commands

### Analysis Outputs

- Population trajectory curves for GMV (Gray Matter Volume) across the lifespan
- Sex-specific developmental patterns
- Confidence intervals and uncertainty quantification
- Age-specific comparisons at key developmental stages
- Summary statistics and peak analysis
- Exported data for further analysis

## Installation

### Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy>=1.20.0
- pandas>=1.3.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- scipy>=1.7.0

## Usage

### Basic Usage

Run the complete analysis:

```bash
python3 lifespan_analysis.py
```

This will:
1. Generate population curves for ages 90 days to 95 years
2. Create basic and advanced visualizations
3. Output summary statistics
4. Save results to CSV files in the `output/` directory
5. Save plots as PNG files

### Output Files

The script generates several output files:

- `basic_gmv_trajectories.png`: Basic trajectory plots matching R visualization
- `advanced_gmv_trajectories.png`: Advanced plots with confidence intervals
- `output/gmv_population_curves.csv`: Full prediction results
- `output/gmv_summary_stats.csv`: Summary statistics by sex

## Key Differences from R Implementation

### Similarities
- Uses the same age transformation (log transform)
- Generates identical prediction grids (32 points from 90 days to 95 years)
- Produces similar quantile predictions (2.5%, 25%, 50%, 75%, 97.5%)
- Matches the R plotting structure and scaling

### Differences
- Uses simulated parameters instead of loading from RDS files
- Implements simplified GAMLSS functionality
- Enhanced visualization with confidence intervals
- Additional summary statistics and export options

## Extending the Implementation

### Using Real Data

To use with real fitted parameters:

1. Replace `create_mock_fit_params()` with a function that loads your actual fitted parameters
2. Modify the parameter application in `apply_fitted_parameters()` to match your model structure
3. Adjust the GAMLSS family if using distributions other than Normal

### Adding New Models

The framework can be extended to other brain measures by:

1. Creating new parameter sets in `create_mock_fit_params()`
2. Adjusting trajectory parameters in `apply_fitted_parameters()`
3. Modifying visualization functions for different measures

## References

This implementation is based on the lifespan brain development analysis methodology described in:

**Bethlehem, Seidlitz, White et al. (2022). Nature.** https://www.nature.com/articles/s41586-022-04554-y

## Original R Implementation

The original R code includes:
- `example.r`: Main analysis script
- `101.common-functions.r`: Contains the `Apply.Param` function
- `300.variables.r`: Variable definitions and parameters

This Python implementation maintains compatibility with the R analysis pipeline while providing enhanced functionality and visualization options.