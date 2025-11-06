# Data Generation Script

Generate custom-sized datasets for training the triple-root classifier.

## Usage

```bash
julia julia/generate_data.jl [n_positive] [n_negative]
```

### Parameters

- `n_positive`: Number of positive samples (points on triple-root surface)  
  Default: 2000
- `n_negative`: Number of negative samples (random points in ℙ⁴)  
  Default: 2000

### Examples

```bash
# Generate default dataset (2000 + 2000)
julia julia/generate_data.jl

# Generate smaller dataset for testing
julia julia/generate_data.jl 500 500

# Generate unbalanced dataset
julia julia/generate_data.jl 1000 3000

# Generate large dataset
julia julia/generate_data.jl 5000 5000
```

## Output

The script generates four CSV files in the `data/` directory:

- `X_train.csv`: Combined training features (n_positive + n_negative) × 10
- `y_train.csv`: Labels (1 for on-surface, 0 for random)
- `positive_samples.csv`: Positive samples only
- `negative_samples.csv`: Negative samples only

## Performance

- **Positive samples**: ~4 seconds per slice (6 points per slice)
  - 500 samples: ~6 minutes
  - 2000 samples: ~22 minutes
  - 5000 samples: ~55 minutes

- **Negative samples**: Instant (random generation)

- **Total time**: Dominated by positive sample generation

## Notes

- Positive samples are generated uniformly on the surface using numerical algebraic geometry
- Each random slice gives exactly 6 complex solutions (surface degree = 6)
- Negative samples are uniformly random in ℙ⁴ projective space
- All samples are verified to satisfy surface equations (positive) or be random (negative)
- Dataset is automatically shuffled before saving

## After Generation

Use the generated data with the Python scripts:

```bash
# Visualize
python3.11 python/visualize.py

# Train model
python3.11 python/train.py

# Test model
python3.11 python/test.py
```
