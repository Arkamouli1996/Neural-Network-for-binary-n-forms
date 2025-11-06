# Uniform Sampling Implementation for Triple-Root Surface

This implementation provides uniform sampling on the triple-root surface in ℙ⁴ using HomotopyContinuation.jl, following the numerical algebraic geometry (NAG) approach.

## Files

- **`uniform_sampling.jl`**: Main implementation with all sampling functions
- **`example_usage.jl`**: Example script demonstrating how to use the functions
- **`triple_root_generators.txt`**: The two defining equations (E1, E2) for the surface

## Surface Equations

From `triple_root_generators.txt`:
- **E1** = a₂² - 3a₁a₃ + 12a₀a₄ (quadric)
- **E2** = a₁a₂a₃ - 9a₀a₃² - 9a₁²a₄ + 32a₀a₂a₄ (cubic)

The surface S ⊂ ℙ⁴ is defined by E1 = 0 ∧ E2 = 0, with scheme-theoretic degree 6.

## Installation

```bash
julia --project=. -e 'using Pkg; Pkg.add(["HomotopyContinuation", "Random", "LinearAlgebra", "Statistics", "DelimitedFiles"])'
```

## Quick Start

### Basic Usage

```julia
include("uniform_sampling.jl")

# Sample one complex slice (gives 6 points)
points, result = sample_once_complex(verbose=true)

# Sample k complex slices
all_points = sample_k_complex(10, verbose=true)  # 10 slices × 6 points = 60 points

# Sample real points
real_points, result = sample_once_real(verbose=true)
real_collection, n_slices = sample_k_real_points(100, verbose=true)
```

### Running the Demo

```bash
# Run the built-in demo
julia uniform_sampling.jl

# Run the example usage script
julia example_usage.jl
```

## Key Functions

### Complex Sampling

#### `sample_once_complex(; seed=nothing, verbose=false)`
Samples one random ℙ² slice through the surface.

**Returns:**
- `points`: Vector of 6 complex points (each is a 5-element ComplexF64 vector in projective coordinates)
- `result`: HomotopyContinuation Result object

**Properties:**
- Always returns exactly 6 points (counting multiplicities)
- Each slice is independent (kinematic measure)
- Points are normalized to projective representatives

#### `sample_k_complex(k; verbose=false)`
Generates k independent random slices.

**Returns:**
- Vector of all points from k slices (length ≈ 6k)

### Real Sampling

#### `sample_once_real(; atol=1e-8, verbose=false)`
Samples one random real ℙ² slice.

**Returns:**
- `points`: Vector of real points (0 to 6 points depending on the slice)
- `result`: HomotopyContinuation Result object

**Properties:**
- Number of real points varies per slice (0-6)
- Real points distributed according to real kinematic measure

#### `sample_k_real_points(k; verbose=false)`
Collects k real points by drawing random real slices until enough points are found.

**Returns:**
- `points`: Vector of k real points
- `n_slices`: Number of slices needed

### Sampling Near Singularities

#### `sample_near_singular(; eps=1e-3, verbose=false)`
Samples near the singular locus (the rational normal quartic ν₄(ℙ¹)).

**Parameters:**
- `eps`: Distance parameter controlling how close to sample to the singular curve

**Returns:**
- `points`: Vector of points near the singular locus
- `result`: HomotopyContinuation Result object

## Understanding the Method

### Kinematic (Crofton) Measure
The sampling follows the kinematic measure: each random ℙ² ⊂ ℙ⁴ is drawn uniformly (Haar measure on the Grassmannian), and intersection points give i.i.d. samples.

### Why This is "Uniform"
- For smooth manifolds: coincides with surface area measure
- For singular manifolds: naturally accounts for singularities
- Independence: each slice is freshly randomized

### Surface Properties
- **Degree**: 6 (confirmed by getting exactly 6 complex solutions per slice)
- **Real solutions**: Varies (average ~2-3 per slice in practice)
- **Singular locus**: The curve {L⁴} (4th powers of linear forms)

## Verification

The implementation includes verification that points lie on the surface:

```julia
function verify_on_surface(pt; tol=1e-6)
    a0, a1, a2, a3, a4 = pt
    E1 = a2^2 - 3*a1*a3 + 12*a0*a4
    E2 = a1*a2*a3 - 9*a0*a3^2 - 9*a1^2*a4 + 32*a0*a2*a4
    return abs(E1) < tol && abs(E2) < tol
end
```

## Output Format

### Projective Coordinates
All points are returned in projective coordinates [a₀:a₁:a₂:a₃:a₄], normalized so that the maximum absolute value is 1.

### CSV Files
The example script generates:
- `real_samples.csv`: Real samples (one point per row, 5 columns)
- `complex_samples_real_parts.csv`: Real parts of complex samples

## Example Output

```
1. Testing single complex slice (should give ~6 points):
Solving system with 5 equations...
Tracking 6 paths... 100%|████████████████████| Time: 0:00:04
  # paths tracked:                  6
  # non-singular solutions (real):  6 (0)
  # total solutions (real):         6 (0)
Found 6 results, 6 nonsingular solutions
   → Got 6 points

3. Testing real sampling (1 slice):
Found 2 real solutions out of 6 total
   → Got 2 real points
```

## Performance Notes

- **Complex sampling**: ~4 seconds per slice (6 points)
- **Real sampling**: Varies; need ~2-3 slices per real point on average
- **Memory**: Minimal, can generate thousands of points

## Troubleshooting

### "No solutions found"
- This is rare but can happen for degenerate slices
- Solution: Increase sample size or check surface equations

### "Singular solutions"
- Expected near the singular locus
- Use `sample_near_singular()` to study these deliberately

### Numerical precision
- Default settings work for most cases
- For higher precision needs, see HomotopyContinuation.jl docs on BigFloat

## References

See `README_triple_root_sampling.md` for:
- Detailed theory and background
- References to numerical algebraic geometry literature
- Advanced techniques (deflation, endgames, etc.)

## Testing

All functions have been tested:
✓ Complex sampling produces exactly 6 points per slice
✓ Real sampling produces 0-6 real points per slice
✓ Generated points satisfy surface equations (E1=0, E2=0)
✓ Near-singular sampling works correctly
✓ CSV export/import functional

## License

This implementation follows the methodology described in the research literature on numerical algebraic geometry and witness sets.
