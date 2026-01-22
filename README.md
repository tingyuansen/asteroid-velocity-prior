# Asteroid Velocity Prior for Smart Stacking

## Motivation

When detecting faint asteroids in survey data (e.g., Rubin/LSST), matched filtering 
requires searching over possible motion vectors. The naive approach searches all 
directions uniformly, which is computationally wasteful.

**Key insight**: Known asteroid populations occupy constrained regions of orbital 
element space. At any sky position (RA, Dec) and time, objects from a given population 
will have a **predictable distribution of apparent sky-plane velocities**.

This package computes `p(μ_α, μ_δ | RA, Dec, t, population)` - the prior distribution 
of sky-plane velocities given sky position, time, and orbital population.

## Key Result

**Using population priors can reduce the matched filter search space by ~100x** while 
still capturing 90% of detectable asteroids. The reduction factor varies across the sky:
- **Opposition (180° elongation)**: Maximum reduction (~100-150x)
- **Quadrature (90° elongation)**: Moderate reduction (~30-60x)
- **Near the ecliptic**: Higher reduction due to constrained inclinations

## Mathematical Framework

### The Forward Problem

```
Orbital Elements (a, e, i, Ω, ω, M) 
    → Heliocentric State (r_helio, v_helio)
    → Geocentric State (r_geo, v_geo)  
    → Sky-Plane Velocity (μ_α, μ_δ)
```

### The Constrained Sampling Approach

Given (RA, Dec, t) and a population prior p(orbital elements):
1. Sample orbital elements from population prior
2. For each sample, uniformly sample mean anomaly M
3. Compute heliocentric → geocentric → sky position
4. **Accept** samples where sky position matches target (within tolerance)
5. Compute sky-plane velocity for accepted samples
6. Build the distribution p(μ | RA, Dec, t)

### Key Geometric Constraint

At a given (RA, Dec), the heliocentric distance r is constrained by:
- The line-of-sight direction (from Earth through the sky position)
- The orbital geometry (object must lie on its orbit)

This makes the problem tractable despite having 6 orbital elements and only 2 sky coordinates.

## Package Structure

```
photometry_stacking/
├── tutorial.ipynb              # Detailed tutorial with visualizations
├── README.md
├── requirements.txt
└── asteroid_velocity_prior/
    ├── orbital_mechanics.py    # Kepler solver, orbital elements ↔ state vectors
    ├── coordinates.py          # Heliocentric ↔ geocentric ↔ sky-plane transforms
    ├── populations.py          # Asteroid population priors (MBA, NEO, TNO, Trojans, Hildas)
    └── velocity_prior.py       # Main computation engine
```

## Installation

```bash
pip install numpy scipy matplotlib astropy
```

## Usage

```python
from asteroid_velocity_prior import (
    VelocityPriorCalculator, 
    MainBeltPopulation,
    compute_search_space_reduction,
)
from astropy.time import Time

# Create calculator for Main Belt asteroids
calc = VelocityPriorCalculator(population=MainBeltPopulation())

# Compute velocity prior at a sky position
ra, dec = 180.0, 0.0  # degrees (opposition)
obs_time = Time('2026-03-21')

result = calc.compute_velocity_prior(ra, dec, obs_time, n_samples=10000)

# result.mu_ra_cosdec, result.mu_dec are velocities in arcsec/hour
print(f"Median speed: {result.mu_total.median():.1f} arcsec/hour")

# Compute search space reduction
stats = compute_search_space_reduction(result)
print(f"Reduction factor: {stats['reduction_factor']:.1f}x")
```

## Available Populations

| Population | Semi-major axis | Typical speed at opposition |
|------------|-----------------|----------------------------|
| Main Belt  | 2.1 - 3.3 AU    | ~35 arcsec/hour            |
| NEOs       | 0.5 - 4 AU      | ~50 arcsec/hour (variable) |
| Trojans    | ~5.2 AU         | ~20 arcsec/hour            |
| TNOs       | 30 - 50 AU      | ~3 arcsec/hour             |
| Hildas     | ~4.0 AU         | ~25 arcsec/hour            |

## Applications

1. **Weighted matched filtering**: Prioritize search directions by prior probability
2. **Population classification**: Use velocity as discriminant for orbit family
3. **Survey optimization**: Predict where/when different populations are detectable
4. **Hierarchical inference**: Jointly constrain population and individual orbits

## Next Steps / Future Work

### Short-term Improvements
1. **Integrate with matched filtering pipelines**: Apply velocity weights to actual 
   stacking algorithms (e.g., KBMOD, HelioLinC)
2. **Refine population models**: Use observed orbital distributions from Minor Planet 
   Center (MPC) rather than parametric approximations
3. **Add uncertainty quantification**: Propagate orbital element uncertainties through 
   to velocity predictions

### Medium-term Extensions
4. **Action-angle formulation**: Work directly in action space for more physically 
   motivated priors and better handling of resonant populations
5. **Multi-epoch constraints**: Use velocity consistency across multiple observations 
   to further constrain the prior
6. **Time evolution**: Pre-compute velocity prior grids for entire survey seasons

### Long-term Goals
7. **Validation on real data**: Test predictions against known asteroid detections 
   in DECam, ZTF, or Rubin commissioning data
8. **Integration with Rubin/LSST**: Interface with LSST Science Pipelines for 
   production deployment
9. **Hierarchical population inference**: Jointly infer population parameters and 
   individual orbits from detections

## Tutorial

See `tutorial.ipynb` for a comprehensive tutorial covering:
- Orbital mechanics fundamentals
- Population prior definitions
- Velocity prior computation
- Search space reduction quantification
- All-sky maps of the reduction factor

