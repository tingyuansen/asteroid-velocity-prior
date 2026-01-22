"""
Velocity Prior Computation

Main engine for computing p(μ_α, μ_δ | RA, Dec, t, population)

The approach:
1. Sample orbital elements from the population prior
2. For each sample, check if the orbit can place the object near the target (RA, Dec)
3. If compatible, compute the sky-plane velocity at that configuration
4. Return the distribution of velocities from compatible orbits

Key insight: For a given (RA, Dec) and orbital elements (a, e, i, Ω, ω),
the mean anomaly M that places the object at that sky position is constrained.
This constraint comes from the line-of-sight intersection with the orbit.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from .orbital_mechanics import orbital_elements_to_state
from .coordinates import (
    get_earth_position,
    heliocentric_to_geocentric,
    geocentric_to_radec,
    geocentric_to_sky_velocity,
)
from .populations import OrbitalPopulation, OrbitalElements


@dataclass
class VelocityPriorResult:
    """Container for velocity prior computation results."""
    
    # Sky-plane velocities of compatible samples
    mu_ra_cosdec: np.ndarray  # arcsec/hour
    mu_dec: np.ndarray        # arcsec/hour
    
    # Additional information about the samples
    helio_distance: np.ndarray  # Heliocentric distance [AU]
    geo_distance: np.ndarray    # Geocentric distance [AU]
    
    # The orbital elements that produced these velocities
    elements: OrbitalElements
    
    # Weights (if any, for importance sampling)
    weights: Optional[np.ndarray] = None
    
    @property
    def n_samples(self) -> int:
        return len(self.mu_ra_cosdec)
    
    @property
    def mu_total(self) -> np.ndarray:
        """Total proper motion [arcsec/hour]."""
        return np.sqrt(self.mu_ra_cosdec**2 + self.mu_dec**2)
    
    @property
    def position_angle(self) -> np.ndarray:
        """Position angle of motion [deg], East from North."""
        pa = np.degrees(np.arctan2(self.mu_ra_cosdec, self.mu_dec))
        return np.mod(pa, 360)


class VelocityPriorCalculator:
    """
    Calculator for sky-plane velocity priors given orbital populations.
    
    Usage
    -----
    >>> calc = VelocityPriorCalculator(MainBeltPopulation())
    >>> result = calc.compute_velocity_prior(ra=180, dec=0, time='2026-03-15')
    >>> print(result.mu_ra_cosdec, result.mu_dec)
    """
    
    def __init__(
        self,
        population: OrbitalPopulation,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize calculator.
        
        Parameters
        ----------
        population : OrbitalPopulation
            The orbital population to sample from
        rng : numpy.random.Generator, optional
            Random number generator for reproducibility
        """
        self.population = population
        self.rng = rng if rng is not None else np.random.default_rng()
    
    def compute_velocity_prior(
        self,
        ra: float,
        dec: float,
        time,
        n_samples: int = 10000,
        angular_tolerance: float = 5.0,
        distance_range: Optional[Tuple[float, float]] = None,
    ) -> VelocityPriorResult:
        """
        Compute the velocity prior at a given sky position and time.
        
        This uses a rejection sampling approach:
        1. Sample orbital elements from the population prior
        2. For each sample, uniformly sample mean anomaly
        3. Accept samples where the resulting sky position is within
           angular_tolerance of the target (ra, dec)
        4. Return the velocity distribution from accepted samples
        
        Parameters
        ----------
        ra : float
            Right ascension [degrees]
        dec : float
            Declination [degrees]
        time : str or astropy.time.Time
            Observation time
        n_samples : int
            Target number of accepted samples
        angular_tolerance : float
            Angular radius for position matching [degrees]
        distance_range : tuple of (min, max), optional
            Restrict geocentric distance range [AU]
            
        Returns
        -------
        VelocityPriorResult
            Container with velocity samples and metadata
        """
        from astropy.time import Time
        
        if not isinstance(time, Time):
            time = Time(time)
        
        # Get Earth position at observation time
        r_earth, v_earth = get_earth_position(time)
        
        # Precompute target trigonometry (optimization)
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        cos_dec_target = np.cos(dec_rad)
        sin_dec_target = np.sin(dec_rad)
        cos_tol = np.cos(np.radians(angular_tolerance))
        
        # Preallocate arrays for efficiency (estimate ~5% acceptance rate)
        estimated_total = n_samples * 50
        max_stored = n_samples * 2  # Don't over-allocate
        
        mu_ra_arr = np.empty(max_stored)
        mu_dec_arr = np.empty(max_stored)
        helio_dist_arr = np.empty(max_stored)
        geo_dist_arr = np.empty(max_stored)
        a_arr = np.empty(max_stored)
        e_arr = np.empty(max_stored)
        i_arr = np.empty(max_stored)
        Omega_arr = np.empty(max_stored)
        omega_arr = np.empty(max_stored)
        M_arr = np.empty(max_stored)
        
        n_accepted = 0
        total_tried = 0
        batch_size = max(50000, n_samples * 10)
        max_iterations = 200
        
        for _ in range(max_iterations):
            # Sample orbital elements
            elements = self.population.sample(batch_size, self.rng)
            
            # Sample mean anomaly uniformly
            M = self.rng.uniform(0, 2 * np.pi, batch_size)
            
            # Compute heliocentric state (fully vectorized)
            r_helio, v_helio = orbital_elements_to_state(
                elements.a, elements.e, elements.i,
                elements.Omega, elements.omega, M
            )
            
            # Transform to geocentric (vectorized)
            r_geo = r_helio - r_earth
            v_geo = v_helio - v_earth
            
            # Compute RA, Dec of each sample (vectorized)
            ra_sample, dec_sample, dist_sample = geocentric_to_radec(r_geo, degrees=True)
            
            # Angular separation using dot product (faster than arccos)
            # cos(sep) = sin(dec1)*sin(dec2) + cos(dec1)*cos(dec2)*cos(ra1-ra2)
            dec_sample_rad = np.radians(dec_sample)
            cos_sep = (
                sin_dec_target * np.sin(dec_sample_rad) +
                cos_dec_target * np.cos(dec_sample_rad) * 
                np.cos(ra_rad - np.radians(ra_sample))
            )
            
            # Apply selection criteria (vectorized boolean operations)
            mask = cos_sep > cos_tol  # equivalent to angular_sep < tolerance
            
            # Distance cuts
            if distance_range is not None:
                mask &= (dist_sample >= distance_range[0]) & (dist_sample <= distance_range[1])
            
            # Heliocentric distance sanity check
            helio_dist = np.sqrt(np.sum(r_helio**2, axis=-1))  # faster than linalg.norm
            mask &= (helio_dist > 0.1) & (helio_dist < 100)
            
            n_new = np.sum(mask)
            
            if n_new > 0:
                # Compute sky velocities only for accepted samples
                r_geo_acc = r_geo[mask]
                v_geo_acc = v_geo[mask]
                mu_ra, mu_dec, _ = geocentric_to_sky_velocity(r_geo_acc, v_geo_acc)
                
                # Store results (handle array bounds)
                end_idx = min(n_accepted + n_new, max_stored)
                n_to_store = end_idx - n_accepted
                
                if n_to_store > 0:
                    mu_ra_arr[n_accepted:end_idx] = mu_ra[:n_to_store]
                    mu_dec_arr[n_accepted:end_idx] = mu_dec[:n_to_store]
                    helio_dist_arr[n_accepted:end_idx] = helio_dist[mask][:n_to_store]
                    geo_dist_arr[n_accepted:end_idx] = dist_sample[mask][:n_to_store]
                    a_arr[n_accepted:end_idx] = elements.a[mask][:n_to_store]
                    e_arr[n_accepted:end_idx] = elements.e[mask][:n_to_store]
                    i_arr[n_accepted:end_idx] = elements.i[mask][:n_to_store]
                    Omega_arr[n_accepted:end_idx] = elements.Omega[mask][:n_to_store]
                    omega_arr[n_accepted:end_idx] = elements.omega[mask][:n_to_store]
                    M_arr[n_accepted:end_idx] = M[mask][:n_to_store]
                    
                    n_accepted = end_idx
            
            total_tried += batch_size
            
            if n_accepted >= n_samples:
                break
            
            # Adaptive batch size
            if total_tried > 0:
                acceptance_rate = n_accepted / total_tried
                if acceptance_rate > 0 and acceptance_rate < 0.001:
                    batch_size = min(batch_size * 2, 500000)
        
        # Trim to requested number
        n_final = min(n_accepted, n_samples)
        
        if n_final == 0:
            raise ValueError(
                f"No samples accepted. The target position ({ra}, {dec}) may be "
                f"incompatible with the {self.population.name} population, or "
                f"increase angular_tolerance."
            )
        
        return VelocityPriorResult(
            mu_ra_cosdec=mu_ra_arr[:n_final].copy(),
            mu_dec=mu_dec_arr[:n_final].copy(),
            helio_distance=helio_dist_arr[:n_final].copy(),
            geo_distance=geo_dist_arr[:n_final].copy(),
            elements=OrbitalElements(
                a=a_arr[:n_final].copy(),
                e=e_arr[:n_final].copy(),
                i=i_arr[:n_final].copy(),
                Omega=Omega_arr[:n_final].copy(),
                omega=omega_arr[:n_final].copy(),
                M=M_arr[:n_final].copy(),
            ),
        )
    
    def compute_velocity_prior_batch(
        self,
        positions: np.ndarray,
        time,
        n_samples_per_position: int = 1000,
        angular_tolerance: float = 5.0,
        n_monte_carlo: int = 500000,
    ) -> Dict[Tuple[float, float], VelocityPriorResult]:
        """
        Efficiently compute velocity priors for multiple sky positions.
        
        This is more efficient than calling compute_velocity_prior repeatedly
        because it shares the orbital element sampling across all positions.
        
        Parameters
        ----------
        positions : ndarray, shape (N, 2)
            Array of (RA, Dec) positions in degrees
        time : str or astropy.time.Time
            Observation time
        n_samples_per_position : int
            Target samples per position
        angular_tolerance : float
            Angular radius for matching [degrees]
        n_monte_carlo : int
            Total Monte Carlo samples to generate
            
        Returns
        -------
        dict
            Dictionary mapping (ra, dec) tuples to VelocityPriorResult
        """
        from astropy.time import Time
        
        if not isinstance(time, Time):
            time = Time(time)
        
        r_earth, v_earth = get_earth_position(time)
        
        # Precompute position trigonometry
        positions = np.asarray(positions)
        n_positions = len(positions)
        ra_rad = np.radians(positions[:, 0])
        dec_rad = np.radians(positions[:, 1])
        cos_dec = np.cos(dec_rad)
        sin_dec = np.sin(dec_rad)
        cos_tol = np.cos(np.radians(angular_tolerance))
        
        # Initialize result containers
        results = {tuple(pos): {'mu_ra': [], 'mu_dec': [], 'helio': [], 'geo': [],
                                'a': [], 'e': [], 'i': [], 'Omega': [], 'omega': [], 'M': []}
                   for pos in positions}
        counts = np.zeros(n_positions, dtype=int)
        
        # Process in batches
        batch_size = 100000
        n_batches = (n_monte_carlo + batch_size - 1) // batch_size
        
        for _ in range(n_batches):
            elements = self.population.sample(batch_size, self.rng)
            M = self.rng.uniform(0, 2 * np.pi, batch_size)
            
            r_helio, v_helio = orbital_elements_to_state(
                elements.a, elements.e, elements.i,
                elements.Omega, elements.omega, M
            )
            
            r_geo = r_helio - r_earth
            v_geo = v_helio - v_earth
            
            ra_sample, dec_sample, dist_sample = geocentric_to_radec(r_geo, degrees=True)
            dec_sample_rad = np.radians(dec_sample)
            ra_sample_rad = np.radians(ra_sample)
            
            helio_dist = np.sqrt(np.sum(r_helio**2, axis=-1))
            valid = (helio_dist > 0.1) & (helio_dist < 100)
            
            # Check each target position
            for idx, pos in enumerate(positions):
                if counts[idx] >= n_samples_per_position:
                    continue
                
                cos_sep = (
                    sin_dec[idx] * np.sin(dec_sample_rad) +
                    cos_dec[idx] * np.cos(dec_sample_rad) * 
                    np.cos(ra_rad[idx] - ra_sample_rad)
                )
                
                mask = valid & (cos_sep > cos_tol)
                n_accept = np.sum(mask)
                
                if n_accept > 0:
                    need = n_samples_per_position - counts[idx]
                    take = min(n_accept, need)
                    idx_accept = np.where(mask)[0][:take]
                    
                    mu_ra, mu_dec, _ = geocentric_to_sky_velocity(
                        r_geo[idx_accept], v_geo[idx_accept]
                    )
                    
                    key = tuple(pos)
                    results[key]['mu_ra'].extend(mu_ra)
                    results[key]['mu_dec'].extend(mu_dec)
                    results[key]['helio'].extend(helio_dist[idx_accept])
                    results[key]['geo'].extend(dist_sample[idx_accept])
                    results[key]['a'].extend(elements.a[idx_accept])
                    results[key]['e'].extend(elements.e[idx_accept])
                    results[key]['i'].extend(elements.i[idx_accept])
                    results[key]['Omega'].extend(elements.Omega[idx_accept])
                    results[key]['omega'].extend(elements.omega[idx_accept])
                    results[key]['M'].extend(M[idx_accept])
                    counts[idx] += take
            
            # Check if all positions are done
            if np.all(counts >= n_samples_per_position):
                break
        
        # Convert to VelocityPriorResult objects
        final_results = {}
        for pos in positions:
            key = tuple(pos)
            data = results[key]
            if len(data['mu_ra']) > 0:
                final_results[key] = VelocityPriorResult(
                    mu_ra_cosdec=np.array(data['mu_ra']),
                    mu_dec=np.array(data['mu_dec']),
                    helio_distance=np.array(data['helio']),
                    geo_distance=np.array(data['geo']),
                    elements=OrbitalElements(
                        a=np.array(data['a']),
                        e=np.array(data['e']),
                        i=np.array(data['i']),
                        Omega=np.array(data['Omega']),
                        omega=np.array(data['omega']),
                        M=np.array(data['M']),
                    ),
                )
            else:
                final_results[key] = None
        
        return final_results


def compute_search_space_reduction(
    result: VelocityPriorResult,
    velocity_range: Tuple[float, float] = (-100, 100),
    n_bins: int = 100,
    coverage_fraction: float = 0.9,
) -> dict:
    """
    Compute the search space reduction factor from using the velocity prior.
    
    Parameters
    ----------
    result : VelocityPriorResult
        Velocity prior result
    velocity_range : tuple
        Range of velocities to consider [arcsec/hour]
    n_bins : int
        Number of bins per dimension for the search grid
    coverage_fraction : float
        Fraction of probability mass to cover (e.g., 0.9 for 90%)
        
    Returns
    -------
    dict with keys:
        'reduction_factor': How much smaller the search space is
        'uniform_cells': Number of cells in uniform search
        'prior_cells': Number of cells needed with prior
        'coverage': Actual coverage achieved
    """
    # Create uniform grid
    v_min, v_max = velocity_range
    mu_ra_bins = np.linspace(v_min, v_max, n_bins + 1)
    mu_dec_bins = np.linspace(v_min, v_max, n_bins + 1)
    
    # Compute 2D histogram (empirical prior)
    hist, _, _ = np.histogram2d(
        result.mu_ra_cosdec, result.mu_dec,
        bins=[mu_ra_bins, mu_dec_bins]
    )
    
    # Normalize to probability
    hist = hist / hist.sum()
    
    # Sort cells by probability (descending)
    flat_hist = hist.ravel()
    sorted_idx = np.argsort(flat_hist)[::-1]
    cumsum = np.cumsum(flat_hist[sorted_idx])
    
    # Find number of cells to achieve coverage
    n_cells_needed = np.searchsorted(cumsum, coverage_fraction) + 1
    total_cells = n_bins * n_bins
    
    # Cells actually containing probability mass
    n_nonzero = np.sum(hist > 0)
    
    return {
        'reduction_factor': total_cells / n_cells_needed,
        'uniform_cells': total_cells,
        'prior_cells': n_cells_needed,
        'nonzero_cells': n_nonzero,
        'coverage': cumsum[n_cells_needed - 1] if n_cells_needed <= len(cumsum) else 1.0,
        'velocity_range': velocity_range,
        'histogram': hist,
        'bin_edges': (mu_ra_bins, mu_dec_bins),
    }


def compute_sky_reduction_map(
    population: OrbitalPopulation,
    time,
    ra_grid: np.ndarray,
    dec_grid: np.ndarray,
    n_samples: int = 2000,
    angular_tolerance: float = 5.0,
    velocity_range: Tuple[float, float] = (-100, 100),
    coverage_fraction: float = 0.9,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Compute search space reduction factor across the sky.
    
    Parameters
    ----------
    population : OrbitalPopulation
        Asteroid population
    time : str or astropy.time.Time
        Observation time
    ra_grid : ndarray
        RA values for grid [degrees]
    dec_grid : ndarray
        Dec values for grid [degrees]
    n_samples : int
        Samples per sky position
    angular_tolerance : float
        Angular tolerance [degrees]
    velocity_range : tuple
        Velocity search range [arcsec/hour]
    coverage_fraction : float
        Target probability coverage
    rng : numpy.random.Generator, optional
        Random number generator
        
    Returns
    -------
    dict with keys:
        'ra_grid': RA grid
        'dec_grid': Dec grid  
        'reduction_map': 2D array of reduction factors
        'valid_mask': 2D boolean array of valid positions
    """
    calc = VelocityPriorCalculator(population, rng=rng)
    
    # Create position array
    RA, DEC = np.meshgrid(ra_grid, dec_grid)
    positions = np.column_stack([RA.ravel(), DEC.ravel()])
    
    # Compute velocity priors for all positions
    results = calc.compute_velocity_prior_batch(
        positions, time,
        n_samples_per_position=n_samples,
        angular_tolerance=angular_tolerance,
    )
    
    # Compute reduction factor for each position
    reduction_map = np.full(RA.shape, np.nan)
    valid_mask = np.zeros(RA.shape, dtype=bool)
    
    for i, dec in enumerate(dec_grid):
        for j, ra in enumerate(ra_grid):
            result = results.get((ra, dec))
            if result is not None and result.n_samples >= 100:
                stats = compute_search_space_reduction(
                    result,
                    velocity_range=velocity_range,
                    coverage_fraction=coverage_fraction,
                )
                reduction_map[i, j] = stats['reduction_factor']
                valid_mask[i, j] = True
    
    return {
        'ra_grid': ra_grid,
        'dec_grid': dec_grid,
        'reduction_map': reduction_map,
        'valid_mask': valid_mask,
        'population': population.name,
        'time': str(time),
        'coverage_fraction': coverage_fraction,
    }
