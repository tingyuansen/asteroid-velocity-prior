"""
Asteroid Population Priors

Defines probability distributions over orbital elements for various
asteroid populations. These serve as priors for the velocity computation.

Each population is defined by distributions in orbital element space:
- a: semi-major axis [AU]
- e: eccentricity
- i: inclination [rad]
- Ω: longitude of ascending node [rad] (typically uniform)
- ω: argument of perihelion [rad] (typically uniform)
- M: mean anomaly [rad] (uniform for random phase)

Population parameters are approximate representations of the known
orbital element distributions for each asteroid family.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class OrbitalElements:
    """Container for orbital elements of a sample."""
    a: np.ndarray      # Semi-major axis [AU]
    e: np.ndarray      # Eccentricity
    i: np.ndarray      # Inclination [rad]
    Omega: np.ndarray  # Longitude of ascending node [rad]
    omega: np.ndarray  # Argument of perihelion [rad]
    M: np.ndarray      # Mean anomaly [rad]
    
    def __len__(self):
        return len(self.a)


class OrbitalPopulation(ABC):
    """Abstract base class for orbital element distributions."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the population."""
        pass
    
    @abstractmethod
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> OrbitalElements:
        """
        Sample orbital elements from the population prior.
        
        Parameters
        ----------
        n : int
            Number of samples
        rng : numpy.random.Generator, optional
            Random number generator
            
        Returns
        -------
        OrbitalElements
            Sampled orbital elements
        """
        pass
    
    @abstractmethod
    def log_prob(self, elements: OrbitalElements) -> np.ndarray:
        """
        Compute log probability density of orbital elements.
        
        Parameters
        ----------
        elements : OrbitalElements
            Orbital elements to evaluate
            
        Returns
        -------
        log_prob : ndarray
            Log probability density
        """
        pass
    
    def _get_rng(self, rng):
        """Get random number generator."""
        if rng is None:
            return np.random.default_rng()
        return rng


class MainBeltPopulation(OrbitalPopulation):
    """
    Main Belt Asteroid population.
    
    The Main Belt extends from ~2.1 AU (inside the 4:1 resonance with Jupiter)
    to ~3.3 AU (outside the 2:1 resonance). The distribution shows structure
    from Kirkwood gaps but we model it as a smooth distribution for simplicity.
    
    Distributions:
    - a: Truncated normal centered at 2.7 AU
    - e: Beta distribution (peaked at low e)
    - i: Rayleigh distribution (peaked at low i)
    - Ω, ω, M: Uniform
    """
    
    def __init__(
        self,
        a_mean: float = 2.7,
        a_std: float = 0.4,
        a_min: float = 2.1,
        a_max: float = 3.3,
        e_alpha: float = 2.0,
        e_beta: float = 5.0,
        i_scale: float = np.radians(8.0),  # ~8 deg characteristic inclination
    ):
        self.a_mean = a_mean
        self.a_std = a_std
        self.a_min = a_min
        self.a_max = a_max
        self.e_alpha = e_alpha
        self.e_beta = e_beta
        self.i_scale = i_scale
    
    @property
    def name(self) -> str:
        return "Main Belt Asteroids"
    
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> OrbitalElements:
        rng = self._get_rng(rng)
        
        # Semi-major axis: truncated normal
        a = np.zeros(n)
        mask = np.ones(n, dtype=bool)
        while np.any(mask):
            a[mask] = rng.normal(self.a_mean, self.a_std, np.sum(mask))
            mask = (a < self.a_min) | (a > self.a_max)
        
        # Eccentricity: beta distribution, scaled to max ~0.4
        e = rng.beta(self.e_alpha, self.e_beta, n) * 0.4
        
        # Inclination: Rayleigh distribution
        # For Rayleigh, sample from sin(i) ~ Rayleigh, but easier to sample i directly
        i = rng.rayleigh(self.i_scale, n)
        i = np.clip(i, 0, np.pi / 2)  # Cap at 90 degrees
        
        # Angular elements: uniform
        Omega = rng.uniform(0, 2 * np.pi, n)
        omega = rng.uniform(0, 2 * np.pi, n)
        M = rng.uniform(0, 2 * np.pi, n)
        
        return OrbitalElements(a, e, i, Omega, omega, M)
    
    def log_prob(self, elements: OrbitalElements) -> np.ndarray:
        from scipy.stats import norm, beta, rayleigh
        
        a, e, i = elements.a, elements.e, elements.i
        
        # Truncated normal for a
        log_p_a = norm.logpdf(a, self.a_mean, self.a_std)
        # Normalization for truncation
        norm_const = norm.cdf(self.a_max, self.a_mean, self.a_std) - \
                     norm.cdf(self.a_min, self.a_mean, self.a_std)
        log_p_a -= np.log(norm_const)
        log_p_a = np.where((a < self.a_min) | (a > self.a_max), -np.inf, log_p_a)
        
        # Beta for e (scaled)
        log_p_e = beta.logpdf(e / 0.4, self.e_alpha, self.e_beta) - np.log(0.4)
        log_p_e = np.where((e < 0) | (e > 0.4), -np.inf, log_p_e)
        
        # Rayleigh for i
        log_p_i = rayleigh.logpdf(i, scale=self.i_scale)
        log_p_i = np.where((i < 0) | (i > np.pi / 2), -np.inf, log_p_i)
        
        # Uniform for angles
        log_p_angles = -3 * np.log(2 * np.pi)
        
        return log_p_a + log_p_e + log_p_i + log_p_angles


class NEOPopulation(OrbitalPopulation):
    """
    Near-Earth Object population.
    
    NEOs have perihelion q < 1.3 AU. They span a wide range of semi-major axes
    and eccentricities.
    
    We model this with broad distributions:
    - a: Wide range from 0.5 to 4 AU, peaked around 1-2 AU
    - e: Must satisfy q = a(1-e) < 1.3 AU
    - i: Broader than Main Belt
    """
    
    def __init__(
        self,
        a_min: float = 0.5,
        a_max: float = 4.0,
        a_peak: float = 1.5,
        a_width: float = 0.8,
        q_max: float = 1.3,  # NEO definition
        i_scale: float = np.radians(15.0),
    ):
        self.a_min = a_min
        self.a_max = a_max
        self.a_peak = a_peak
        self.a_width = a_width
        self.q_max = q_max
        self.i_scale = i_scale
    
    @property
    def name(self) -> str:
        return "Near-Earth Objects"
    
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> OrbitalElements:
        rng = self._get_rng(rng)
        
        # Vectorized NEO sampling with efficient rejection
        # Oversample to account for rejection
        oversample_factor = 5
        samples_needed = n * oversample_factor
        
        # Sample a from truncated lognormal (vectorized rejection)
        a = np.empty(samples_needed)
        filled = 0
        while filled < samples_needed:
            batch = min(samples_needed * 2, 100000)
            candidates = rng.lognormal(
                np.log(self.a_peak), 
                np.log(1 + self.a_width / self.a_peak),
                batch
            )
            valid = (candidates >= self.a_min) & (candidates <= self.a_max)
            n_valid = np.sum(valid)
            take = min(n_valid, samples_needed - filled)
            a[filled:filled + take] = candidates[valid][:take]
            filled += take
        
        # Eccentricity constraint: q = a(1-e) < q_max => e > 1 - q_max/a
        e_min = np.maximum(0.0, 1 - self.q_max / a)
        e_max = np.minimum(0.95, 1 - 0.01 / a)  # Keep perihelion > 0.01 AU
        
        # Filter valid (a, e) pairs
        valid = e_max > e_min
        a = a[valid]
        e_min = e_min[valid]
        e_max = e_max[valid]
        
        # Sample e uniformly in valid range (vectorized)
        e = rng.uniform(e_min, e_max)
        
        # Trim to exactly n samples
        if len(a) < n:
            # Need more samples - recursive call with more oversampling
            # This is rare, so OK to be less efficient
            extra = self.sample(n - len(a), rng)
            a = np.concatenate([a, extra.a])
            e = np.concatenate([e, extra.e])
        
        a = a[:n]
        e = e[:n]
        
        # Inclination: broader Rayleigh
        i = rng.rayleigh(self.i_scale, n)
        i = np.clip(i, 0, np.pi)
        
        # Angular elements: uniform
        Omega = rng.uniform(0, 2 * np.pi, n)
        omega = rng.uniform(0, 2 * np.pi, n)
        M = rng.uniform(0, 2 * np.pi, n)
        
        return OrbitalElements(a, e, i, Omega, omega, M)
    
    def log_prob(self, elements: OrbitalElements) -> np.ndarray:
        # Simplified - would need proper normalization for real use
        return np.zeros(len(elements.a))


class TNOPopulation(OrbitalPopulation):
    """
    Trans-Neptunian Object population.
    
    Simplified model covering classical Kuiper Belt objects.
    - a: 30-50 AU (classical belt)
    - e: Low to moderate (0 - 0.3)
    - i: Bimodal (cold and hot populations)
    
    For simplicity, we model a mixed population.
    """
    
    def __init__(
        self,
        a_min: float = 30.0,
        a_max: float = 50.0,
        e_max: float = 0.3,
        i_cold_scale: float = np.radians(2.0),
        i_hot_scale: float = np.radians(15.0),
        cold_fraction: float = 0.5,
    ):
        self.a_min = a_min
        self.a_max = a_max
        self.e_max = e_max
        self.i_cold_scale = i_cold_scale
        self.i_hot_scale = i_hot_scale
        self.cold_fraction = cold_fraction
    
    @property
    def name(self) -> str:
        return "Trans-Neptunian Objects"
    
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> OrbitalElements:
        rng = self._get_rng(rng)
        
        # Semi-major axis: uniform in the classical belt
        a = rng.uniform(self.a_min, self.a_max, n)
        
        # Eccentricity: uniform in [0, e_max]
        e = rng.uniform(0, self.e_max, n)
        
        # Inclination: mixture of cold and hot (efficient sampling)
        n_cold = int(self.cold_fraction * n)
        n_hot = n - n_cold
        i_cold = rng.rayleigh(self.i_cold_scale, n_cold)
        i_hot = rng.rayleigh(self.i_hot_scale, n_hot)
        i = np.concatenate([i_cold, i_hot])
        rng.shuffle(i)  # Randomize order
        i = np.clip(i, 0, np.pi / 2)
        
        # Angular elements: uniform
        Omega = rng.uniform(0, 2 * np.pi, n)
        omega = rng.uniform(0, 2 * np.pi, n)
        M = rng.uniform(0, 2 * np.pi, n)
        
        return OrbitalElements(a, e, i, Omega, omega, M)
    
    def log_prob(self, elements: OrbitalElements) -> np.ndarray:
        return np.zeros(len(elements.a))


class JupiterTrojanPopulation(OrbitalPopulation):
    """
    Jupiter Trojan asteroid population.
    
    Trojans are co-orbital with Jupiter (a ≈ 5.2 AU) librating around
    L4 (leading) or L5 (trailing) Lagrange points.
    
    - a: Narrow range around Jupiter's semi-major axis
    - e: Low to moderate
    - i: Bimodal (similar to TNOs)
    """
    
    def __init__(
        self,
        a_jupiter: float = 5.203,
        a_spread: float = 0.15,
        e_mean: float = 0.05,
        e_spread: float = 0.05,
        i_scale: float = np.radians(15.0),
    ):
        self.a_jupiter = a_jupiter
        self.a_spread = a_spread
        self.e_mean = e_mean
        self.e_spread = e_spread
        self.i_scale = i_scale
    
    @property
    def name(self) -> str:
        return "Jupiter Trojans"
    
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> OrbitalElements:
        rng = self._get_rng(rng)
        
        # Semi-major axis: narrow Gaussian around Jupiter
        a = rng.normal(self.a_jupiter, self.a_spread, n)
        a = np.clip(a, self.a_jupiter - 0.5, self.a_jupiter + 0.5)
        
        # Eccentricity: half-normal (peaked at zero)
        e = np.abs(rng.normal(self.e_mean, self.e_spread, n))
        e = np.clip(e, 0, 0.2)
        
        # Inclination: Rayleigh
        i = rng.rayleigh(self.i_scale, n)
        i = np.clip(i, 0, np.pi / 2)
        
        # Angular elements
        # For Trojans, mean longitude is constrained relative to Jupiter
        # For simplicity, we'll use uniform here (full treatment would track libration)
        Omega = rng.uniform(0, 2 * np.pi, n)
        omega = rng.uniform(0, 2 * np.pi, n)
        M = rng.uniform(0, 2 * np.pi, n)
        
        return OrbitalElements(a, e, i, Omega, omega, M)
    
    def log_prob(self, elements: OrbitalElements) -> np.ndarray:
        return np.zeros(len(elements.a))


class HildaPopulation(OrbitalPopulation):
    """
    Hilda asteroid population (3:2 resonance with Jupiter).
    
    Hildas have a ≈ 4.0 AU and moderate eccentricities.
    """
    
    def __init__(
        self,
        a_center: float = 3.97,
        a_spread: float = 0.05,
        e_mean: float = 0.15,
        e_spread: float = 0.08,
        i_scale: float = np.radians(8.0),
    ):
        self.a_center = a_center
        self.a_spread = a_spread
        self.e_mean = e_mean
        self.e_spread = e_spread
        self.i_scale = i_scale
    
    @property
    def name(self) -> str:
        return "Hilda Asteroids"
    
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> OrbitalElements:
        rng = self._get_rng(rng)
        
        a = rng.normal(self.a_center, self.a_spread, n)
        a = np.clip(a, 3.7, 4.2)
        
        e = rng.normal(self.e_mean, self.e_spread, n)
        e = np.clip(e, 0.01, 0.35)
        
        i = rng.rayleigh(self.i_scale, n)
        i = np.clip(i, 0, np.pi / 3)
        
        Omega = rng.uniform(0, 2 * np.pi, n)
        omega = rng.uniform(0, 2 * np.pi, n)
        M = rng.uniform(0, 2 * np.pi, n)
        
        return OrbitalElements(a, e, i, Omega, omega, M)
    
    def log_prob(self, elements: OrbitalElements) -> np.ndarray:
        return np.zeros(len(elements.a))


# Convenience function to get all populations
def get_all_populations() -> dict:
    """Return dictionary of all available populations."""
    return {
        'mba': MainBeltPopulation(),
        'neo': NEOPopulation(),
        'tno': TNOPopulation(),
        'trojan': JupiterTrojanPopulation(),
        'hilda': HildaPopulation(),
    }
