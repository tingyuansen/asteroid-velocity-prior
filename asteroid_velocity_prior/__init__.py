"""
Asteroid Velocity Prior for Smart Stacking

Compute the prior distribution of sky-plane velocities for asteroid populations
at given sky positions and observation times.
"""

from .orbital_mechanics import (
    solve_kepler,
    mean_to_true_anomaly,
    orbital_elements_to_state,
    state_to_orbital_elements,
)

from .coordinates import (
    ecliptic_to_equatorial,
    equatorial_to_ecliptic,
    get_earth_position,
    heliocentric_to_geocentric,
    geocentric_to_sky_velocity,
)

from .populations import (
    OrbitalPopulation,
    MainBeltPopulation,
    NEOPopulation,
    TNOPopulation,
    JupiterTrojanPopulation,
)

from .velocity_prior import (
    VelocityPriorCalculator,
    VelocityPriorResult,
    compute_search_space_reduction,
    compute_sky_reduction_map,
)

__version__ = "0.1.0"

__all__ = [
    # Orbital mechanics
    "solve_kepler",
    "mean_to_true_anomaly", 
    "orbital_elements_to_state",
    "state_to_orbital_elements",
    # Coordinates
    "ecliptic_to_equatorial",
    "equatorial_to_ecliptic",
    "get_earth_position",
    "heliocentric_to_geocentric",
    "geocentric_to_sky_velocity",
    # Populations
    "OrbitalPopulation",
    "MainBeltPopulation",
    "NEOPopulation",
    "TNOPopulation",
    "JupiterTrojanPopulation",
    # Main calculator and results
    "VelocityPriorCalculator",
    "VelocityPriorResult",
    "compute_search_space_reduction",
    "compute_sky_reduction_map",
]
