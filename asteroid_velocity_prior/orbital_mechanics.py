"""
Orbital Mechanics

Core orbital mechanics functions:
- Kepler equation solver
- Anomaly conversions (Mean ↔ Eccentric ↔ True)
- Orbital elements ↔ state vectors

Coordinate system: Heliocentric ecliptic (J2000)
Units: AU, AU/day, radians (unless specified)

Orbital Elements Convention:
    a   - Semi-major axis [AU]
    e   - Eccentricity [dimensionless]
    i   - Inclination [rad]
    Ω   - Longitude of ascending node [rad]  
    ω   - Argument of perihelion [rad]
    M   - Mean anomaly [rad]
"""

import numpy as np
from numpy import sin, cos, sqrt, arctan2, arccos


# Gravitational parameter: GM_sun in AU^3/day^2
# GM_sun = 1.32712440018e20 m^3/s^2
# 1 AU = 1.495978707e11 m, 1 day = 86400 s
# GM_sun = 2.959122082855911e-4 AU^3/day^2
GM_SUN = 2.959122082855911e-4  # AU^3/day^2


def solve_kepler(M, e, tol=1e-12, max_iter=50):
    """
    Solve Kepler's equation: M = E - e*sin(E)
    
    Parameters
    ----------
    M : float or ndarray
        Mean anomaly [rad]
    e : float or ndarray
        Eccentricity (0 <= e < 1 for elliptical orbits)
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    E : float or ndarray
        Eccentric anomaly [rad]
    """
    M = np.asarray(M)
    e = np.asarray(e)
    scalar_input = M.ndim == 0
    M = np.atleast_1d(M)
    e = np.atleast_1d(e)
    
    # Broadcast to common shape
    M, e = np.broadcast_arrays(M, e)
    
    # Initial guess (Danby's method for better convergence)
    E = M + 0.85 * e * np.sign(np.sin(M))
    
    # Newton-Raphson iteration
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        delta = f / f_prime
        E = E - delta
        if np.all(np.abs(delta) < tol):
            break
    
    if scalar_input:
        return E.item()
    return E


def eccentric_to_true_anomaly(E, e):
    """
    Convert eccentric anomaly to true anomaly.
    
    Parameters
    ----------
    E : float or ndarray
        Eccentric anomaly [rad]
    e : float or ndarray
        Eccentricity
        
    Returns
    -------
    nu : float or ndarray
        True anomaly [rad]
    """
    # tan(nu/2) = sqrt((1+e)/(1-e)) * tan(E/2)
    half_nu = arctan2(
        sqrt(1 + e) * np.sin(E / 2),
        sqrt(1 - e) * np.cos(E / 2)
    )
    return 2 * half_nu


def true_to_eccentric_anomaly(nu, e):
    """
    Convert true anomaly to eccentric anomaly.
    
    Parameters
    ----------
    nu : float or ndarray
        True anomaly [rad]
    e : float or ndarray
        Eccentricity
        
    Returns
    -------
    E : float or ndarray
        Eccentric anomaly [rad]
    """
    # tan(E/2) = sqrt((1-e)/(1+e)) * tan(nu/2)
    half_E = arctan2(
        sqrt(1 - e) * np.sin(nu / 2),
        sqrt(1 + e) * np.cos(nu / 2)
    )
    return 2 * half_E


def mean_to_true_anomaly(M, e):
    """
    Convert mean anomaly to true anomaly.
    
    Parameters
    ----------
    M : float or ndarray
        Mean anomaly [rad]
    e : float or ndarray
        Eccentricity
        
    Returns
    -------
    nu : float or ndarray
        True anomaly [rad]
    """
    E = solve_kepler(M, e)
    return eccentric_to_true_anomaly(E, e)


def true_to_mean_anomaly(nu, e):
    """
    Convert true anomaly to mean anomaly.
    
    Parameters
    ----------
    nu : float or ndarray
        True anomaly [rad]
    e : float or ndarray
        Eccentricity
        
    Returns
    -------
    M : float or ndarray
        Mean anomaly [rad]
    """
    E = true_to_eccentric_anomaly(nu, e)
    return E - e * np.sin(E)


def orbital_elements_to_state(a, e, i, Omega, omega, M):
    """
    Convert orbital elements to heliocentric state vector.
    
    Parameters
    ----------
    a : float or ndarray
        Semi-major axis [AU]
    e : float or ndarray
        Eccentricity
    i : float or ndarray
        Inclination [rad]
    Omega : float or ndarray
        Longitude of ascending node [rad]
    omega : float or ndarray
        Argument of perihelion [rad]
    M : float or ndarray
        Mean anomaly [rad]
        
    Returns
    -------
    r : ndarray, shape (..., 3)
        Heliocentric position [AU], ecliptic coordinates
    v : ndarray, shape (..., 3)
        Heliocentric velocity [AU/day], ecliptic coordinates
    """
    # Ensure arrays
    a = np.asarray(a)
    e = np.asarray(e)
    i = np.asarray(i)
    Omega = np.asarray(Omega)
    omega = np.asarray(omega)
    M = np.asarray(M)
    
    # Broadcast to common shape
    a, e, i, Omega, omega, M = np.broadcast_arrays(a, e, i, Omega, omega, M)
    original_shape = a.shape
    
    # Flatten for computation
    a = a.ravel()
    e = e.ravel()
    i = i.ravel()
    Omega = Omega.ravel()
    omega = omega.ravel()
    M = M.ravel()
    
    # True anomaly
    nu = mean_to_true_anomaly(M, e)
    
    # Distance from focus
    r_mag = a * (1 - e**2) / (1 + e * cos(nu))
    
    # Position in orbital plane (perifocal coordinates)
    # x-axis points to perihelion, z-axis is angular momentum
    x_orb = r_mag * cos(nu)
    y_orb = r_mag * sin(nu)
    
    # Velocity in orbital plane
    # From vis-viva: v^2 = GM(2/r - 1/a)
    # Components: v_x = -sqrt(GM/p) * sin(nu), v_y = sqrt(GM/p) * (e + cos(nu))
    p = a * (1 - e**2)  # Semi-latus rectum
    h = sqrt(GM_SUN * p)  # Specific angular momentum magnitude
    
    vx_orb = -GM_SUN / h * sin(nu)
    vy_orb = GM_SUN / h * (e + cos(nu))
    
    # Rotation matrices: R = R_z(-Omega) @ R_x(-i) @ R_z(-omega)
    # or equivalently, the combined rotation from orbital to ecliptic frame
    
    cos_O = cos(Omega)
    sin_O = sin(Omega)
    cos_i = cos(i)
    sin_i = sin(i)
    cos_w = cos(omega)
    sin_w = sin(omega)
    
    # Combined rotation matrix elements (Murray & Dermott Eq. 2.119-2.121)
    P1 = cos_w * cos_O - sin_w * sin_O * cos_i
    P2 = cos_w * sin_O + sin_w * cos_O * cos_i
    P3 = sin_w * sin_i
    
    Q1 = -sin_w * cos_O - cos_w * sin_O * cos_i
    Q2 = -sin_w * sin_O + cos_w * cos_O * cos_i
    Q3 = cos_w * sin_i
    
    # Position in ecliptic coordinates
    x = P1 * x_orb + Q1 * y_orb
    y = P2 * x_orb + Q2 * y_orb
    z = P3 * x_orb + Q3 * y_orb
    
    # Velocity in ecliptic coordinates
    vx = P1 * vx_orb + Q1 * vy_orb
    vy = P2 * vx_orb + Q2 * vy_orb
    vz = P3 * vx_orb + Q3 * vy_orb
    
    # Stack into arrays
    r = np.stack([x, y, z], axis=-1)
    v = np.stack([vx, vy, vz], axis=-1)
    
    # Reshape to original shape + (3,)
    if original_shape == ():
        r = r.squeeze()
        v = v.squeeze()
    else:
        r = r.reshape(original_shape + (3,))
        v = v.reshape(original_shape + (3,))
    
    return r, v


def state_to_orbital_elements(r, v):
    """
    Convert heliocentric state vector to orbital elements.
    
    Parameters
    ----------
    r : ndarray, shape (..., 3)
        Heliocentric position [AU], ecliptic coordinates
    v : ndarray, shape (..., 3)
        Heliocentric velocity [AU/day], ecliptic coordinates
        
    Returns
    -------
    elements : dict
        Dictionary with keys 'a', 'e', 'i', 'Omega', 'omega', 'M', 'nu'
        All angles in radians
    """
    r = np.asarray(r)
    v = np.asarray(v)
    
    # Handle shapes
    if r.shape[-1] != 3:
        raise ValueError("Last dimension of r must be 3")
    
    original_shape = r.shape[:-1]
    r = r.reshape(-1, 3)
    v = v.reshape(-1, 3)
    
    # Magnitudes
    r_mag = np.linalg.norm(r, axis=-1)
    v_mag = np.linalg.norm(v, axis=-1)
    
    # Specific angular momentum
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h, axis=-1)
    
    # Node vector (z × h)
    z_hat = np.array([0, 0, 1])
    n = np.cross(z_hat, h)
    n_mag = np.linalg.norm(n, axis=-1)
    
    # Eccentricity vector
    # e_vec = ((v^2 - GM/r) * r - (r·v) * v) / GM
    rdotv = np.sum(r * v, axis=-1)
    e_vec = ((v_mag**2 - GM_SUN / r_mag)[:, np.newaxis] * r - 
             rdotv[:, np.newaxis] * v) / GM_SUN
    e = np.linalg.norm(e_vec, axis=-1)
    
    # Semi-major axis (vis-viva)
    energy = v_mag**2 / 2 - GM_SUN / r_mag
    a = -GM_SUN / (2 * energy)
    
    # Inclination
    i = arccos(np.clip(h[:, 2] / h_mag, -1, 1))
    
    # Longitude of ascending node
    Omega = arctan2(n[:, 1], n[:, 0])
    Omega = np.mod(Omega, 2 * np.pi)
    
    # Argument of perihelion
    # cos(omega) = n · e / (|n| |e|)
    ndote = np.sum(n * e_vec, axis=-1)
    omega = arccos(np.clip(ndote / (n_mag * e + 1e-30), -1, 1))
    # Quadrant check: if e_z < 0, omega is in [π, 2π]
    omega = np.where(e_vec[:, 2] < 0, 2 * np.pi - omega, omega)
    
    # True anomaly
    edotr = np.sum(e_vec * r, axis=-1)
    nu = arccos(np.clip(edotr / (e * r_mag + 1e-30), -1, 1))
    # Quadrant check: if r · v < 0, nu is in [π, 2π]
    nu = np.where(rdotv < 0, 2 * np.pi - nu, nu)
    
    # Mean anomaly
    M = true_to_mean_anomaly(nu, e)
    M = np.mod(M, 2 * np.pi)
    
    # Reshape outputs
    def reshape_output(arr):
        if original_shape == ():
            return arr.item() if arr.size == 1 else arr.squeeze()
        return arr.reshape(original_shape)
    
    return {
        'a': reshape_output(a),
        'e': reshape_output(e),
        'i': reshape_output(i),
        'Omega': reshape_output(Omega),
        'omega': reshape_output(omega),
        'M': reshape_output(M),
        'nu': reshape_output(nu),
    }


def mean_motion(a):
    """
    Compute mean motion from semi-major axis.
    
    Parameters
    ----------
    a : float or ndarray
        Semi-major axis [AU]
        
    Returns
    -------
    n : float or ndarray
        Mean motion [rad/day]
    """
    return sqrt(GM_SUN / a**3)


def orbital_period(a):
    """
    Compute orbital period from semi-major axis.
    
    Parameters
    ----------
    a : float or ndarray
        Semi-major axis [AU]
        
    Returns
    -------
    P : float or ndarray
        Orbital period [days]
    """
    return 2 * np.pi / mean_motion(a)


def propagate_mean_anomaly(M0, a, dt):
    """
    Propagate mean anomaly by a time interval.
    
    Parameters
    ----------
    M0 : float or ndarray
        Initial mean anomaly [rad]
    a : float or ndarray
        Semi-major axis [AU]
    dt : float or ndarray
        Time interval [days]
        
    Returns
    -------
    M : float or ndarray
        Final mean anomaly [rad], wrapped to [0, 2π)
    """
    n = mean_motion(a)
    M = M0 + n * dt
    return np.mod(M, 2 * np.pi)
