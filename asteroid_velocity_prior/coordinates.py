"""
Coordinate Transformations

Transformations between:
- Ecliptic ↔ Equatorial coordinates
- Heliocentric ↔ Geocentric frames
- 3D Cartesian ↔ Spherical sky coordinates
- 3D velocity → Sky-plane velocity

Units: AU, AU/day, radians (degrees where noted)
"""

import numpy as np
from numpy import sin, cos, arctan2, arcsin

# Obliquity of the ecliptic (J2000)
OBLIQUITY_J2000 = np.radians(23.439291)  # radians


def ecliptic_to_equatorial(x_ecl, y_ecl, z_ecl, obliquity=OBLIQUITY_J2000):
    """
    Transform ecliptic Cartesian coordinates to equatorial (J2000).
    
    Parameters
    ----------
    x_ecl, y_ecl, z_ecl : float or ndarray
        Ecliptic Cartesian coordinates
    obliquity : float
        Obliquity of the ecliptic [rad]
        
    Returns
    -------
    x_eq, y_eq, z_eq : float or ndarray
        Equatorial Cartesian coordinates
    """
    cos_eps = cos(obliquity)
    sin_eps = sin(obliquity)
    
    x_eq = x_ecl
    y_eq = cos_eps * y_ecl - sin_eps * z_ecl
    z_eq = sin_eps * y_ecl + cos_eps * z_ecl
    
    return x_eq, y_eq, z_eq


def equatorial_to_ecliptic(x_eq, y_eq, z_eq, obliquity=OBLIQUITY_J2000):
    """
    Transform equatorial Cartesian coordinates to ecliptic (J2000).
    
    Parameters
    ----------
    x_eq, y_eq, z_eq : float or ndarray
        Equatorial Cartesian coordinates
    obliquity : float
        Obliquity of the ecliptic [rad]
        
    Returns
    -------
    x_ecl, y_ecl, z_ecl : float or ndarray
        Ecliptic Cartesian coordinates
    """
    cos_eps = cos(obliquity)
    sin_eps = sin(obliquity)
    
    x_ecl = x_eq
    y_ecl = cos_eps * y_eq + sin_eps * z_eq
    z_ecl = -sin_eps * y_eq + cos_eps * z_eq
    
    return x_ecl, y_ecl, z_ecl


def cartesian_to_spherical(x, y, z, degrees=False):
    """
    Convert Cartesian to spherical coordinates.
    
    Parameters
    ----------
    x, y, z : float or ndarray
        Cartesian coordinates
    degrees : bool
        If True, return angles in degrees
        
    Returns
    -------
    r : float or ndarray
        Radial distance
    lon : float or ndarray
        Longitude (azimuthal angle), [0, 2π) or [0°, 360°)
    lat : float or ndarray
        Latitude (polar angle from equator), [-π/2, π/2] or [-90°, 90°]
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    lon = arctan2(y, x)
    lon = np.mod(lon, 2 * np.pi)  # [0, 2π)
    lat = arcsin(np.clip(z / (r + 1e-30), -1, 1))
    
    if degrees:
        lon = np.degrees(lon)
        lat = np.degrees(lat)
    
    return r, lon, lat


def spherical_to_cartesian(r, lon, lat, degrees=False):
    """
    Convert spherical to Cartesian coordinates.
    
    Parameters
    ----------
    r : float or ndarray
        Radial distance
    lon : float or ndarray
        Longitude [rad or deg]
    lat : float or ndarray
        Latitude [rad or deg]
    degrees : bool
        If True, input angles are in degrees
        
    Returns
    -------
    x, y, z : float or ndarray
        Cartesian coordinates
    """
    if degrees:
        lon = np.radians(lon)
        lat = np.radians(lat)
    
    x = r * cos(lat) * cos(lon)
    y = r * cos(lat) * sin(lon)
    z = r * sin(lat)
    
    return x, y, z


def get_earth_position(time):
    """
    Get Earth's heliocentric position and velocity at a given time.
    
    Uses astropy for accurate ephemeris.
    
    Parameters
    ----------
    time : astropy.time.Time or str
        Observation time
        
    Returns
    -------
    r_earth : ndarray, shape (3,)
        Earth's heliocentric position [AU], ecliptic coordinates
    v_earth : ndarray, shape (3,)
        Earth's heliocentric velocity [AU/day], ecliptic coordinates
    """
    from astropy.time import Time
    from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
    from astropy import units as u
    
    if not isinstance(time, Time):
        time = Time(time)
    
    # Use JPL ephemeris for accuracy
    with solar_system_ephemeris.set('builtin'):
        # Get Earth-Moon barycenter position relative to solar system barycenter
        earth_pos, earth_vel = get_body_barycentric_posvel('earth', time)
        sun_pos, sun_vel = get_body_barycentric_posvel('sun', time)
        
        # Convert to heliocentric
        # Position: CartesianRepresentation - use .get_xyz() for array
        r_earth_eq = (earth_pos - sun_pos).get_xyz().to(u.AU).value
        
        # Velocity: CartesianRepresentation (with AU/day units) - access .x, .y, .z
        vel_diff = earth_vel - sun_vel
        v_earth_eq = np.array([
            vel_diff.x.to(u.AU / u.day).value,
            vel_diff.y.to(u.AU / u.day).value,
            vel_diff.z.to(u.AU / u.day).value,
        ])
    
    # Convert from equatorial (ICRS) to ecliptic
    r_earth = np.array(equatorial_to_ecliptic(*r_earth_eq))
    v_earth = np.array(equatorial_to_ecliptic(*v_earth_eq))
    
    return r_earth, v_earth


def heliocentric_to_geocentric(r_helio, v_helio, r_earth, v_earth):
    """
    Transform from heliocentric to geocentric frame.
    
    Parameters
    ----------
    r_helio : ndarray, shape (..., 3)
        Heliocentric position [AU]
    v_helio : ndarray, shape (..., 3)
        Heliocentric velocity [AU/day]
    r_earth : ndarray, shape (3,)
        Earth's heliocentric position [AU]
    v_earth : ndarray, shape (3,)
        Earth's heliocentric velocity [AU/day]
        
    Returns
    -------
    r_geo : ndarray, shape (..., 3)
        Geocentric position [AU]
    v_geo : ndarray, shape (..., 3)
        Geocentric velocity [AU/day]
    """
    r_geo = r_helio - r_earth
    v_geo = v_helio - v_earth
    return r_geo, v_geo


def geocentric_to_radec(r_geo, degrees=True):
    """
    Convert geocentric Cartesian to RA, Dec.
    
    Parameters
    ----------
    r_geo : ndarray, shape (..., 3)
        Geocentric position [AU], ecliptic coordinates
        
    Returns
    -------
    ra : float or ndarray
        Right ascension [deg or rad]
    dec : float or ndarray
        Declination [deg or rad]
    distance : float or ndarray
        Geocentric distance [AU]
    """
    r_geo = np.asarray(r_geo)
    
    # Convert to equatorial
    x_eq, y_eq, z_eq = ecliptic_to_equatorial(
        r_geo[..., 0], r_geo[..., 1], r_geo[..., 2]
    )
    
    # Convert to spherical
    distance, ra, dec = cartesian_to_spherical(x_eq, y_eq, z_eq, degrees=degrees)
    
    return ra, dec, distance


def radec_to_unit_vector(ra, dec, degrees=True):
    """
    Convert RA, Dec to unit vector in equatorial Cartesian coordinates.
    
    Parameters
    ----------
    ra : float or ndarray
        Right ascension [deg or rad]
    dec : float or ndarray
        Declination [deg or rad]
    degrees : bool
        If True, input is in degrees
        
    Returns
    -------
    unit_vec : ndarray, shape (..., 3)
        Unit vector in equatorial Cartesian coordinates
    """
    if degrees:
        ra = np.radians(ra)
        dec = np.radians(dec)
    
    x = cos(dec) * cos(ra)
    y = cos(dec) * sin(ra)
    z = sin(dec)
    
    return np.stack([x, y, z], axis=-1)


def geocentric_to_sky_velocity(r_geo, v_geo):
    """
    Compute sky-plane velocity (proper motion) from geocentric state.
    
    The sky-plane velocity is the component of velocity perpendicular to
    the line of sight, expressed in angular units.
    
    Parameters
    ----------
    r_geo : ndarray, shape (..., 3)
        Geocentric position [AU], ecliptic coordinates
    v_geo : ndarray, shape (..., 3)
        Geocentric velocity [AU/day], ecliptic coordinates
        
    Returns
    -------
    mu_ra_cosdec : ndarray
        Proper motion in RA*cos(Dec) [arcsec/hour]
    mu_dec : ndarray
        Proper motion in Dec [arcsec/hour]
    v_radial : ndarray
        Radial velocity [AU/day]
    """
    r_geo = np.asarray(r_geo)
    v_geo = np.asarray(v_geo)
    
    # Convert position to equatorial
    x_eq, y_eq, z_eq = ecliptic_to_equatorial(
        r_geo[..., 0], r_geo[..., 1], r_geo[..., 2]
    )
    r_eq = np.stack([x_eq, y_eq, z_eq], axis=-1)
    
    # Convert velocity to equatorial  
    vx_eq, vy_eq, vz_eq = ecliptic_to_equatorial(
        v_geo[..., 0], v_geo[..., 1], v_geo[..., 2]
    )
    v_eq = np.stack([vx_eq, vy_eq, vz_eq], axis=-1)
    
    # Distance
    distance = np.linalg.norm(r_eq, axis=-1)
    
    # Unit vector to object (line of sight)
    r_hat = r_eq / distance[..., np.newaxis]
    
    # RA, Dec of object
    ra = arctan2(y_eq, x_eq)
    dec = arcsin(np.clip(z_eq / distance, -1, 1))
    
    # Unit vectors in RA and Dec directions (tangent plane basis)
    # e_ra points East (increasing RA)
    # e_dec points North (increasing Dec)
    e_ra = np.stack([-sin(ra), cos(ra), np.zeros_like(ra)], axis=-1)
    e_dec = np.stack([-sin(dec) * cos(ra), -sin(dec) * sin(ra), cos(dec)], axis=-1)
    
    # Radial velocity (along line of sight)
    v_radial = np.sum(v_eq * r_hat, axis=-1)
    
    # Tangential velocity components
    v_ra = np.sum(v_eq * e_ra, axis=-1)  # AU/day
    v_dec = np.sum(v_eq * e_dec, axis=-1)  # AU/day
    
    # Convert to angular velocity
    # mu = v_transverse / distance [rad/day]
    # Convert to arcsec/hour: 
    #   rad/day * (180/π * 3600 arcsec/rad) * (1 day / 24 hour)
    #   = rad/day * 206264.8 / 24 arcsec/hour
    #   = rad/day * 8594.37 arcsec/hour
    ARCSEC_PER_RADIAN_PER_HOUR = 206264.806 / 24.0  # arcsec/hour per rad/day
    
    mu_ra_cosdec = (v_ra / distance) * ARCSEC_PER_RADIAN_PER_HOUR  # already includes cos(dec)
    mu_dec = (v_dec / distance) * ARCSEC_PER_RADIAN_PER_HOUR
    
    return mu_ra_cosdec, mu_dec, v_radial


def sky_velocity_to_angle_speed(mu_ra_cosdec, mu_dec):
    """
    Convert (mu_ra*cos(dec), mu_dec) to total proper motion and position angle.
    
    Parameters
    ----------
    mu_ra_cosdec : float or ndarray
        Proper motion in RA*cos(Dec) [arcsec/hour]
    mu_dec : float or ndarray
        Proper motion in Dec [arcsec/hour]
        
    Returns
    -------
    mu_total : float or ndarray
        Total proper motion [arcsec/hour]
    position_angle : float or ndarray
        Position angle of motion [degrees], measured East from North
    """
    mu_total = np.sqrt(mu_ra_cosdec**2 + mu_dec**2)
    # Position angle: 0° = North, 90° = East
    position_angle = np.degrees(arctan2(mu_ra_cosdec, mu_dec))
    position_angle = np.mod(position_angle, 360)
    
    return mu_total, position_angle


def ecliptic_longitude_latitude_from_radec(ra, dec, degrees=True):
    """
    Convert RA, Dec to ecliptic longitude and latitude.
    
    Parameters
    ----------
    ra : float or ndarray
        Right ascension
    dec : float or ndarray
        Declination
    degrees : bool
        If True, angles are in degrees
        
    Returns
    -------
    ecl_lon : float or ndarray
        Ecliptic longitude
    ecl_lat : float or ndarray
        Ecliptic latitude
    """
    # RA, Dec -> equatorial Cartesian
    unit_eq = radec_to_unit_vector(ra, dec, degrees=degrees)
    
    # Equatorial -> ecliptic Cartesian
    x_ecl, y_ecl, z_ecl = equatorial_to_ecliptic(
        unit_eq[..., 0], unit_eq[..., 1], unit_eq[..., 2]
    )
    
    # Ecliptic Cartesian -> ecliptic spherical
    _, ecl_lon, ecl_lat = cartesian_to_spherical(x_ecl, y_ecl, z_ecl, degrees=degrees)
    
    return ecl_lon, ecl_lat
