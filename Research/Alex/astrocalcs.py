from pylab import *
import numpy as np
import os as os

def effective_radius_pc(Theta,distance_mpc):
    Rad_effective_pc = (Theta) * (distance_mpc *1e6 / 2e5)
    return Rad_effective_pc

def recession_velocity_kms(z):
    recession_velocity_kms = z * 3e5
    return recession_velocity_kms
    
def distance_to_obj_mpc(recession_velocity_kms):
    distance_mpc = recession_velocity_kms / 67.8
    return distance_mpc

def V_band_magnitude(g_mag,r_mag):
    V_mag = g_mag - 0.565 * (g_mag - r_mag) - 0.016
    return V_mag

def absolute_V_band_magnitude(V_mag,distance_mpc): 
    absolute_V_mag = V_mag - 5 * log10((distance_mpc *(1e6))/10)
    return absolute_V_mag
    
def Luminosity_V_band(absolute_V_mag):
    Luminosity_V = 10**((4.81-absolute_V_mag)/2.5)
    return Luminosity_V
    
def Radius_half_kpc(Rad_effective_pc):
    Radius_half_kpc = 10**log10(Rad_effective_pc) / 1000.0
    return Radius_half_kpc

def full_radius_kpc(petro_rad,distance_mpc):
    full_radius = (petro_rad) * ((distance_mpc *(1e6)) / 2e5) #in parsecs
    return full_radius


def Stellar_mass_solmass(absolute_V_mag):
    Stellar_mass = 2 * 10**((4.81-absolute_V_mag)/2.5)
    return Stellar_mass

def Dynamical_mass_solmass(sigma,full_radius):
    grav = 6.67408e-11
    Sol_mass = 1.9885e30
    Dynamical_mass = ((5 * ((sigma * 1000)**2) *  full_radius) * 3.086e16/ grav) / Sol_mass
    return Dynamical_mass

def constants_used():
    print 'M_Sun_v = 4.81 for Solar absolute magnitud, V-band'
    print 'Sol_mass = 1.9885e30 in kg'
    print 'c = 3e5 in km/sec'
    print 'H0 = 67.8 Hubble constant in (km/s/Mpc)'
    print 'parsec_to_meter = 3.086e16 m'
    print 'grav = 6.67408e-11'

def writer(objid,ra,dec):
    name = raw_input('Choose a file name ')+'.csv'
    Line_number = arange(0,len(objid))
    outstring = zip(objid,ra, dec)
    f = open(name, 'w')
    f.write('objid,ra, dec \n')
    for line in outstring:
        f.write(",".join(str(x) for x in line) + "\n")
    f.close()
    return writer

def scaling_upper_theta(redshift):
    theta_upper = ( (.700 * 1000) / (obj_redshift * 22123.89) )
    return theta_upper
    
def scaling_lower_theta(redshift):
    theta_lower = ( (.077 * 1000) / (obj_redshift * 22123.89) )
    return theta_lower
    
def devexp_radius(deVRad_r,fracDeV_r,expRad_r):
    devexp = ( (deVRad_r * fracDeV_r) + expRad_r * (1-fracDeV_r))
    return devexp
    
def Lv_in_g_r_z(g,r,z):
    Lv_in_g_r_z = 10**((4.81-((g-0.565*(g-r)-0.016)-5*log10(z*(3e10/67.8))))/2.5)
    return Lv_in_g_r_z
    
def rough_estimate_density(Lv,Radius_half_kpc):
    Mass_to_Light_V = 2.0
    ro_Ms_per_Kpc2 = ( ((Mass_to_Light_V * Lv)/2.0) / (pi * Radius_half_kpc**2) )
    return ro_Ms_per_Kpc2
    
def color(a,b):
    color = a - b
    return color
#for SQL ((( (deVRad_r * fracDeV_r) + expRad_r * (1-fracDeV_r))) * (((z * 3e5) / 67.8) *1e6 / 2e5)) / 1000.0 <= ( 10**((4.81-((g-0.565*(g-r)-0.016)-5*log10(z*(3e10/67.8))))/2.5) + 1e7)*(3e-12)
#================================================================================
#********************************************************************************
#================================================================================    
