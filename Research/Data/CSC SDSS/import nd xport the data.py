#Program for importing Chandra CSC-SDSS cross reference data as tab delimited txt files
#And unpacking them for calculations and filtering. Txt Files must be sorted by SDSS OBJID
#For results indexing to be consistant.

import numpy as np


DATA = np.genfromtxt('CSCdata.txt',skip_footer=2,names=True) #imports the entire catalog of cross matched objects into a dt.array
objid_ra_dec = np.genfromtxt('CSCdata msid objid ra dec.txt',skip_footer=2,skip_header=2,usecols=(1,2,3))
msid_ra_dec = np.genfromtxt('CSCdata msid objid ra dec.txt',skip_footer=2,skip_header=2,usecols=(0,2,3),dtype=float)

# The following commands extract data from Data array into callable dictionary arrays
MSID = DATA['MSID']
OBJID = DATA['OBJID']
SDSS_TYPE = DATA['SDSS_TYPE']
SEPARATION = DATA['SEPARATION']
FLUX_APER90_W = DATA['FLUX_APER90_W']
FLUX_APER90_LOLIM_W = DATA['FLUX_APER90_LOLIM_W']
FLUX_APER90_HILIM_W = DATA['FLUX_APER90_HILIM_W']
RA_SDSS = DATA['RA_SDSS']
DEC_SDSS = DATA['DEC_SDSS']
RA_SDSS_ERR = DATA['RA_SDSS_ERR']
DEC_SDSS_ERR = DATA['DEC_SDSS_ERR']
U = DATA['U']
ERR_U = DATA['ERR_U']
G = DATA['G']
ERR_G = DATA['ERR_G']
R = DATA['R']
ERR_R = DATA['ERR_R']
I = DATA['I']
ERR_I = DATA['ERR_I']
Z = DATA['Z']
ERR_Z = DATA['ERR_Z']
FLUX_APER90_B = DATA['FLUX_APER90_B']
FLUX_APER90_LOLIM_B = DATA['FLUX_APER90_LOLIM_B']
FLUX_APER90_HILIM_B = DATA['FLUX_APER90_HILIM_B']
SIGNIFICANCE = DATA['SIGNIFICANCE']
Line_number = np.arange(0,len(MSID))

'''
===============================================================================
'''

#np.savetxt('test export.txt',OBJID,header='Objid',fmt='%20.d')# fmt 20 unit iteger prevents python from converting to sci notation
#np.savetxt('ra dec.txt',objid_ra_dec, fmt='%f')
#np.savetxt('ra dec 2.txt',(RA_SDSS,DEC_SDSS), fmt='%f')
#np.savetxt('grrr.txt',(MSID,OBJID),delimiter=',',fmt='%f')
'''
with open ('processed_seq.txt','a') as proc_seqf:
    for a, am in zip(RA_SDSS, DEC_SDSS):
        proc_seqf.write("{}\t{}".format(a, am))
'''
'''
f = open("myfile.txt", "w")
#mylist = [1, 2 ,6 ,56, 78]
f.write("\n".join(map(lambda x: str(x), (RA_SDSS,DEC_SDSS))))
f.close()
'''
Line_number = np.arange(0,len(MSID))

outstring = zip(Line_number,RA_SDSS, DEC_SDSS)
f = open('PLEASE PLEASE PLEASE.txt', 'w')
#f.write('fuck my life')
for line in outstring:
    f.write(" ".join(str(x) for x in line) + "\n")
f.close()

'''
===============================================================================
'''

SDSS_DATA = np.genfromtxt('SDSS data mark 2.txt',skip_footer=2,names=True,delimiter=',') # CSC matched withing 1 arc second to sdss photometric objects
Line_number2 = SDSS_DATA['Line_number2']
objID = SDSS_DATA['objID']
ra = SDSS_DATA['ra']
dec = SDSS_DATA['dec']
photometric_type = SDSS_DATA['type']
p_U = SDSS_DATA['modelMag_u']
p_G = SDSS_DATA['modelMag_g']
p_R = SDSS_DATA['modelMag_r']
p_I = SDSS_DATA['modelMag_i']
p_Z = SDSS_DATA['modelMag_z']
Spectral_objid = SDSS_DATA['specObjID']
Redshift = SDSS_DATA['Redshift']

indexing = np.genfromtxt('SDSS data mark 2.txt',skip_footer=2,skip_header=1,delimiter=',',usecols=0,dtype=int)
'''
===============================================================================
'''


def Lux(flux,redshift):
    #c = 2.998e10 #cm/sec
    #h_o = 2.294e-18 #seconds
    #v = c * redshift #velocity in cm/sec
    #d = v / h_o #distance in centemeters
    #Lux = flux * 4 * np.pi * d**2
    c = 2.998e8
    h_o = 70.8 #seconds
    v = c * redshift #velocity in cm/sec
    d = v / h_o #distance in centemeters
    d_l = d * 3.086e24
    Lux = flux * 4 * np.pi * d_l**2
    return Lux

Lux_Values = Lux(FLUX_APER90_B[indexing],Redshift)

outstring2 = zip(Line_number,photometric_type,Lux_Values)
y = open('Lux mag values.txt', 'w')
#f.write('fuck my life')
for line in outstring2:
    y.write(" ".join(str(x) for x in line) + "\n")
y.close()

print FLUX_APER90_B[indexing]
print Redshift


