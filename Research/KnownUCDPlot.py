import matplotlib.pyplot as plt
import numpy as np

u, g, r, i, z, pet_r, d = np.genfromtxt("KnownUCDs.txt", skip_header=1,usecols=(1,2,3,4,5,6,7), unpack=True);
name = np.genfromtxt("KnownUCDs.txt", dtype=str,skip_header=1,usecols=0);

m_V = g - 0.5784 * (g-r) - 0.0038;
DistanceModulus = 5 *  (np.log10(d*1e-6) + 5);
M = m_V - DistanceModulus;
color = g-i;

fig1 = plt.figure(figsize=(8,6));

plt.plot(color,M,'mo');
plt.xlabel('Color')
plt.ylabel('Absolute Magnitude')
plt.title('Color vs. Absolute Magnitude of Known UCDs')
plt.ylim(-8,-15)
plt.xlim(0.6,1.3)
j=0
for i in name:
    plt.annotate(i, xy=(color[j]+0.01,M[j]+0.11));
    j=j+1;
plt.show()
fig1.savefig('KnownUCDPlot.jpg')



