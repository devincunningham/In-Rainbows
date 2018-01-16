import numpy as np
import random  as rd
from numba import jit
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

G = 6.67408e-11 / 2e30 # m**3 kg**-1 s**-2 ----- Gravitational constant / M_sun

nParticles = 5000
nt_simulation = 10

#=======================================================================
#------------------------Setting up function----------------------------
#=======================================================================

@jit
def galsim(particle, particlev,nParticles):
    dt = .001
    G = 6.67408e-11 / 2e30
    for i in range(nParticles):
        Fx = 0.0
        Fy = 0.0
        for j in range(nParticles):
            if j != i:
                dx = particle[j,0] - particle[i,0]
                dy = particle[j,1] - particle[i,1]
                drSquared = dx * dx + dy * dy
                inverse_square = .025 / (drSquared)
                Fx += dx * inverse_square
                Fy += dy * inverse_square
            particlev[i, 0] += dt * Fx
            particlev[i, 1] += dt * Fy
    for i in range(nParticles):
        particle[i,0] += particlev[i,0] * dt
        particle[i,1] += particlev[i,1] * dt
    return particle, particlev

#=======================================================================
#--------------------------Setting up arrays----------------------------
#=======================================================================

#Setting up initial X and Y coordinates of each particle

#Galaxy One:
init_particle_1 = np.random.standard_normal((nParticles/4, 2))
init_particle_1_x = init_particle_1[:,0] / 2 ** 2
init_particle_1_y = init_particle_1[:,1] / 2 ** 2


#Galaxy Two:
init_particle_2 = np.random.standard_normal((nParticles*(.75), 2))
init_particle_2_x = init_particle_2[:,0] ** 2
init_particle_2_x = init_particle_2_x + 50 #Shifting this galaxy away from the first one
init_particle_2_y = init_particle_2[:,1] ** 2
init_particle_x = np.concatenate((init_particle_1_x,init_particle_2_x), axis=0)
init_particle_y = np.concatenate((init_particle_1_y,init_particle_2_y), axis=0)
init_particle = np.vstack((init_particle_x,init_particle_y)).T

#Setting up initial X and Y velocities of each particle
init_particlev = np.random.standard_normal((nParticles, 2))

print init_particle

#Creating the initial state of the universe:
init_gal = galsim(init_particle, init_particlev,nParticles)

#Initializing lists
states = []
x = []
y = []

#Creating lists of the X and Y coordinates of each particle from t=0 to t=nt_simulation
for t in range(nt_simulation):
    gal_data = galsim(init_gal[0],init_gal[1],nParticles)
    states.append(gal_data[0])
    current_state = states[t]
    current_copy_x = list(current_state[:,0])
    x.insert(t,current_copy_x)
    current_copy_y = list(current_state[:,1])
    y.insert(t,current_copy_y)
    init_gal = gal_data

#=======================================================================
#--------------------------Animation----------------------------
#=======================================================================

fig = plt.figure(figsize=(20,4))
ax = plt.axes(xlim=(-30, 80), ylim=(-10, 10))
stars, = ax.plot([],[],'o',color='wheat',alpha=0.7,markersize=1.5)
plt.title('Galaxy Simulaton')
plt.xlabel('x')
plt.ylabel('y')

def initial():
    stars.set_data([], [])
    return stars,

def animate(i):
    stars.set_data(x[i],y[i])
    fig.canvas.draw()
    return stars,

anim = animation.FuncAnimation(fig, animate, init_func=initial, frames=nt_simulation, interval=60, blit=True)
plt.show()