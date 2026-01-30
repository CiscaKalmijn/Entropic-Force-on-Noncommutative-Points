import numpy as np

# --- FORCE DEFINITION ---
def EntropicForce(r,beta: float=1 ):
    return np.where(r<1e-3,
                    0,
                    (beta / (2* r**3) ) / np.cosh( beta / ( 2*r ))**2 )

# --- DRAWING TRIANGLE FROM SIDES ---
def coordinating(sidesTriangle):
    a, b, c = sidesTriangle
    if a<0 or b<0 or c<0:
        print(f'Invalid sides given')
    else:
        #We fix points (0,0) and (0,c), so the y-coordinate of the third point is the height of the triangle:
        y = a*np.sin(np.arccos((a**2+c**2-b**2)/(2*a*c)))       
        #From Pythagorean theorem we get
        x = np.sqrt(b**2-y**2)        
        # Returning the coordinates of the vertices
        if np.arccos((b**2+c**2-a**2)/(2*b*c))<np.arccos(0):
            return [[0.0, 0.0], [float(c), 0.0], [float(np.around(x,1)), float(np.around(y,1))]]
        #If np.arccos((b**2+c**2-a**2)/(2*b*c))>np.arccos(0)=90 degrees we need -x as x-coordinate of the third point
        else:
            return [[0.0, 0.0], [float(c), 0.0], [-float(np.around(x,1)), float(np.around(y,1))]]

# --- CLASS of DISTANCES and ACTION of ENTROPIC FORCE ---
class PolygonDistances:
    def __init__(self, initial_distances: np.ndarray, masses: np.ndarray = None, name: str= None):          #Making the object, consisting of the distances, masses of these distances and a name for reference later, works for any amount of pairs/distances
        self.initial_distances= initial_distances                                                           #No limitations are placed on the distances, while not enforced here, negative distances give nonsensical answers, as entropic force only defined in non-negative case
        self.masses = masses if masses is not None else np.array([2]*self.number_of_distances())            #Here we attribute a weight to each distance (so each pair), instead of per point....
        self.name = name if name is not None else f"A random collection of {self.number_of_distances()} distances between points"
    def number_of_distances(self) -> int:
        return len(self.initial_distances)     
#Modelling of entropic force acting on an element of the class
    def EntropicTrajectory(self, dt: float=0.03, steps: int=800,beta: float=1):
        distances = np.array(self.initial_distances)
        trace_distances = [distances.copy()]
        velocities = np.zeros_like(self.initial_distances)
        trace_velocities = [velocities.copy()]
        for _ in range(steps):
            for i in range(self.number_of_distances()):
                velocities[i] += EntropicForce(distances[i],beta)*dt / self.masses[i]
                distances[i] += velocities[i]*dt
            trace_distances.append(distances.copy())
            trace_velocities.append(velocities.copy())
        return np.vstack(trace_distances)                       #Array of distances at each time step

# --- EXPLICIT EXAMPLE ---
pythagoras = PolygonDistances(np.array([3.0,4.0,5.0],dtype=float), name='Pythagorean Triangle')
pythagoras_evolved = np.around(pythagoras.EntropicTrajectory(dt=0.1, steps=2000),1)

print(f'Pythagorean triangle evolves from {pythagoras_evolved[0]} to {pythagoras_evolved[500]} to {pythagoras_evolved[1000]} to {pythagoras_evolved[1500]} to {pythagoras_evolved[2000]}')    

for sides in [pythagoras_evolved[0], pythagoras_evolved[500], pythagoras_evolved[1000], pythagoras_evolved[1500], pythagoras_evolved[2000]]:
    print(f'Triangle with sides {sides} has coordinates {coordinating(sides)}')