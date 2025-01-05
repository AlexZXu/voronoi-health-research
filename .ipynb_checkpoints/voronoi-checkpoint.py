import numpy as np #numpy for lin alg and data functions
import matplotlib.pyplot as plt #matplotlib for graphing and visual
from scipy.spatial import Voronoi, Delaunay, voronoi_plot_2d #scipy to use Voronoi helpers
import sys #system helpers
import math #math helpers
from matplotlib.widgets import Button, Slider

#method that starts voronoi graph
def startVoronoi(source_points, bounding_box, refresh_time):
    #create matplotlib figure
    fig, ax = plt.subplots()
    
    #modify source points to include distant ghost points so the actual facilities' cells won't be "infinite" in size
    upd_source_points = np.append(source_points, [[9999,9999], [-9999,9999], [9999,-9999], [-9999,-9999]], axis = 0)
    #start matplotlib interactive mode
    plt.ion()
    
    #define colors
    colors = {
        13: "#ffad66",
        12: "#f2dd1b",
        11: "#5ea86e",
        10: "#5ea86e",
        9: "#58db74",
        8: "#bdf0c8",
        7: "#60e06d",
    }
    
    it = 0
    
    #while true keeps interactive mode updating
    while (True):
        print(it)
        #calculate voronoi diagram using helper
        vor = Voronoi(upd_source_points)
        #plot the voronoi diagram
        voronoi_plot_2d(vor, ax=ax)
        
        source_point_pops = recomputePopulations(pop_points, upd_source_points)
        
        #set bounds from bounding_box
        # ax.set_xlim(bounding_box[0],bounding_box[1]) 
        # ax.set_ylim(bounding_box[2],bounding_box[3])
        ax.set_xlim(bounding_box[0],bounding_box[1]) 
        ax.set_ylim(bounding_box[2],bounding_box[3])
        
                
        #access all calculated regions in voronoi diagram
        regions = vor.point_region
        #loop through regions to test coloring region
        for i in range(len(regions)):
            #create polygon from region to color
            region = vor.regions[vor.point_region[i]]
            polygon = [vor.vertices[i] for i in region]
            
            #determine cell color
            color = "#ffffff"
            if (i < len(source_points)):
                dif = round(source_point_pops[i]/pop_avg * 10)
                
                if (dif < 7):
                    color = "#11a841"
                elif (dif > 13):
                    color = "#f01b07"
                else:
                    color = colors[dif]
                
            #color region
            ax.fill(*zip(*polygon), alpha=0.4, data={"hello": 1}, facecolor=color)

            #test point labelling with proper index
            if (i < len(source_points)):
                ax.annotate("Pop: " + str(int(source_point_pops[i])), (upd_source_points[i, 0], upd_source_points[i, 1]))
        

        #wait some time to next iteration based on defined refresh time 
        plt.pause(refresh_time)
        
        calculated_vectors = physicsGenerateVectors(upd_source_points, source_point_pops)
        
        for i in range(len(calculated_vectors)):
            upd_source_points[i, 0] += calculated_vectors[i, 0]
            upd_source_points[i, 1] += calculated_vectors[i, 1]

            upd_source_points[i, 0] = bound(upd_source_points[i, 0], bounding_box[0], bounding_box[1])        
            upd_source_points[i, 1] = bound(upd_source_points[i, 1], bounding_box[2], bounding_box[3])
            
        it += 1 #increment iteration count
        
        #on refresh clear to redraw
        ax.clear()


def bound(val, min_bound, max_bound):
    return min(max(val, min_bound), max_bound)


def startPopulationGraph(pop_points, bounding_box):
    #create matplotlib figure
    fig2, ax2 = plt.subplots()

    #show scatter plot of points
    plt.scatter(pop_points[:, 0], pop_points[:, 1])

    #set bounds from bounding_box
    ax2.set_xlim(bounding_box[0],bounding_box[1]) 
    ax2.set_ylim(bounding_box[2],bounding_box[3])
    
    for pop_point in pop_points:
        ax2.annotate(pop_point[2], (pop_point[0], pop_point[1]))    
    
    fig2.show()
        
            
def recomputePopulations(pop_points, upd_source_points):
    #remove ghost points from calculation
    upd_source_points = upd_source_points[0:-4]
    
    
    source_point_pops = np.zeros(len(upd_source_points))
    
    for point in pop_points:
        p_x = point[0]
        p_y = point[1]
        
        min_dist = 1e12
        min_index = -1
        for i in range(0, len(upd_source_points)):
            source_x = upd_source_points[i, 0]
            source_y = upd_source_points[i, 1]
        
            #keep square for efficiency
            dist = (source_x - p_x)**2 + (source_y - p_y)**2
            if (dist < min_dist):
                min_dist = dist
                min_index = i

        source_point_pops[min_index] += point[2]
    
    return source_point_pops


def algorithmGenerateVectors(upd_source_points, source_point_pops):
    #remove ghost points from calculation
    upd_source_points = upd_source_points[0:-4]
    
    k_max = 1
    
    apply_vectors = np.array(np.zeros((len(upd_source_points), 2)))
    
    indptr_neigh, neighbours = Delaunay(upd_source_points).vertex_neighbor_vertices
    
    worst_facility = -1
    worst_pop = 0
    
    for i in range(len(source_point_pops)):
        if (source_point_pops[i] > worst_pop):
            worst_pop = source_point_pops[i]
            worst_facility = i
    
    
    #Accessing the neighbours
    i_neigh = neighbours[indptr_neigh[worst_facility]:indptr_neigh[worst_facility+1]]
    
    least_loaded_surrounding = -1
    least_surround_pop = 1e12
    
    for neighbor in i_neigh:
        if (source_point_pops[neighbor] < least_surround_pop):
            least_surround_pop = source_point_pops[neighbor]
            least_loaded_surrounding = neighbor
    
    vector = upd_source_points[worst_facility] - upd_source_points[least_loaded_surrounding]
    u_vector = vector / np.linalg.norm(vector)
    
    apply_vector = limitingFunction2(worst_facility, least_loaded_surrounding, source_point_pops) * minVector(vector / 10, u_vector * k_max)
    
    apply_vectors[least_loaded_surrounding] = apply_vector
        
    return apply_vectors

def physicsGenerateVectors(upd_source_points, source_point_pops):
    #remove ghost points from calculation
    upd_source_points = upd_source_points[0:-4]
    
    apply_vectors = np.array(np.zeros((len(upd_source_points), 2)))
    
    for i in range(len(source_point_pops)):
        curr_vector = np.zeros(2)
        for j in range(len(pop_list)):
            force = (source_point_pops[i] - pop_list[j][2])/(source_point_pops[i])
            dist = math.sqrt((upd_source_points[i][0] - pop_list[j][0] / 30.5)**2 + (upd_source_points[i][1] - pop_list[j][1] / 30.5)**2)
            force /= dist
            
            force = min(force, 2)
        
            print(force)
                        
    return apply_vectors
            

def minVector(vec1, vec2):
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    
    # Compare and return the smaller magnitude vector
    return vec1 if magnitude1 < magnitude2 else vec2

def limitingFunction1(p, q, source_point_pops):
    val = 1 - (source_point_pops[q] / source_point_pops[p])
    return val

def limitingFunction2(p, q, source_point_pops):
    val = 1 - ((pop_avg / source_point_pops[p]) if (pop_avg / source_point_pops[p] < 0.9) else 1)
    return val

    
#defines the bounding box
bounding_box = np.array([0., 30.5, 0., 30.5]) #x_min, x_max, y_min, y_max
#defines the "source points"/positions of facilities
source_points = np.array([(1, 1), (6, 10), (25, 2), (15, 23), (6, 7)], dtype='float64')
#set time between iterations
refresh_time = 0.1

#create population list (which will be transfered to np array)
pop_list = []
#compute facility "average" (optimal solution)
pop_avg = 0

#sample population list
for i in range(0, 5):
    for j in range(0, 5):
        pop_val = 45 - 5 * (i + j)
        pop_list.append([7 * i, 7 * j, pop_val])
        
        #add value to average sum
        pop_avg += pop_val

#divide by number of facilities to get true average
pop_avg /= len(source_points)

#holds the data points of population
pop_points = np.array(pop_list)


#start population graph
startPopulationGraph(pop_points, bounding_box)
#start voronoi diagram
startVoronoi(source_points, bounding_box, refresh_time)
