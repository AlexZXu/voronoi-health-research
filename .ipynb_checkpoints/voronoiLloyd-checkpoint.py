import numpy as np #numpy for lin alg and data functions
import pandas as pd
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
        
        arr = recomputePopulations(pop_list, upd_source_points)
        source_point_pops = arr[0]
        pop_sources = arr[1]
        
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
        
        calculated_vectors = clusteringGenerateVectors(upd_source_points, source_point_pops, pop_sources)
        
        for i in range(len(calculated_vectors)):
            upd_source_points[i] = calculated_vectors[i];

            
        it += 1 #increment iteration count
        
        #on refresh clear to redraw
        ax.clear()


def bound(val, min_bound, max_bound):
    return min(max(val, min_bound), max_bound)


def startPopulationGraph(pop_points, bounding_box):
    #create matplotlib figure
    fig2, ax2 = plt.subplots()

    #show scatter plot of points
    plt.scatter(pop_points[:, 2], pop_points[:, 3])

    #set bounds from bounding_box
    ax2.set_xlim(bounding_box[0],bounding_box[1]) 
    ax2.set_ylim(bounding_box[2],bounding_box[3])
    
    for pop_point in pop_points:
        ax2.annotate(pop_point[1], (pop_point[2], pop_point[3]))    
    
    fig2.show()
        
            
def recomputePopulations(pop_points, upd_source_points):
    #remove ghost points from calculation
    upd_source_points = upd_source_points[0:-4]
    
    source_point_pops = np.array(np.zeros(len(upd_source_points)))
    pop_sources = np.array(np.zeros(len(pop_points)), dtype=np.int16)
    
    #loop through all population points
    for i in range(0, len(pop_points)):
        #get x and y of specific point
        p_x = pop_points[i, 2]
        p_y = pop_points[i, 3]
        
        min_dist = 1e12
        min_index = -1
        for j in range(0, len(upd_source_points)):
            source_x = upd_source_points[j, 0]
            source_y = upd_source_points[j, 1]
        
            #keep square for efficiency
            dist = (source_x - p_x)**2 + (source_y - p_y)**2
            if (dist < min_dist):
                min_dist = dist
                min_index = j

        source_point_pops[min_index] += pop_points[i, 2]
        pop_sources[i] = min_index
        
    return [source_point_pops, pop_sources]



def clusteringGenerateVectors(upd_source_points, source_point_pops, pop_sources):
    #remove ghost points from calculation
    upd_source_points = upd_source_points[0:-4]
    
    means = np.array(np.zeros((len(upd_source_points), 2)))
    
    #calculate weighted mean based solely on population
    
    for i in range(len(pop_list)):
        x = pop_list[i, 0]
        y = pop_list[i, 1]
        pop = pop_list[i, 2]
        source = pop_sources[i]
                    
        means[source][0] += x * pop
        means[source][1] += y * pop
        
        
    for i in range(len(means)):
        if (source_point_pops[i] == 0):
            means[i][0] = pop_list[i, 0]
            means[i][1] = pop_list[i, 1]
        else:
            means[i][0] /= source_point_pops[i]
            means[i][1] /= source_point_pops[i]
    
    return means
    

    
#defines the bounding box
bounding_box = np.array([33.6, 34., -84.6, -84.2]) #x_min, x_max, y_min, y_max

df = pd.read_csv('C:\\Alex Xu Files\\ghpmath\\research\\atlanta-zip-coords.csv')
ar = df.to_numpy(dtype="float64")

#defines the "source points"/positions of facilities
source_points = np.array([(0, 0)], dtype='float64')


#set time between iterations
refresh_time = 1

#create population list (which will be transfered to np array)
pop_list = ar

#compute facility "average" 
pop_avg = 0

for el in pop_list:
    pop_avg += el[1]

#divide by number of facilities to get true average
pop_avg /= len(source_points)

#start population graph
startPopulationGraph(pop_list, bounding_box)
#start voronoi diagram
startVoronoi(source_points, bounding_box, refresh_time)
