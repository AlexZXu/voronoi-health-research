import numpy as np #numpy for lin alg and data functions
import pandas as pd
import matplotlib.pyplot as plt #matplotlib for graphing and visual
from scipy.spatial import Voronoi, Delaunay, voronoi_plot_2d #scipy to use Voronoi helpers
import sys #system helpers
import math #math helpers
from matplotlib.widgets import Button, Slider
import csv

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

        arr = recomputeScores(pop_list, upd_source_points)
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
                ax.annotate("Score: " + str(int(source_point_pops[i])), (upd_source_points[i, 0], upd_source_points[i, 1]))


        #wait some time to next iteration based on defined refresh time
        plt.pause(refresh_time)

        calculated_vectors = clusteringGenerateVectors(upd_source_points, source_point_pops, pop_sources)

        for i in range(len(calculated_vectors)):
            upd_source_points[i] = calculated_vectors[i];

        # add_vectors = algorithmGenerateVectors(upd_source_points, source_point_pops)
        # for i in range(len(add_vectors)):
        #     print(add_vectors[i, 0], add_vectors[i, 1])
        #     upd_source_points[i, 0] += add_vectors[i, 0]
        #     upd_source_points[i, 1] += add_vectors[i, 1]

        # print(upd_source_points)

        it += 1 #increment iteration count

        #on refresh clear to redraw
        ax.clear()


def bound(val, min_bound, max_bound):
    return min(max(val, min_bound), max_bound)


def startPopulationGraph(pop_points, source_points, bounding_box):
    #create matplotlib figure
    fig2, ax2 = plt.subplots()

    #show scatter plot of points
    plt.scatter(pop_points[:, 2], pop_points[:, 3], s=10)

    plt.scatter(source_points[:, 0], source_points[:, 1], c='red')


    #set bounds from bounding_box
    ax2.set_xlim(bounding_box[0],bounding_box[1])
    ax2.set_ylim(bounding_box[2],bounding_box[3])

    for source_pt in source_points:
        src_x = source_pt[0]
        src_y = source_pt[1]
        print(src_x, src_y)

        ax2.add_patch(plt.Circle((src_x, src_y), r, color='g', fill=False, edgecolor='b', linewidth=1))
        ax2.add_patch(plt.Circle((src_x, src_y), c, color='b', fill=False, edgecolor='b', linewidth=1))

    vor = Voronoi(source_points)
    voronoi_plot_2d(vor, show_points=False, line_alpha=0.6, show_vertices=False, ax=ax2)

    fig2.show()


def recomputeScores(pop_points, upd_source_points):
    global pop_avg
    #remove ghost points from calculation
    upd_source_points = upd_source_points[0:-4]

    #stores the scores of each source point
    source_point_scores = np.array(np.zeros(len(upd_source_points)))
    #stores the nearest source point (cell id) for each population point
    pop_sources = np.array(np.zeros(len(pop_points)), dtype=np.int16)

    pop_reg = np.array(np.zeros(len(upd_source_points)))
    pop_outside = np.array(np.zeros(len(upd_source_points)))
    pop_exclude = np.array(np.zeros(len(upd_source_points)))

    #code to count normal population
    for i in range(0, len(pop_points)):
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

        source_point_scores[min_index] += pop_points[i, 1]
        pop_sources[i] = min_index

        pop_reg[min_index] += pop_points[i, 1]
        #if out of range, the population holds double the weight
        if (min_dist > r*r):
            source_point_scores[min_index] += pop_points[i, 1]
            pop_outside[min_index] += pop_points[i, 1]


    #code to handle fuzziness, weighted omega
    for i in range(0, len(pop_points)):
        p_x = pop_points[i, 2]
        p_y = pop_points[i, 3]

        for j in range(0, len(upd_source_points)):
            source_x = upd_source_points[j, 0]
            source_y = upd_source_points[j, 1]

            dist = (source_x - p_x)**2 + (source_y - p_y)**2
            if (dist <= c*c and pop_sources[i] != j):
                source_point_scores[j] += omega * pop_points[i, 1]
                pop_exclude[j] += omega * pop_points[i, 1]

    for el in source_point_scores:
        pop_avg += el


    for i in range(0, len(source_point_scores)):
        print(source_names[i], source_point_scores[i], pop_reg[i], pop_outside[i], pop_exclude[i])

    #divide by number of facilities to get true average
    pop_avg /= len(source_points)




    return [source_point_scores, pop_sources]


def clusteringGenerateVectors(upd_source_points, source_point_pops, pop_sources):
    #remove ghost points from calculation
    upd_source_points = upd_source_points[0:-4]

    means = np.array(np.zeros((len(upd_source_points), 2)))

    weights = np.array(np.zeros(len(upd_source_points)))
    for i in range(len(pop_list)):
        x = pop_list[i, 2]
        y = pop_list[i, 3]
        pop = pop_list[i, 1]
        source = pop_sources[i]

        means[source][0] += x * pop
        means[source][1] += y * pop

        weights[source] += pop

        if ((x - upd_source_points[source, 0])**2 + (y - upd_source_points[source, 1])**2 > r*r):
            means[source][0] += x * pop
            means[source][1] += y * pop

            weights[source] += pop


    #code to handle fuzziness, weighted omega
    for i in range(0, len(pop_list)):
        p_x = pop_list[i, 2]
        p_y = pop_list[i, 3]

        for j in range(0, len(upd_source_points)):
            source_x = upd_source_points[j, 0]
            source_y = upd_source_points[j, 1]

            dist = (source_x - p_x)**2 + (source_y - p_y)**2
            if (dist <= c*c and pop_sources[i] != j):
                means[j][0] += omega * pop_list[i, 1] * p_x
                means[j][1] += omega * pop_list[i, 1] * p_y
                weights[j] += omega * pop_list[i, 1]

    for i in range(len(means)):
        if (weights[i] == 0):
            means[i][0] = pop_list[i, 2]
            means[i][1] = pop_list[i, 3]
        else:
            means[i][0] /= weights[i]
            means[i][1] /= weights[i]

    return means


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

    apply_vector = limitingFunction2(worst_facility, least_loaded_surrounding, source_point_pops) * (vector)/5

    apply_vectors[least_loaded_surrounding] = apply_vector

    return apply_vectors


# defines the bounding box
bounding_box = np.array([33.6, 34., -84.6, -84.2]) #x_min, x_max, y_min, y_max

# specific file path may be different
df = pd.read_csv('C:\\Alex Xu Files\\ghpmath\\research\\data_sets\\atlanta-zip-coords.csv')
ar = df.to_numpy(dtype="float64")

source_arr = []
source_names = []
cnt = 0

# specific file path may be different
with open('C:\\Alex Xu Files\\ghpmath\\research\\data_sets\\atlanta-health-clinics.csv', mode='r') as file:
    csvFile = csv.reader(file)
    next(csvFile, None)
    for lines in csvFile:
        source_arr.append([lines[1], lines[2]])
        source_names.append(lines[0])
        cnt += 1

#defines the "source points"/positions of facilities
source_points = np.array(source_arr, dtype='float64')

r = 0.05
c = 0.05
omega = 0.2

#set time between iterations
refresh_time = 1

#create population list (which will be transfered to np array)
pop_list = ar
print(pop_list)

#compute facility "average"
pop_avg = 0


#start population graph
startPopulationGraph(pop_list, source_points, bounding_box)
#start voronoi diagram
startVoronoi(source_points, bounding_box, refresh_time)
