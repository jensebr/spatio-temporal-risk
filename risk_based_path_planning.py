from osgeo import osr, gdal
from skimage.graph import route_through_array
import numpy as np
import itertools
from math import exp
from pyproj import Proj
from scipy.io import savemat

import time

def raster2array(rasterfn):
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(1)
    array = band.ReadAsArray()
    #new_array = [element**1.5 for element in array]
    #new_array = [[element**1.5 for element in row] for row in array]

    new_array = array
    shape = np.shape(new_array)
    rows = shape[0]
    cols = shape[1]
    for col in range(cols):
        for row in range(rows):
            if new_array[row][col] > 0:
                #new_array[row][col] = (0.01*new_array[row][col])**4
                #new_array[row][col] = 1.0 # shortest path
                #new_array[row][col] = exp(0.1*new_array[row][col]) #This is the results for ripp with moderate altitude

                if new_array[row][col] > 1000.0:
                    new_array[row][col] = 1000000000000000000000000000000000000000000000000
                else:
                    #new_array[row][col] = exp(0.0002*new_array[row][col]) #This is the results for ripp with close altitude (did I also include something else??)
                    #new_array[row][col] = exp(0.01*new_array[row][col]) #This is the results for ripp with close altitude (did I also include something else??)
                    #new_array[row][col] = exp(0.005*new_array[row][col]) #This is the results for ripp with close altitude (did I also include something else??)
                    #new_array[row][col] = exp(0.005*new_array[row][col]) #THIS ONE WORKS FOR SP, MODERATE ALTITUDE!!!!!!!
                    new_array[row][col] = exp(0.005*new_array[row][col]) #THIS ONE WORKS FOR SP, MODERATE ALTITUDE!!!!!!!

                #new_array[row][col] = new_array[row][col]

    return new_array

def coord2pixelOffset(rasterfn,x,y):
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    xOffset = int((x - originX)/pixelWidth)
    yOffset = int((y - originY)/pixelHeight)
    return xOffset,yOffset

def solveTsp(CostSurfacefn, costSurfaceArray):

    neighborhoods = [1, 2, 3, 4, 5, 6, 7]

    wp_list = [[576200, 7051762],
    [576095, 7052734],
    [575689, 7053076],
    [575231, 7052865],
    [575199, 7053332],
    [574838, 7053655],
    [575068, 7053873]]

    #neighborhoods = [1, 2, 3]
    #wp_list = [[576200, 7051762],
    #[576095, 7052734],
    #[575689, 7053076]]

    #total_path = np.zeros_like(costSurfaceArray)

    optimal_path = np.zeros_like(costSurfaceArray)
    optimal_risk = 10000000000000000000000000000000000000000

    counter = 0
    counter_2 = 0

    # To do: Remove symmetric
    # To do: Use some heuristics to skip some solutions with "if that, then continue"

    # For each possible solution
    all_permutations = itertools.permutations(neighborhoods)
    i = 0
    for subset in all_permutations:

        counter_2 += 1
        if counter_2 > 2520:
            break

        continue_flag = False

        for i in range(len(subset) - 1):

            j = subset[i] - 1
            k = subset[i+1] - 1

            if abs(k-j) > 4:
                continue_flag = True
                break

        if continue_flag:
            continue

        print(subset)

        total_path = np.zeros_like(costSurfaceArray)
        total_risk = 0

        counter = counter+1
        print(counter)

        # for each neighborhood waypoint
        for i in range(len(subset) - 1):

            j = subset[i] - 1
            k = subset[i+1] - 1

            startCoord = wp_list[j]
            stopCoord = wp_list[k]

            startCoord = (startCoord[0], startCoord[1])
            stopCoord = (stopCoord[0], stopCoord[1])

            subpath, risk, indices = createPath(CostSurfacefn, costSurfaceArray, startCoord, stopCoord)
            total_risk += risk
            total_path = np.maximum(total_path, subpath)
            #print(np.max(total_path))

        if total_risk < optimal_risk:
            optimal_path = total_path
            optimal_risk = total_risk
            optimal_subset = subset

    print(optimal_subset)

    return optimal_path

def solveTsp_simplified(CostSurfacefn, costSurfaceArray):

    neighborhoods = [1, 2, 3, 4, 5, 6, 7]

    wp_list = [[576200, 7051762],
    [576095, 7052734],
    [575689, 7053076],
    [575231, 7052865],
    [575199, 7053332],
    [574838, 7053655],
    [575068, 7053873]]

    # removed 2 points (exceeded 1000)
    wp_list = [[576200, 7051762],
    [576095, 7052734],
    [575689, 7053076],
    [575199, 7053332],
    [575068, 7053873]]

    #neighborhoods = [1, 2, 3]
    #wp_list = [[576200, 7051762],
    #[576095, 7052734],
    #[575689, 7053076]]

    #total_path = np.zeros_like(costSurfaceArray)

    subset = (1,2,3,4,5,6,7)

    # removed 2 points
    subset = (1,2,3,4,5)

    total_path = np.zeros_like(costSurfaceArray)
    total_risk = 0

    init = True

    # for each neighborhood waypoint
    for i in range(len(subset) - 1):

        j = subset[i] - 1
        k = subset[i+1] - 1

        startCoord = wp_list[j]
        stopCoord = wp_list[k]

        startCoord = (startCoord[0], startCoord[1])
        stopCoord = (stopCoord[0], stopCoord[1])

        subpath, risk, indices = createPath(CostSurfacefn, costSurfaceArray, startCoord, stopCoord)
        total_risk += risk
        total_path = np.maximum(total_path, subpath)
        if init:
            total_indices = indices
            init = False
        else:
            total_indices = np.concatenate((total_indices, indices), axis=0)
        #print(np.max(total_path))

    print(total_indices)

    return total_path, total_indices

def solveTsp_simplified_v2(CostSurfacefn, costSurfaceArray):

    raster_ds = gdal.Open("/Users/jensbremnes/Documents/GIS_files/current_direction.tif")

    neighborhoods = [1, 2, 3, 4, 5, 6, 7]

    wp_list = [[576200, 7051762],
    [576095, 7052734],
    [575689, 7053076],
    [575231, 7052865],
    [575199, 7053332],
    [574838, 7053655],
    [575068, 7053873]]

    subset = (1,2,3,4,5,6,7)

    total_path = np.zeros_like(costSurfaceArray)
    total_risk = 0

    # for each neighborhood waypoint
    for i in range(len(subset) - 1):

        j = subset[i] - 1
        k = subset[i+1] - 1

        startCoord = wp_list[j]
        stopCoord = wp_list[k]

        startCoord = (startCoord[0], startCoord[1])
        stopCoord = (stopCoord[0], stopCoord[1])

        subpath, risk, indices = createPath(CostSurfacefn, costSurfaceArray, startCoord, stopCoord)
        total_risk += risk
        total_path = np.maximum(total_path, subpath)
        #print(np.max(total_path))

    return total_path

def createPath(CostSurfacefn,costSurfaceArray,startCoord,stopCoord):

    # coordinates to array index
    startCoordX = startCoord[0]
    startCoordY = startCoord[1]
    startIndexX,startIndexY = coord2pixelOffset(CostSurfacefn,startCoordX,startCoordY)

    stopCoordX = stopCoord[0]
    stopCoordY = stopCoord[1]
    stopIndexX,stopIndexY = coord2pixelOffset(CostSurfacefn,stopCoordX,stopCoordY)

    # create path
    indices, weight = route_through_array(costSurfaceArray, (startIndexY,startIndexX), (stopIndexY,stopIndexX),geometric=True,fully_connected=True)

    temp_indices = indices
    indices = np.array(indices).T

    #print(indices)

    path = np.zeros_like(costSurfaceArray)
    path[indices[0], indices[1]] = 1
    return path, weight, temp_indices

def calculate_length(array):

    cols = array.shape[1]
    rows = array.shape[0]

    for row in range(rows):
        for col in range(cols):
            if array[row][col] > 0.0:
                start_row = row
                start_col = col

    prev_row, prev_col = None, None
    distance = 0.0

    finished = False

    row = start_row
    col = start_col
    break_out_flag = False
    while not finished:

        found_next_point = False
        for d_row in range(-1, 2):
            for d_col in range(-1, 2):

                if ((d_row, d_col) == (0, 0)) or ((row + d_row, col + d_col) == (prev_row, prev_col)):
                    continue

                # If next point is not zero, and you are not evaluating the current spot, and you are not evaluating the previous spot
                if array[row + d_row][col + d_col] > 0:

                    prev_row = row
                    prev_col = col
                    
                    row += d_row
                    col += d_col

                    # find distance
                    if (d_col == 0) or (d_row == 0):
                        distance += 1
                    else:
                        distance += np.sqrt(2)

                 
                    break_out_flag = True
                    break
            
            if break_out_flag:
                break

        # If I end up here, then I did not find any more points
        if break_out_flag:
            break_out_flag = False
        else:
            finished = True
    
    print(distance)
    print(sum_array(array))

def sum_array(array):
 
    sum = 0
    cols = array.shape[1]
    rows = array.shape[0]
 
    # Finding the sum
    for row in range(rows):
        for col in range(cols):
           
            # Add the element
            sum = sum + array[row][col]
 
    return sum

def array2raster(newRasterfn,rasterfn,array):
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = array.shape[1]
    rows = array.shape[0]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Byte)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    outband.SetNoDataValue(0)

def find_indices_with_altitude(total_indices):

    # open raster layers
    #ds = gdal.Open("/Users/jensbremnes/Documents/GIS_files/altitude_path_0812_mild_cond.tif")
    ds = gdal.Open("/Users/jensbremnes/Documents/GIS_files/altitude_path_0812_mod_cond.tif")
    array = ds.GetRasterBand(1).ReadAsArray()
    zero_column = np.zeros((len(total_indices), 1))

    print(np.shape(total_indices))
    print(np.shape(zero_column))

    indices_with_altitude = np.concatenate((total_indices, zero_column), axis=1)

    (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = ds.GetGeoTransform()

    myProj = Proj(proj='utm',zone=32,ellps='WGS84', preserve_units=False)

    for i, wp in enumerate(total_indices):

        x_index = wp[1] # col
        y_index = wp[0] # row
        x_coord = x_size*x_index + x_rotation*y_index + x_size*0.5 + x_rotation*0.5 + upper_left_x
        y_coord = y_rotation*x_index + y_size*y_index + y_rotation*0.5 + y_size*0.5 + upper_left_y

        lon_deg, lat_deg = myProj(x_coord, y_coord, inverse=True)
        #lon_deg = lon_rad * 180/np.pi
        #lat_deg = lat_rad * 180/np.pi

        if array[wp[0]][wp[1]] == 1:
            altitude = 5
        elif array[wp[0]][wp[1]] == 2:
            altitude = 8
        else:
            altitude = 12

        indices_with_altitude[i][0] = lat_deg
        indices_with_altitude[i][1] = lon_deg
        indices_with_altitude[i][2] = altitude

    path = np.delete(indices_with_altitude, np.where((indices_with_altitude < -1))[0], axis=0)

    #kernel_size = 3
    #kernel = np.ones(kernel_size) / kernel_size
    #path[:, 2] = np.convolve(path[:, 2], kernel, mode='same')

    #savemat('path.mat', {'path': path})
    np.set_printoptions(suppress=True)
    np.savetxt('tsp_path_moderate_conditions.txt', path, fmt='%f')


    print(indices_with_altitude)

def write_shortest_path_to_text(total_indices):

    # open raster layers
    #ds = gdal.Open("/Users/jensbremnes/Documents/GIS_files/Path_0612_no_reward_mild_cond_close_alt_final.tif")
    #ds = gdal.Open('/Users/jensbremnes/Documents/GIS_files/Path_0612_no_reward_moderate_cond_close_alt_final.tif')
    #ds = gdal.Open('/Users/jensbremnes/Documents/GIS_files/Path_0812_no_reward_mild_cond_close_alt.tif')
    #ds = gdal.Open('/Users/jensbremnes/Documents/GIS_files/Path_0812_no_reward_moderate_cond_close_alt.tif')

    ds = gdal.Open('/Users/jensbremnes/Documents/GIS_files/Path_0812_no_reward_mild_cond_mod_alt_final.tif')

    array = ds.GetRasterBand(1).ReadAsArray()
    zero_column = np.zeros((len(total_indices), 1))

    print(np.shape(total_indices))
    print(np.shape(zero_column))

    indices_with_altitude = np.concatenate((total_indices, zero_column), axis=1)

    (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = ds.GetGeoTransform()

    myProj = Proj(proj='utm',zone=32,ellps='WGS84', preserve_units=False)

    for i, wp in enumerate(total_indices):

        x_index = wp[1] # col
        y_index = wp[0] # row
        x_coord = x_size*x_index + x_rotation*y_index + x_size*0.5 + x_rotation*0.5 + upper_left_x
        y_coord = y_rotation*x_index + y_size*y_index + y_rotation*0.5 + y_size*0.5 + upper_left_y

        lon_deg, lat_deg = myProj(x_coord, y_coord, inverse=True)
        #lon_deg = lon_rad * 180/np.pi
        #lat_deg = lat_rad * 180/np.pi


        indices_with_altitude[i][0] = lat_deg
        indices_with_altitude[i][1] = lon_deg
        indices_with_altitude[i][2] = 8.0

    path = np.delete(indices_with_altitude, np.where((indices_with_altitude < -1))[0], axis=0)

    #savemat('path.mat', {'path': path})
    np.set_printoptions(suppress=True)
    np.savetxt('ripp_moderate_conditions_6m_0812.txt', path, fmt='%f')


    print(indices_with_altitude)

def main(CostSurfacefn,outputPathfn,startCoord,stopCoord):

    costSurfaceArray = raster2array(CostSurfacefn) # creates array from cost surface raster

    start_time = time.time()

    # tsp
    pathArray, total_indices = solveTsp_simplified(CostSurfacefn, costSurfaceArray) # creates path array
    #pathArray = solveTsp(CostSurfacefn, costSurfaceArray) # creates path array
    indices_with_altitude = find_indices_with_altitude(total_indices)

    # shortest path
    #pathArray, risk, total_indices = createPath(CostSurfacefn, costSurfaceArray, startCoord, stopCoord)
    #temp = write_shortest_path_to_text(total_indices)


    array2raster(outputPathfn,CostSurfacefn,pathArray) # converts path array to raster

    #calculate_length(pathArray)

    print("--- %s seconds ---" % (time.time() - start_time))


#if __name__ == "__main__":
print("Hello world")

#startCoord = (576200, 7051762)
#stopCoord = (575068, 7053873)
startCoord = (576118,7051393)
stopCoord = (574461, 7053896)

#CostSurfacefn = '/Users/jensbremnes/Documents/GIS_files/optimal_risk_25_11.tif'

# 2x shortest path
#outputPathfn = '/Users/jensbremnes/Documents/GIS_files/obsolete_mod.tif'
#CostSurfacefn = '/Users/jensbremnes/Documents/GIS_files/risk_map_25_11_Mild_Moderate_altitude_no_reward.tif'
#outputPathfn = '/Users/jensbremnes/Documents/GIS_files/Path_2511_no_reward_mild.tif'
#main(CostSurfacefn,outputPathfn,startCoord,stopCoord)
#CostSurfacefn = '/Users/jensbremnes/Documents/GIS_files/risk_map_25_11_Moderate_Moderate_altitude_no_reward.tif'
#outputPathfn = '/Users/jensbremnes/Documents/GIS_files/Path_2511_no_reward_moderate.tif'
#main(CostSurfacefn,outputPathfn,startCoord,stopCoord)

# 2x shortest path with close altitude!
# cost function is: exp(0.01*new_array[row][col])
#CostSurfacefn = '/Users/jensbremnes/Documents/GIS_files/risk_map_30_11_Mild_Close_altitude_no_reward.tif'
#outputPathfn = '/Users/jensbremnes/Documents/GIS_files/Path_0612_no_reward_mild_cond_close_alt_final.tif'
#main(CostSurfacefn,outputPathfn,startCoord,stopCoord)
#CostSurfacefn = '/Users/jensbremnes/Documents/GIS_files/risk_map_30_11_Moderate_Close_altitude_no_reward.tif'
#outputPathfn = '/Users/jensbremnes/Documents/GIS_files/Path_0612_no_reward_moderate_cond_close_alt_final.tif'
#main(CostSurfacefn,outputPathfn,startCoord,stopCoord)
#CostSurfacefn = '/Users/jensbremnes/Documents/GIS_files/risk_map_30_11_Moderate_Close_altitude_no_reward.tif'
#outputPathfn = '/Users/jensbremnes/Documents/GIS_files/Path_0612_shortest_path_final_testtime.tif'
#start_time = time.time()
#main(CostSurfacefn,outputPathfn,startCoord,stopCoord)
#print("--- %s seconds ---" % (time.time() - start_time))
# lengths are: 362.3919189857876 (mod) and 332.4751801064729 (mild) * 10 meter and 

# 4x shortest path with mod/close altitude!
# cost function is: exp(0.01*new_array[row][col]) or 0.005
# DEEZ
#CostSurfacefn = '/Users/jensbremnes/Documents/GIS_files/h1_08_12_Mild_Moderate_altitude_no_reward.tif' # mild conditions, moderate altitude
#outputPathfn = '/Users/jensbremnes/Documents/GIS_files/Path_0812_no_reward_mild_cond_mod_alt_final.tif'
#main(CostSurfacefn,outputPathfn,startCoord,stopCoord)
#CostSurfacefn = '/Users/jensbremnes/Documents/GIS_files/h1_08_12_Moderate_Moderate_altitude_no_reward.tif' # moderate conditions, moderate altitude
#outputPathfn = '/Users/jensbremnes/Documents/GIS_files/Path_0812_no_reward_moderate_cond_mod_alt_tuning.tif'
#main(CostSurfacefn,outputPathfn,startCoord,stopCoord)
# DEEZ
#CostSurfacefn = '/Users/jensbremnes/Documents/GIS_files/h1_08_12_Mild_Close_altitude_no_reward.tif' # mild conditions, close altitude
#outputPathfn = '/Users/jensbremnes/Documents/GIS_files/Path_0812_no_reward_mild_cond_close_alt.tif'
#main(CostSurfacefn,outputPathfn,startCoord,stopCoord)
#CostSurfacefn = '/Users/jensbremnes/Documents/GIS_files/h1_08_12_Moderate_Close_altitude_no_reward.tif' # moderate conditions, close altitude
#outputPathfn = '/Users/jensbremnes/Documents/GIS_files/Path_0812_no_reward_moderate_cond_close_alt.tif'
#main(CostSurfacefn,outputPathfn,startCoord,stopCoord)

# 2x TSP
#CostSurfacefn = '/Users/jensbremnes/Documents/GIS_files/optimal_risk_02_12_mild_cond.tif'
#outputPathfn = '/Users/jensbremnes/Documents/GIS_files/Path_02_12_mild_cond_tsp_time.tif'
#start_time = time.time()
#main(CostSurfacefn,outputPathfn,startCoord,stopCoord)
#print("--- %s seconds ---" % (time.time() - start_time))
#CostSurfacefn = '/Users/jensbremnes/Documents/GIS_files/optimal_risk_02_12_mod_cond.tif'
#outputPathfn = '/Users/jensbremnes/Documents/GIS_files/Path_02_12_mod_cond_tsp_constrained.tif'
#main(CostSurfacefn,outputPathfn,startCoord,stopCoord)

# DEEZ
#CostSurfacefn = '/Users/jensbremnes/Documents/GIS_files/optimal_risk_08_12_mild_cond.tif'
#outputPathfn = '/Users/jensbremnes/Documents/GIS_files/Path_08_12_mild_cond_tsp.tif'
#main(CostSurfacefn,outputPathfn,startCoord,stopCoord)
CostSurfacefn = '/Users/jensbremnes/Documents/GIS_files/optimal_risk_08_12_mod_cond.tif'
outputPathfn = '/Users/jensbremnes/Documents/GIS_files/Path_08_12_mod_cond_tsp.tif'
main(CostSurfacefn,outputPathfn,startCoord,stopCoord)
# DEEZ

# 2x shortest path with cvar
#CostSurfacefn = '/Users/jensbremnes/Documents/GIS_files/risk_map_06_12_Severe_Close_altitude_cvar.tif'
#outputPathfn = '/Users/jensbremnes/Documents/GIS_files/Path_0612_severe_close_cvar_no_reward.tif'
#main(CostSurfacefn,outputPathfn,startCoord,stopCoord)
#CostSurfacefn = '/Users/jensbremnes/Documents/GIS_files/risk_map_06_12_Mild_Close_altitude_cvar.tif'
#outputPathfn = '/Users/jensbremnes/Documents/GIS_files/Path_0612_mild_close_cvar_no_reward.tif'
#main(CostSurfacefn,outputPathfn,startCoord,stopCoord)