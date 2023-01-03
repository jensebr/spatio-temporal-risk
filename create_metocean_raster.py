from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj
import requests
import xmltodict

def pixel2coord(x, y):
    xoff, a, b, yoff, d, e = raster.GetGeoTransform()

    xp = a * x + b * y + a * 0.5 + b * 0.5 + xoff
    yp = d * x + e * y + d * 0.5 + e * 0.5 + yoff
    return(xp, yp)

def create_raster():

    # new raster filename
    current_dir_filename = "/Users/jensbremnes/Documents/GIS_files/current_direction.tif"
    current_speed_filename = "/Users/jensbremnes/Documents/GIS_files/current_speed.tif"
    wave_dir_filename = "/Users/jensbremnes/Documents/GIS_files/wave_direction.tif"
    wave_height_filename = "/Users/jensbremnes/Documents/GIS_files/wave_height.tif"

    # existing raster filename
    input_filename = "/Users/jensbremnes/Documents/GIS_files/depth.tif"

    # open existing raster
    input_dataset = gdal.Open(input_filename)

    # Get geotransform and projection of an existing raster
    geotransform = input_dataset.GetGeoTransform()
    projection = input_dataset.GetProjection()

    # Get a gdal driver to use for raster creation
    driver_tiff = gdal.GetDriverByName("GTiff")

    # Create new raster dataset
    current_dir_dataset = driver_tiff.Create(current_dir_filename, xsize=input_dataset.RasterXSize, ysize=input_dataset.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    current_speed_dataset = driver_tiff.Create(current_speed_filename, xsize=input_dataset.RasterXSize, ysize=input_dataset.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    wave_dir_dataset = driver_tiff.Create(wave_dir_filename, xsize=input_dataset.RasterXSize, ysize=input_dataset.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    wave_height_dataset = driver_tiff.Create(wave_height_filename, xsize=input_dataset.RasterXSize, ysize=input_dataset.RasterYSize, bands=1, eType=gdal.GDT_Float32)

    # set the geotransform
    current_dir_dataset.SetGeoTransform(geotransform)
    current_speed_dataset.SetGeoTransform(geotransform)
    wave_dir_dataset.SetGeoTransform(geotransform)
    wave_height_dataset.SetGeoTransform(geotransform)

    # set the projection
    current_dir_dataset.SetProjection(projection)
    current_speed_dataset.SetProjection(projection)
    wave_dir_dataset.SetProjection(projection)
    wave_height_dataset.SetProjection(projection)

    current_dir_band = current_dir_dataset.GetRasterBand(1)
    current_dir_array = current_dir_band.ReadAsArray()
    current_speed_band = current_speed_dataset.GetRasterBand(1)
    current_speed_array = current_speed_band.ReadAsArray()
    wave_dir_band = wave_dir_dataset.GetRasterBand(1)
    wave_dir_array = wave_dir_band.ReadAsArray()
    wave_height_band = wave_height_dataset.GetRasterBand(1)
    wave_height_array = wave_height_band.ReadAsArray()

    print("Starting to fill in data")

    (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = current_dir_dataset.GetGeoTransform()
    shape = np.shape(current_dir_array)
    rows = shape[0]
    cols = shape[1]
    counter = 0

    myProj = Proj(proj='utm',zone=32,ellps='WGS84', preserve_units=False) # remember to change zone!!!

    col_counter = 0
    fetch_data_row = False
    fetch_data_col = False
    currentSpeed = 0
    currentDirection = 0
    waveHeight = 0
    waveDirection = 0
    for col in range(cols):

        if (col_counter % 50 == 0):
            fetch_data_col = True
        else:
            fetch_data_col = False
        col_counter += 1
        row_counter = 0

        for row in range(rows):

            if (row_counter % 50 == 0):
                fetch_data_row = True
            else:
                fetch_data_row = False
            row_counter += 1

            # assuming that x is west-east and y is north-south
            x_index = col
            y_index = row
            x_coord = x_size*x_index + x_rotation*y_index + x_size*0.5 + x_rotation*0.5 + upper_left_x
            y_coord = y_rotation*x_index + y_size*y_index + y_rotation*0.5 + y_size*0.5 + upper_left_y

            if fetch_data_row == True and fetch_data_col == True:

                lon, lat = myProj(x_coord, y_coord, inverse=True)
                url = "https://api.met.no/weatherapi/oceanforecast/2.0/mox?lat=" + str(lat) + "&lon=" + str(lon)
                headers = {'User-Agent': 'jens.e.bremnes@ntnu.no'}

                try:
                    raw_response = requests.get(url, headers=headers)  # using requests to get webcontent
                    text = raw_response.text
                    dict = xmltodict.parse(text)
                except ConnectionRefusedError:
                    print("Connection refused")
                    return

                try:
                    x = dict['mox:Forecasts']['mox:forecast']
                    depth = x[0]['metno:OceanForecast']['mox:seaBottomTopography']['#text']
                    seaIcePresence = x[0]['metno:OceanForecast']['mox:seaIcePresence']['#text']
                    waveDirection = x[0]['metno:OceanForecast']['mox:meanTotalWaveDirection']['#text']
                    waveHeight = x[0]['metno:OceanForecast']['mox:significantTotalWaveHeight']['#text']
                    currentDirection = x[0]['metno:OceanForecast']['mox:seaCurrentDirection']['#text']
                    currentSpeed = x[0]['metno:OceanForecast']['mox:seaCurrentSpeed']['#text']
                    currentTemperature = x[0]['metno:OceanForecast']['mox:seaTemperature']['#text']

                except KeyError:
                    currentSpeed = -1
                    currentDirection = -1
                    waveHeight = -1
                    waveDirection = -1

            elif fetch_data_row == False:
                currentDirection = current_dir_array[row-1][col]
                currentSpeed = current_speed_array[row-1][col]
                waveDirection = wave_dir_array[row-1][col]
                waveHeight = wave_height_array[row-1][col]
            elif fetch_data_col == False:
                currentDirection = current_dir_array[row][col-1]
                currentSpeed = current_speed_array[row][col-1]
                waveDirection = wave_dir_array[row][col-1]
                waveHeight = wave_height_array[row][col-1]

            current_dir_array[row][col] = currentDirection
            current_speed_array[row][col] = currentSpeed
            wave_dir_array[row][col] = waveDirection
            wave_height_array[row][col] = waveHeight

    print("Finished")

    # Write data
    current_dir_band.WriteArray(current_dir_array, 0, 0)
    current_speed_band.WriteArray(current_speed_array, 0, 0)
    wave_dir_band.WriteArray(wave_dir_array, 0, 0)
    wave_height_band.WriteArray(wave_height_array, 0, 0)

    # flush data to disk, set the NoData value and calculate stats
    current_dir_band.FlushCache()
    current_dir_band.SetNoDataValue(-99)
    current_speed_band.FlushCache()
    current_speed_band.SetNoDataValue(-99)
    wave_dir_band.FlushCache()
    wave_dir_band.SetNoDataValue(-99)
    wave_height_band.FlushCache()
    wave_height_band.SetNoDataValue(-99)

    del(current_dir_dataset)
    del(current_speed_dataset)
    del(wave_dir_dataset)
    del(wave_height_dataset)

if __name__ == '__main__':
    create_raster()
