from osgeo import gdal
import numpy as np
from pyproj import Proj
import pysmile
from shapely.geometry import Point, Polygon, LineString
from scipy.spatial import distance
from datetime import date
import math

pysmile.License((
	b"SMILE LICENSE a8b85441 f70f29ff a5924ead "
	b"THIS IS AN ACADEMIC LICENSE AND CAN BE USED "
	b"SOLELY FOR ACADEMIC RESEARCH AND TEACHING, "
	b"AS DEFINED IN THE BAYESFUSION ACADEMIC "
	b"SOFTWARE LICENSING AGREEMENT. "
	b"Serial #: 75cvryzqvul2c69mim85ap4nm "
	b"Issued for: Jens Einar Bremnes (jensbremnes@gmail.com) "
	b"Academic institution: Norwegian University of Science and Technology "
	b"Valid until: 2023-03-03 "
	b"Issued by BayesFusion activation server"
	),[
	0xe7,0x13,0xe5,0xeb,0x50,0x38,0x5e,0xcf,0x05,0xda,0xaa,0x8c,0x1a,0xe5,0x9a,0x73,
	0x7f,0x00,0x71,0x73,0x4c,0x5b,0x29,0x90,0x8e,0xfd,0xc2,0x16,0x00,0x5c,0x83,0x67,
	0xbe,0x41,0x10,0xf1,0x08,0xd8,0x93,0xb6,0xee,0xee,0xd0,0x28,0x3c,0x28,0x82,0x53,
	0x8c,0x74,0xe6,0xaa,0x50,0x78,0x9d,0x73,0x0f,0x17,0x22,0x8b,0xd2,0x37,0xfa,0xed])

def create_risk_raster(altitude_ev, case):

    today = date.today()
    today_string = today.strftime("%d_%m_")

    name_postfix = today_string + case + "_" + altitude_ev + "_altitude.tif"

    # open raster layers
    current_speed_ds = gdal.Open("/Users/jensbremnes/Documents/GIS_files/current_filtered.tif")
    slope_ds = gdal.Open("/Users/jensbremnes/Documents/GIS_files/slope_filtered_s10_r20.tif")
    ruggedness_ds = gdal.Open("/Users/jensbremnes/Documents/GIS_files/ruggedness_filtered.tif")
    bpi_broad_ds = gdal.Open("/Users/jensbremnes/Documents/GIS_files/bpi_broad_reduced.tif")
    depth_ds = gdal.Open("/Users/jensbremnes/Documents/GIS_files/depth_reduced.tif")
    coral_ds = gdal.Open("/Users/jensbremnes/Documents/GIS_files/corals_filtered_358_374.tif")

    # Get arrays
    current_speed_array = current_speed_ds.GetRasterBand(1).ReadAsArray()
    slope_array = slope_ds.GetRasterBand(1).ReadAsArray()
    ruggedness_array = ruggedness_ds.GetRasterBand(1).ReadAsArray()
    bpi_broad_array = bpi_broad_ds.GetRasterBand(1).ReadAsArray()
    depth_array = depth_ds.GetRasterBand(1).ReadAsArray()
    coral_array = coral_ds.GetRasterBand(1).ReadAsArray()

    # Get geotransform and projection of an existing raster
    geotransform = depth_ds.GetGeoTransform()
    projection = depth_ds.GetProjection()

    # Get a gdal driver to use for raster creation
    driver_tiff = gdal.GetDriverByName("GTiff")

    # Create new raster dataset
    risk_map_ds = driver_tiff.Create("/Users/jensbremnes/Documents/GIS_files/risk_map_" + name_postfix, xsize=depth_ds.RasterXSize, ysize=depth_ds.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    h1_ds  = driver_tiff.Create("/Users/jensbremnes/Documents/GIS_files/h1_" + name_postfix, xsize=depth_ds.RasterXSize, ysize=depth_ds.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    h2_ds  = driver_tiff.Create("/Users/jensbremnes/Documents/GIS_files/h2_" + name_postfix, xsize=depth_ds.RasterXSize, ysize=depth_ds.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    h3_ds  = driver_tiff.Create("/Users/jensbremnes/Documents/GIS_files/h3_" + name_postfix, xsize=depth_ds.RasterXSize, ysize=depth_ds.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    h4_ds  = driver_tiff.Create("/Users/jensbremnes/Documents/GIS_files/h4_" + name_postfix, xsize=depth_ds.RasterXSize, ysize=depth_ds.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    h5_ds  = driver_tiff.Create("/Users/jensbremnes/Documents/GIS_files/h5_" + name_postfix, xsize=depth_ds.RasterXSize, ysize=depth_ds.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    h6_ds  = driver_tiff.Create("/Users/jensbremnes/Documents/GIS_files/h6_" + name_postfix, xsize=depth_ds.RasterXSize, ysize=depth_ds.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    h7_ds  = driver_tiff.Create("/Users/jensbremnes/Documents/GIS_files/h7_" + name_postfix, xsize=depth_ds.RasterXSize, ysize=depth_ds.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    h8_ds  = driver_tiff.Create("/Users/jensbremnes/Documents/GIS_files/h8_" + name_postfix, xsize=depth_ds.RasterXSize, ysize=depth_ds.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    a1_ds  = driver_tiff.Create("/Users/jensbremnes/Documents/GIS_files/a1_" + name_postfix, xsize=depth_ds.RasterXSize, ysize=depth_ds.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    a2_ds  = driver_tiff.Create("/Users/jensbremnes/Documents/GIS_files/a2_" + name_postfix, xsize=depth_ds.RasterXSize, ysize=depth_ds.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    a3_ds  = driver_tiff.Create("/Users/jensbremnes/Documents/GIS_files/a3_" + name_postfix, xsize=depth_ds.RasterXSize, ysize=depth_ds.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    a4_ds  = driver_tiff.Create("/Users/jensbremnes/Documents/GIS_files/a4_" + name_postfix, xsize=depth_ds.RasterXSize, ysize=depth_ds.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    a5_ds  = driver_tiff.Create("/Users/jensbremnes/Documents/GIS_files/a5_" + name_postfix, xsize=depth_ds.RasterXSize, ysize=depth_ds.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    rec_ds = driver_tiff.Create("/Users/jensbremnes/Documents/GIS_files/rec_" + name_postfix, xsize=depth_ds.RasterXSize, ysize=depth_ds.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    cata_ds = driver_tiff.Create("/Users/jensbremnes/Documents/GIS_files/cata_" + name_postfix, xsize=depth_ds.RasterXSize, ysize=depth_ds.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    crit_ds = driver_tiff.Create("/Users/jensbremnes/Documents/GIS_files/crit_" + name_postfix, xsize=depth_ds.RasterXSize, ysize=depth_ds.RasterYSize, bands=1, eType=gdal.GDT_Float32)
    mod_ds = driver_tiff.Create("/Users/jensbremnes/Documents/GIS_files/mod_" + name_postfix, xsize=depth_ds.RasterXSize, ysize=depth_ds.RasterYSize, bands=1, eType=gdal.GDT_Float32)

    # set geotransform and projection
    risk_map_ds.SetGeoTransform(geotransform)
    risk_map_ds.SetProjection(projection)

    h1_ds.SetGeoTransform(geotransform)
    h1_ds.SetProjection(projection)
    h2_ds.SetGeoTransform(geotransform)
    h2_ds.SetProjection(projection)
    h3_ds.SetGeoTransform(geotransform)
    h3_ds.SetProjection(projection)
    h4_ds.SetGeoTransform(geotransform)
    h4_ds.SetProjection(projection)
    h5_ds.SetGeoTransform(geotransform)
    h5_ds.SetProjection(projection)
    h6_ds.SetGeoTransform(geotransform)
    h6_ds.SetProjection(projection)
    h7_ds.SetGeoTransform(geotransform)
    h7_ds.SetProjection(projection)
    h8_ds.SetGeoTransform(geotransform)
    h8_ds.SetProjection(projection)

    a1_ds.SetGeoTransform(geotransform)
    a1_ds.SetProjection(projection)
    a2_ds.SetGeoTransform(geotransform)
    a2_ds.SetProjection(projection)
    a3_ds.SetGeoTransform(geotransform)
    a3_ds.SetProjection(projection)
    a4_ds.SetGeoTransform(geotransform)
    a4_ds.SetProjection(projection)
    a5_ds.SetGeoTransform(geotransform)
    a5_ds.SetProjection(projection)
    rec_ds.SetGeoTransform(geotransform)
    rec_ds.SetProjection(projection)

    cata_ds.SetGeoTransform(geotransform)
    cata_ds.SetProjection(projection)
    crit_ds.SetGeoTransform(geotransform)
    crit_ds.SetProjection(projection)
    mod_ds.SetGeoTransform(geotransform)
    mod_ds.SetProjection(projection)

    # Get array to be filled in
    risk_map_band = risk_map_ds.GetRasterBand(1)
    risk_map_array = risk_map_band.ReadAsArray()

    h1_band = h1_ds.GetRasterBand(1)
    h1_array = h1_band.ReadAsArray()
    h2_band = h2_ds.GetRasterBand(1)
    h2_array = h2_band.ReadAsArray()
    h3_band = h3_ds.GetRasterBand(1)
    h3_array = h3_band.ReadAsArray()
    h4_band = h4_ds.GetRasterBand(1)
    h4_array = h4_band.ReadAsArray()
    h5_band = h5_ds.GetRasterBand(1)
    h5_array = h5_band.ReadAsArray()
    h6_band = h6_ds.GetRasterBand(1)
    h6_array = h6_band.ReadAsArray()
    h7_band = h7_ds.GetRasterBand(1)
    h7_array = h7_band.ReadAsArray()
    h8_band = h8_ds.GetRasterBand(1)
    h8_array = h8_band.ReadAsArray()

    a1_band = a1_ds.GetRasterBand(1)
    a1_array = a1_band.ReadAsArray()
    a2_band = a2_ds.GetRasterBand(1)
    a2_array = a2_band.ReadAsArray()
    a3_band = a3_ds.GetRasterBand(1)
    a3_array = a3_band.ReadAsArray()
    a4_band = a4_ds.GetRasterBand(1)
    a4_array = a4_band.ReadAsArray()
    a5_band = a5_ds.GetRasterBand(1)
    a5_array = a5_band.ReadAsArray()
    rec_band = rec_ds.GetRasterBand(1)
    rec_array = rec_band.ReadAsArray()

    cata_band = cata_ds.GetRasterBand(1)
    cata_array = cata_band.ReadAsArray()
    crit_band = crit_ds.GetRasterBand(1)
    crit_array = crit_band.ReadAsArray()
    mod_band = mod_ds.GetRasterBand(1)
    mod_array = mod_band.ReadAsArray()

    print("Starting to fill in data")

    (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = h1_ds.GetGeoTransform()
    shape = np.shape(h1_array)
    rows = shape[0]
    cols = shape[1]
    counter = 0

    # Get projection used to convert between lat/lon and WGS84
    myProj_32 = Proj(proj='utm',zone=32,ellps='WGS84', preserve_units=False)

    # Open Bayesian network
    bn = pysmile.Network()
    bn.read_file("/Users/jensbremnes/Desktop/STPA/spatio-temporal_0512_dev.xdsl")
    for handle in bn.get_all_nodes():
        if bn.get_node_id(handle) == "Risk":
            risk_handle = handle
            break

    num_iter = cols*rows
    it = 1
    for col in range(cols):

        for row in range(rows):

            # print progress
            print(it/num_iter)
            it += 1

            # x is west-east and y is north-south, given in UTM32V.
            x_index = col
            y_index = row
            x_coord = x_size*x_index + x_rotation*y_index + x_size*0.5 + x_rotation*0.5 + upper_left_x
            y_coord = y_rotation*x_index + y_size*y_index + y_rotation*0.5 + y_size*0.5 + upper_left_y
            coord = x_coord, y_coord

            lon, lat = myProj_32(x_coord, y_coord, inverse=True)

            # Get respective values from raster layers
            current_speed = current_speed_array[row][col]
            slope = slope_array[row][col]
            ruggedness = ruggedness_array[row][col]
            bpi = bpi_broad_array[row][col]
            depth = -depth_array[row][col]
            coral = coral_array[row][col]

            if coral > 1.0 or coral < 0.0 or slope < -1:
                h1_array[row][col] = -99
                h2_array[row][col] = -99
                h3_array[row][col] = -99
                h4_array[row][col] = -99
                h5_array[row][col] = -99
                h6_array[row][col] = -99
                h7_array[row][col] = -99
                h8_array[row][col] = -99
                a1_array[row][col] = -99
                a2_array[row][col] = -99
                a3_array[row][col] = -99
                a4_array[row][col] = -99
                a5_array[row][col] = -99
                risk_map_array[row][col] = -99
                rec_array[row][col] = -99
                cata_array[row][col] = -99
                crit_array[row][col] = -99
                mod_array[row][col] = -99
            else:

                # For each node, set the right evidence
                bn.set_evidence("Altitude_setpoint", altitude_ev)
                bn.set_evidence("Speed_setpoint", "Mild")
                bn.set_evidence("Motion_capabilities", "Mild")
                bn.set_evidence("Vessel_type", "Mild")

                bn.set_evidence("Visibility", case)
                bn.set_evidence("Rain", case)
                bn.set_evidence("Temperature", case)
                bn.set_evidence("Experience", case)
                bn.set_evidence("Waves", case)
                bn.set_evidence("Mission_duration", case)
                bn.set_evidence("Wind_speed", case)

                # distance to AUV
                center = 575541, 7052712
                distance_to_center = distance.euclidean(center, coord)

                # Distance to AUV
                if distance_to_center > 1500:
                    distance_to_center_ev = "Severe"
                elif distance_to_center > 750:
                    distance_to_center_ev = "Moderate"
                else:
                    distance_to_center_ev = "Mild"
                bn.set_evidence("Distance_to_AUV", distance_to_center_ev)

                if case == "Moderate" or case == "Severe":
                    if current_speed > 0.15:
                        current_speed_ev = "Severe"
                    else:
                        current_speed_ev = "Moderate"
                else:
                    if current_speed > 0.15:
                        current_speed_ev = "Moderate"
                    else:
                        current_speed_ev = "Mild"
                bn.set_evidence("Current", current_speed_ev)

                # depth
                if depth < 20:
                    depth_ev = "Severe"
                elif depth > 50:
                    depth_ev = "Moderate"
                else:
                    depth_ev = "Mild"
                bn.set_evidence("Depth", depth_ev)

                if abs(bpi) > 100:
                    bpi_ev = "Severe"
                elif abs(bpi) > 50:
                    bpi_ev = "Moderate"
                else:
                    bpi_ev = "Mild"
                bn.set_evidence("BPI", bpi_ev)

                # Slope
                if slope > 15:
                    slope_ev = "Severe"
                elif slope > 10:
                    slope_ev = "Moderate"
                else:
                    slope_ev = "Mild"
                bn.set_evidence("Slope", slope_ev)

                if ruggedness > 0.01:
                    ruggedness_ev = "Severe"
                elif ruggedness > 0.005:
                    ruggedness_ev = "Moderate"
                else:
                    ruggedness_ev = "Mild"
                bn.set_evidence("Ruggedness", ruggedness_ev)

                for handle in bn.get_all_nodes():
                    if bn.get_node_id(handle) == "Predicted_OOI":
                        coral_handle = handle
                        break

                if coral >= 0.0 and coral <= 1.0:
                    bn.update_beliefs()
                    coral_value = bn.get_node_value("H8")
                    coral_value[0] = 1.0-coral
                    coral_value[1] = coral
                    bn.set_virtual_evidence("H8", coral_value)

                # Find distance to center (USBL)
                center = 578823, 7054089  # easting/northing (UTM zone 32V)
                distance_to_center = distance.euclidean(center, coord)

                # Shipping lane
                shipping_lane = Polygon([[574522, 7051894], [576095, 7053037], [576721, 7052426], [575133, 7051033]])
                point = Point(coord)
                distance_to_lane = shipping_lane.boundary.distance(point)
                if shipping_lane.contains(point):
                    if case == "Severe":
                        traffic_ev = "Severe"
                    else:
                        traffic_ev = "Moderate"
                else:
                    traffic_ev = "Mild"
                bn.set_evidence("Ship_traffic", traffic_ev)

                # Fishing activity
                fishing_area = Polygon([[574318.8,7053518.5], [574338.4,7052783.6], [575715.1,7052842.4], [575431.0,7053469.5]])
                if fishing_area.contains(point):
                    fishing_ev = "Severe"
                else:
                    fishing_ev = "Mild"
                bn.set_evidence("Fishing_area", fishing_ev)

                # distance to shore
                north_shoreline = Polygon([[571026.63, 7052041.62], [571027.63, 7052042.62], [580614.31, 7059807.04], [580615.31, 7059808.04]])
                tautra_shoreline = Polygon([[578590.45, 7049525.48], [578591.45, 7049526.48], [580574.27, 7051991.52], [580575.27, 7051992.52]])
                distance_to_land = min(north_shoreline.boundary.distance(point), tautra_shoreline.boundary.distance(point))

                if distance_to_land < 200:
                    distance_to_land_ev = "Severe"
                elif distance_to_land < 500:
                    distance_to_land_ev = "Moderate"
                else:
                    distance_to_land_ev = "Mild"
                bn.set_evidence("Distance_to_shore", distance_to_land_ev)

                # Update beliefs (can I update beliefs on only one node?)
                bn.update_beliefs()

                # Retrive risk value
                h1_distribution = bn.get_node_value("H1")
                h2_distribution = bn.get_node_value("H2")
                h3_distribution = bn.get_node_value("H3")
                h4_distribution = bn.get_node_value("H4")
                h5_distribution = bn.get_node_value("H5")
                h6_distribution = bn.get_node_value("H6")
                h7_distribution = bn.get_node_value("H7")
                h8_distribution = bn.get_node_value("H8")

                a1_distribution = bn.get_node_value("A1")
                a2_distribution = bn.get_node_value("A2")
                a3_distribution = bn.get_node_value("A3")
                a4_distribution = bn.get_node_value("A4")
                a5_distribution = bn.get_node_value("A5")
                rec_distribution = bn.get_node_value("Recovery_effectiveness")
                samlenode_distribution = bn.get_node_value("samlenode")

                risk_value = bn.get_node_value("Total_risk")
                #risk_map_array[row][col] = risk_value[0]

                h1_value = h1_distribution[0]
                h1_array[row][col] = risk_value[0] #1.0 - h1_value
                h2_value = h2_distribution[0]
                h2_array[row][col] = 1.0 - h2_value
                h3_value = h3_distribution[0]
                h3_array[row][col] = 1.0 - h3_value
                h4_value = h4_distribution[0]
                h4_array[row][col] = 1.0 - h4_value
                h5_value = h5_distribution[0]
                h5_array[row][col] = h5_value
                h6_value = h6_distribution[0]
                h6_array[row][col] = 1.0 - h6_value
                h7_value = h7_distribution[0]
                h7_array[row][col] = h7_value
                h8_value = h8_distribution[0]
                h8_array[row][col] = h8_value

                #l1_value = l1_distribution[0]
                a1_value = a1_distribution[0]
                a1_array[row][col] = math.log10(a1_value)
                a2_value = a2_distribution[0]
                a2_array[row][col] = math.log10(a2_value)
                a3_value = a3_distribution[0]
                a3_array[row][col] = math.log10(a3_value)
                a4_value = a4_distribution[0]
                a4_array[row][col] = math.log10(a4_value)
                a5_value = a5_distribution[0]
                a5_array[row][col] = a5_value
                rec_value = 1*rec_distribution[0] + 0.5*rec_distribution[1]
                rec_array[row][col] = rec_value

                cata_array[row][col] = math.log10(samlenode_distribution[0])
                crit_array[row][col] = math.log10(samlenode_distribution[1])
                mod_array[row][col] = math.log10(samlenode_distribution[2])

                alpha = 1/3
                cost_vec = [3000000, 8000, 800, 400]

                c1_dist = bn.get_node_value("C1")
                if c1_dist[0] > (1-alpha):
                    c1_cvar = 3000000*c1_dist[4]/alpha + 8000*c1_dist[3]/alpha + 800*c1_dist[2]/alpha + 400*c1_dist[1]/alpha
                elif (c1_dist[0] + c1_dist[1]) > (1-alpha):
                    rem = (c1_dist[0] + c1_dist[1]) - (1-alpha)
                    c1_cvar = 3000000*c1_dist[4]/alpha + 8000*c1_dist[3]/alpha + 800*c1_dist[2]/alpha + 400*rem/alpha
                elif (c1_dist[0] + c1_dist[1] + c1_dist[2]) > (1-alpha):
                    rem = (c1_dist[0] + c1_dist[1] + c1_dist[2]) - (1-alpha)
                    c1_cvar = 3000000*c1_dist[4]/alpha + 8000*c1_dist[3]/alpha + 800*rem/alpha

                c2_dist = bn.get_node_value("C2")
                if c2_dist[0] > (1-alpha):
                    c2_cvar = 3000000*c2_dist[4]/alpha + 8000*c2_dist[3]/alpha + 800*c2_dist[2]/alpha + 400*c2_dist[1]/alpha
                elif (c2_dist[0] + c2_dist[1]) > (1-alpha):
                    rem = (c2_dist[0] + c2_dist[1]) - (1-alpha)
                    c2_cvar = 3000000*c2_dist[4]/alpha + 8000*c2_dist[3]/alpha + 800*c2_dist[2]/alpha + 400*rem/alpha
                elif (c2_dist[0] + c2_dist[1] + c2_dist[2]) > (1-alpha):
                    rem = (c2_dist[0] + c2_dist[1] + c2_dist[2]) - (1-alpha)
                    c2_cvar = 3000000*c2_dist[4]/alpha + 8000*c2_dist[3]/alpha + 800*rem/alpha

                c3_dist = bn.get_node_value("C3")
                if c3_dist[0] > (1-alpha):
                    c3_cvar = 3000000*c3_dist[4]/alpha + 8000*c3_dist[3]/alpha + 800*c3_dist[2]/alpha + 400*c3_dist[1]/alpha
                elif (c3_dist[0] + c3_dist[1]) > (1-alpha):
                    rem = (c3_dist[0] + c3_dist[1]) - (1-alpha)
                    c3_cvar = 3000000*c3_dist[4]/alpha + 8000*c3_dist[3]/alpha + 800*c3_dist[2]/alpha + 400*rem/alpha
                elif (c3_dist[0] + c3_dist[1] + c3_dist[2]) > (1-alpha):
                    rem = (c3_dist[0] + c3_dist[1] + c3_dist[2]) - (1-alpha)
                    c3_cvar = 3000000*c3_dist[4]/alpha + 8000*c3_dist[3]/alpha + 800*rem/alpha

                c4_dist = bn.get_node_value("C4")
                if c4_dist[0] > (1-alpha):
                    c4_cvar = 3000000*c4_dist[4]/alpha + 8000*c4_dist[3]/alpha + 800*c4_dist[2]/alpha + 400*c4_dist[1]/alpha
                elif (c4_dist[0] + c4_dist[1]) > (1-alpha):
                    rem = (c4_dist[0] + c4_dist[1]) - (1-alpha)
                    c4_cvar = 3000000*c4_dist[4]/alpha + 8000*c4_dist[3]/alpha + 800*c4_dist[2]/alpha + 400*rem/alpha
                elif (c4_dist[0] + c4_dist[1] + c4_dist[2]) > (1-alpha):
                    rem = (c4_dist[0] + c4_dist[1] + c4_dist[2]) - (1-alpha)
                    c4_cvar = 3000000*c4_dist[4]/alpha + 8000*c4_dist[3]/alpha + 800*rem/alpha

                c5_dist = bn.get_node_value("C5")
                if c5_dist[0] > (1-alpha):
                    c5_cvar = 3000000*c5_dist[4]/alpha + 8000*c5_dist[3]/alpha + 800*c5_dist[2]/alpha + 400*c5_dist[1]/alpha
                elif (c5_dist[0] + c5_dist[1]) > (1-alpha):
                    rem = (c5_dist[0] + c5_dist[1]) - (1-alpha)
                    c5_cvar = 3000000*c5_dist[4]/alpha + 8000*c5_dist[3]/alpha + 800*c5_dist[2]/alpha + 400*rem/alpha
                elif (c5_dist[0] + c5_dist[1] + c5_dist[2]) > (1-alpha):
                    rem = (c5_dist[0] + c5_dist[1] + c5_dist[2]) - (1-alpha)
                    c5_cvar = 3000000*c5_dist[4]/alpha + 8000*c5_dist[3]/alpha + 800*rem/alpha

                risk_map_array[row][col] = c1_cvar + c2_cvar + c3_cvar + c4_cvar + c5_cvar

    print("Finished")

    # Write data
    risk_map_band.WriteArray(risk_map_array, 0, 0)

    h1_band.WriteArray(h1_array, 0, 0)
    h2_band.WriteArray(h2_array, 0, 0)
    h3_band.WriteArray(h3_array, 0, 0)
    h4_band.WriteArray(h4_array, 0, 0)
    h5_band.WriteArray(h5_array, 0, 0)
    h6_band.WriteArray(h6_array, 0, 0)
    h7_band.WriteArray(h7_array, 0, 0)
    h8_band.WriteArray(h8_array, 0, 0)

    a1_band.WriteArray(a1_array, 0, 0)
    a2_band.WriteArray(a2_array, 0, 0)
    a3_band.WriteArray(a3_array, 0, 0)
    a4_band.WriteArray(a4_array, 0, 0)
    a5_band.WriteArray(a5_array, 0, 0)
    rec_band.WriteArray(rec_array, 0, 0)

    cata_band.WriteArray(cata_array, 0, 0)
    crit_band.WriteArray(crit_array, 0, 0)
    mod_band.WriteArray(mod_array, 0, 0)

    # flush data to disk, set the NoData value and calculate stats
    risk_map_band.FlushCache()
    risk_map_band.SetNoDataValue(-99)

    h1_band.FlushCache()
    h1_band.SetNoDataValue(-99)
    h2_band.FlushCache()
    h2_band.SetNoDataValue(-99)
    h3_band.FlushCache()
    h3_band.SetNoDataValue(-99)
    h4_band.FlushCache()
    h4_band.SetNoDataValue(-99)
    h5_band.FlushCache()
    h5_band.SetNoDataValue(-99)
    h6_band.FlushCache()
    h6_band.SetNoDataValue(-99)
    h7_band.FlushCache()
    h7_band.SetNoDataValue(-99)
    h8_band.FlushCache()
    h8_band.SetNoDataValue(-99)

    a1_band.FlushCache()
    a1_band.SetNoDataValue(-99)
    a2_band.FlushCache()
    a2_band.SetNoDataValue(-99)
    a3_band.FlushCache()
    a3_band.SetNoDataValue(-99)
    a4_band.FlushCache()
    a4_band.SetNoDataValue(-99)
    a5_band.FlushCache()
    a5_band.SetNoDataValue(-99)
    rec_band.FlushCache()
    rec_band.SetNoDataValue(-99)

    cata_band.FlushCache()
    cata_band.SetNoDataValue(-99)
    crit_band.FlushCache()
    crit_band.SetNoDataValue(-99)
    mod_band.FlushCache()
    mod_band.SetNoDataValue(-99)

    del(risk_map_ds)
    del(h1_ds)
    del(h2_ds)
    del(h3_ds)
    del(h4_ds)
    del(h5_ds)
    del(h6_ds)
    del(h7_ds)
    del(h8_ds)
    del(a1_ds)
    del(a2_ds)
    del(a3_ds)
    del(a4_ds)
    del(a5_ds)
    del(rec_ds)
    del(cata_ds)
    del(crit_ds)
    del(mod_ds)

if __name__ == '__main__':
    create_risk_raster(altitude_ev = "Close", case = "Severe")
