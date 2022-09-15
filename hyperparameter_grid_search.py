# # point cloud processing tools
# import pdal
# raster/vector tools
from osgeo import gdal, ogr, osr
# image manipulation and visualization tools
import matplotlib.pyplot as plt
from skimage.morphology import watershed
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import ndimage as ndi
from matplotlib.colors import ListedColormap
# graph tools
import networkx as nx
from graphviz import Digraph as dg
# system tools
import time
import os
import csv
import warnings

warnings.filterwarnings("ignore")

area_condition = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,
                  90, 95, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210,
                  220, 230, 240, 250, 260, 270, 280, 290, 300, 350, 400, 450, 500, 550,
                  600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200]

steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for k in area_condition:
    for w in steps:

        print('area_cond is ' + str(k) + ', and step is ' + str(w))
        pathname = './single_tree_proj/data_ut/junk/'      # ./single_tree_proj/data_ut/loop_finer_2/
        # Create a folder for each iteration to save files within them.
        os.makedirs(pathname + str(k) + '_' + str(w))
        savepath = pathname + str(k) + '_' + str(w) + '/'

        data = gdal.Open(pathname + 'CHM.tif', gdal.GA_ReadOnly)
        data_array1 = data.ReadAsArray()
        data_array1[data_array1 == -9999.0] = 0
        # apply gaussian smoothing
        g_filtered = gaussian_filter(data_array1, sigma=2)
        # plt.figure(), plt.imshow(g_filtered, cmap='jet')
        # plt.show(block=False), plt.pause(5), plt.close()

        # TODO Performing Level Cutting on the smoothed CHM and save them.
        start = 2.0  # minimum height
        stop = np.max(g_filtered)  # maximum height
        step = w  # vertical step
        levels = np.zeros_like(g_filtered, dtype=bool)
        for i in np.arange(start, stop, step):
            levels = np.dstack((levels, g_filtered > i))
        levels = np.delete(levels, 0, 2)
        print('>>>   Level-Cut step resulted into an array of the following shape:  ')
        print(levels.shape)

        t0 = time.time()
        # -------------------------------------------------------------------------------------   The magic happens here
        # TODO Building graph on the objects in each level
        '''
        Working with NetworkX
        The networkX graphs are used to construct the tree structure and
        to be able to query the graph for individual trees.

        For the sake of graph visualization we use Graphviz. For graphviz,
        we export a dot file and then plot it through command line options.

        Descriptions are added line by line.

        CMD: dot -Tpdf first_graph.dot -o first_graph.pdf
        '''
        labels = np.zeros_like(levels, dtype=np.int)
        G = nx.Graph()  # NetworkX graph
        G.add_node('ROOT')
        viz_graph = dg()  # Graphviz visualization
        viz_graph.node('ROOT')
        for i in range(levels.shape[2]):
            # extract i-th layer of the level-cut matrix
            ith_level = np.atleast_2d(levels[:, :, i])
            # assign a label to each separated segment (counting them)
            labels[:, :, i], num_of_segments = ndi.label(ith_level)

            # loop through each individual segment in each level-cut layer to find
            # its parent and connect it to the graph.
            for l in range(num_of_segments):
                # apply an area constraint to remove small objects
                area_cond = len(np.where(labels[:, :, i] == l + 1)[0]) > k
                if area_cond:
                    viz_graph.node(str(i) + '/' + str(l + 1))
                    G.add_node((str(i), l + 1), layer=str(i))

                    if i == 0:
                        G.add_edge('ROOT', (str(i), l + 1))
                        viz_graph.edge('ROOT', str(i) + '/' + str(l + 1))
                    else:
                        parent = np.unique(labels[:, :, i - 1][labels[:, :, i] == l + 1])[0]
                        G.add_edge((str(i - 1), parent), (str(i), l + 1))
                        viz_graph.edge(str(i - 1) + '/' + str(parent), str(i) + '/' + str(l + 1))

        # Print the result in a dot file to prepare for visualization
        with open(savepath + 'graph.dot', 'w') as f:
            print(viz_graph.source, file=f)
        cmd = 'dot -Tpdf ' + savepath + 'graph.dot' + ' -o ' + savepath + 'graph.pdf'
        os.system(cmd)

        # TODO Perform graph pruning based on each node's degree.
        for n in G.nodes:
            don = G.degree[n]

            if don == 2:
                pre = list(G.neighbors(n))[0]
                post = list(G.neighbors(n))[1]
                G.remove_edge(pre, n)
                G.remove_edge(n, post)
                G.add_edge(pre, post)

        # TODO Draw improved graph with graphviz, after pruning
        # first, remove all isolated points
        G.remove_nodes_from(list(nx.isolates(G)))
        vizz_graph = dg()  # define a new graphviz object and plot in it.
        for n in G.nodes:
            if n == 'ROOT':
                vizz_graph.node('ROOT')
            else:
                vizz_graph.node(n[0] + '/' + str(n[1]))

        for e in G.edges:
            if e[0] == 'ROOT':
                vizz_graph.edge('ROOT', e[1][0] + '/' + str(e[1][1]), color='blue')
            else:
                vizz_graph.edge(e[0][0] + '/' + str(e[0][1]),
                                e[1][0] + '/' + str(e[1][1]), color='blue')

        with open(savepath + 'pruned.dot', 'w') as f:
            print(vizz_graph.source, file=f)
        cmd = 'dot -Tpdf ' + savepath + 'pruned.dot' + ' -o ' + savepath + 'pruned.pdf'
        os.system(cmd)

        # TODO Find individual tree tops
        '''
        Tree tops are those nodes in the graph that have degree of 1
        and are located at the end of the hierarchy.

        Here we create a list of tree tops containing their level number
        and their object label number. So we can find them.

        After finding the IDs, we find the corresponding pixels.

        The number of tree tops found is highly related to the AREA_COND
        defined in the graph building procedure. The larger the AREA_COND number,
        the less the trees will be found; but the results may look more real.

        We call tree tops the "detected peaks".
        '''
        trees = np.zeros((1, 2), dtype=np.int)
        for n in G.nodes:
            if G.degree[n] == 1:
                tree = np.array([int(n[0], 10), n[1]])
                trees = np.vstack((trees, tree))
        trees = np.delete(trees, 0, axis=0)

        # Record processing time
        print(time.time() - t0)

        print('###    Number of tree candidates are:    ', trees.shape[0])
        TREE = np.zeros((labels.shape[0], labels.shape[1]))
        for i in range(trees.shape[0]):
            temp = np.atleast_2d(labels[:, :, trees[i, 0]])
            TREE[temp == trees[i, 1]] = 1
        # ----------------------------------------------------------------------------------------------   And ends here

        detected_peaks = TREE
        markers = ndi.label(detected_peaks)[0]  # unique blobs
        unique_labels = np.amax(markers)  # Number of detected trees
        r_center = []
        c_center = []
        for i in range(1, unique_labels + 1):
            r = np.where(markers == i)[0]
            c = np.where(markers == i)[1]
            r_center.append(np.mean(r))
            c_center.append(np.mean(c))

        plt.figure(figsize=(8, 8)), plt.tight_layout()
        plt.imshow(TREE, cmap='gray'), plt.plot(c_center, r_center, 'r*', markersize=2)
        plt.title('Tree Tops locations and tree centroids (red *)')
        plt.xticks([]), plt.yticks([])
        plt.savefig(savepath + 'tree-top locations.png', dpi=300)
        plt.show(block=False), plt.pause(7), plt.close()

        # TODO Perform watershed segmentation to expand tree tops found in the
        #  previous step to whole tree canopy region.
        '''
        First, a distance map is created based on the CHM raster.

        Then, we consider the tree tops as local maximums and try to expand them.
        This requires to label the tree tops from 1 to N.

        We mask all point clouds below a threshold because they do not 
        show any trees. This is done through Watershed function arguments.
        '''
        distance = ndi.distance_transform_edt(data_array1)
        # local_maxi = detected_peaks
        # markers = ndi.label(local_maxi)[0]
        # plt.figure()
        # plt.subplot(131), plt.imshow(distance, cmap='gray'), plt.title('distance image')
        # plt.subplot(132), plt.imshow(local_maxi, cmap='gray'), plt.title('local maximums')
        # plt.subplot(133), plt.imshow(markers, cmap='tab20b'), plt.title('labeled markers')
        # plt.show()
        labels = watershed(-data_array1, markers, mask=data_array1 > 1)
        # plt.figure()
        # plt.subplot(121), plt.imshow(labels, cmap='tab20b')
        # plt.subplot(122), plt.imshow(data_array1, cmap='jet')
        # plt.show()

        # TODO Colorize the labels with random Red, Green, and Blue numbers.
        '''
        First of all, we create a random colormap to show individual trees with
        individual random colors. This is visually better than the matplotlib colormaps.

        Then we create three bands and assign each band a random value to form the RGB image.
        '''
        colmap = np.random.uniform(0, 1, 3 * 1000).reshape((1000, 3))
        colmap = ListedColormap(colmap)
        plt.figure(figsize=(8, 8)), plt.tight_layout()
        plt.imshow(labels, cmap=colmap), plt.title('Random Colormap')
        plt.xticks([]), plt.yticks([])
        plt.savefig(savepath + 'random colored trees.png', dpi=300)
        plt.show(block=False), plt.pause(7), plt.close()

        rgb_labels = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=int)
        colmap2 = np.random.randint(1, 257, 3 * 1000).reshape((1000, 3))

        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                dn = int(labels[i, j])
                rgb_labels[i, j, :] = colmap2[dn, :]

        # plt.figure(), plt.imshow(rgb_labels), plt.title('RGB format of labels')
        # plt.show(block=False), plt.pause(5), plt.close()

        # TODO Saving individual tree segments in GeoTiff format
        '''
        To save in GeoTiff we need to have geographic coordinates and also
        a defined projection system. We found the min/max of latitude and longitudes
        from the bounding box of the raw point cloud. In fact, the coordinates were in
        UTM and using an online converter, we changed them to appropriate lat/lon degrees.

        The only thing required to do to get to a colorized point cloud of individual trees
        is to create a .tfw georeference file in an external software like QGIS to append the
        colors to the points. Coloring will be done in LAStools.
        '''
        lat = [36.57815, 36.579193]
        lon = [52.039653, 52.04097658]

        nX = labels.shape[0]
        nY = labels.shape[1]

        xmin, ymin, xmax, ymax = [min(lon), min(lat), max(lon), max(lat)]
        xres = (xmax - xmin) / float(nX)
        yres = (ymax - ymin) / float(nY)

        # geotransform = (xmin, xres, 0, ymax, 0, -yres)  # Save the raster/vector file in Lat/Lon projection.
        geotransform = data.GetGeoTransform()  # Save the raster/vector file in UTM projection.
        # (compatible with original files)

        # create the 3-band raster file
        dst_ds = gdal.GetDriverByName('GTiff').Create(savepath + 'RGB_labels.tif',
                                                      nY, nX, 3, gdal.GDT_Byte)
        dst_ds.SetGeoTransform(geotransform)  # specify coordinatess
        srs = osr.SpatialReference()  # establish encoding
        dst_ds.SetProjection(srs.ExportToWkt())  # export coordinates to file
        dst_ds.GetRasterBand(1).WriteArray(rgb_labels[:, :, 0])  # write r-band to the raster
        dst_ds.GetRasterBand(2).WriteArray(rgb_labels[:, :, 1])  # write g-band to the raster
        dst_ds.GetRasterBand(3).WriteArray(rgb_labels[:, :, 2])  # write b-band to the raster
        dst_ds.FlushCache()  # write to disk
        dst_ds = None

        # Save centroids
        centroids = np.zeros((unique_labels, 5))
        centroids[:, 0] = c_center  # image columns (x)
        centroids[:, 1] = r_center  # image rows (y)
        centroids[:, 2] = geotransform[0] + centroids[:, 0] * geotransform[1]  # X coord
        centroids[:, 3] = geotransform[3] + centroids[:, 1] * geotransform[5]  # Y coord
        for i in range(centroids.shape[0]):
            centroids[i, 4] = g_filtered[int(centroids[i, 1]), int(centroids[i, 0])]  # Tree heights
        wtr = csv.writer(open(savepath + 'centroids.csv', 'w'), delimiter=',', lineterminator='\n')
        for i in range(centroids.shape[0]): wtr.writerow(centroids[i, :])

        # TODO Vectorize (Polygonize) the tree locations.
        '''
        To be able to make polygons from each tree region we need to open the georeferenced
        image (RGB labels) which was already created.
        '''
        # Reading the already created raster file and polygonize it.
        raster = gdal.Open(savepath + 'RGB_labels.tif')
        band = raster.GetRasterBand(1)
        drv = ogr.GetDriverByName('ESRI Shapefile')
        outfile = drv.CreateDataSource(savepath + 'tree_vec.shp')
        outlayer = outfile.CreateLayer('polygonized raster', srs=None)
        newField = ogr.FieldDefn('Area', ogr.OFTInteger)
        outlayer.CreateField(newField)
        gdal.Polygonize(band, None, outlayer, 0, [])
        outfile = None
