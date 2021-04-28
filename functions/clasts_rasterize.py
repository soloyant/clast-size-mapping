import pandas as pd
import scipy
import math
import os
import gdal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def clasts_rasterize(ClastImageFilePath, ClastSizeListCSVFilePath, RasterFileWritingPath, field = "Clast_length", parameter="quantile", cellsize=1, percentile=0.5, plot=True, figuresize = (15,20)):

    """Converts the clast information from vector type to raster type.
    ClastImageFilePath: Path of the geotiff file used to realize the clast detection and measurement
    ClastSizeListCSVFilePath: Path of the CSV file containing the list of previously detected and measured clasts
    RasterFileWritingPath: Path to be used for writing the raster produced by the present function
    field: field (i.e. clast dimension) to be considered for computation (default = "Clast_length")
    parameter: Parameter to be computed for each cell: 
        - "quantile": returns the quantile valued for the threshold specified by the "percentile" keyword 
        - "density": returns the density of objects per cell size unit
        - "average": returns the average value for each cell
        - "std": returns the standard deviation for each cell
        - "kurtosis": returns the kurtosis size for each cell
        - "skewness": returns the skewness value for each cell
    cellsize: Wanted output raster cell size (same unit as the geotiff file used to realize the clast detection and measurement
    percentile: Percentile to be used for computing the quantile of each cell (default = 0.5, i.e. median)
    plot: Switch for displaying the produced maps (default = True)
    figuresize: Size of the displayed figure (default = (10,10))
    """
    
    clasts = pd.read_csv(ClastSizeListCSVFilePath)
    local_clasts = clasts.copy()
    local_clasts['y'] = clasts['y']-np.min(clasts['y'])
    local_clasts['x'] = clasts['x']-np.min(clasts['x'])
    n_rows = math.ceil(np.max(local_clasts['y'])/cellsize)
    n_cols = math.ceil(np.max(local_clasts['x'])/cellsize)
    
    p = np.zeros((n_rows,n_cols))
    for m in range(0, n_rows):
        for n in range(0, n_cols):
            crop = local_clasts[(local_clasts['y']>=m*cellsize) & 
                                 (local_clasts['y']<(m+1)*cellsize) & 
                                 (local_clasts['x']>=n*cellsize) & 
                                 (local_clasts['x']<(n+1)*cellsize)]
            if np.shape(crop)[0]>0:
                if str.lower(parameter) == "quantile":
                    p[m,n] = np.quantile(crop[field], percentile)
                if str.lower(parameter) == "density":
                    p[m,n] = (np.shape(crop[field])[0]+1)/(cellsize**2)
                if str.lower(parameter) == "average":
                    p[m,n] = np.mean(crop[field])
                if str.lower(parameter) == "kurtosis":
                    p[m,n] = scipy.stats.kurtosis(crop[field])
                if str.lower(parameter) == "skewness":
                    p[m,n] = scipy.stats.skew(crop[field])
                if str.lower(parameter) == "std":
                    p[m,n] = np.std(crop[field])

    raster = gdal.Open(ClastImageFilePath, gdal.GA_ReadOnly)
    geotransform = raster.GetGeoTransform()
    image = np.dstack([raster.GetRasterBand(1).ReadAsArray(), raster.GetRasterBand(2).ReadAsArray(),raster.GetRasterBand(3).ReadAsArray()] )
    
    driver = gdal.GetDriverByName("GTiff")
    arr_out = p
    outdata = driver.Create(RasterFileWritingPath, n_cols, n_rows, 1, gdal.GDT_Float64)
    outdata.SetGeoTransform([np.min(clasts['x']), cellsize, 0, np.min(clasts['y']), 0, cellsize ])
    outdata.SetProjection(raster.GetProjection())
    outdata.GetRasterBand(1).WriteArray(arr_out)
    outdata.GetRasterBand(1).SetNoDataValue(0)
    outdata.FlushCache() 
    outdata = None
    ds=None
    if plot==True:
        fig = plt.figure(figsize = figuresize)
        ax1 = fig.add_subplot(2,1,1)
        ax1.imshow(image, interpolation='none')
        ax1.set_title("Ortho-image")
        ax2 = fig.add_subplot(2,1,2) 
        pos = ax2.imshow(np.flipud(p), interpolation='none')
        if parameter=="quantile":
            ax2.set_title("D"+str(int(percentile*100))+" map")
        else:
            ax2.set_title(parameter+" map")
        fig.colorbar(pos, ax=ax2)
    return p