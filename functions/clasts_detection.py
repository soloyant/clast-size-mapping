import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import measure
from os import listdir
from os.path import isfile, join
from osgeo import gdal
import pandas as pd

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN library & config file
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from functions import clasts_config
config = clasts_config.clastsConfig() 

# Import Measurement and post-processing functions
from functions import ellipse


def get_ax(rows=1, cols=1, size=16):

    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def clasts_detect(mode, imgpath, resolution=0.001, metric_cropsize=1, plot=True, saveplot=False, saveresults=False, devicemode="gpu"):

    """Detects and measure the clasts on a scaled image that can be either terrestrial (i.e. photograph of known resolution) or UAV derived (georeferenced ortho-image)
    mode: "terrestrial", "UAV"
    resolution: resolution of a terrestrial image (default = 0.001 m/pixel)
    imgpath: full path of the image/ortho-image
    metric_cropsize: Window size used for croping a UAV image into tiles (default = 1m)
    plot: (boolean) Display detections and histograms
    saveplot: (boolean) Save the detection and histogram figures
    saveresults: (boolean) Save the clast sizes into a CSV file
    devicemode: "gpu", "cpu" (default = "gpu") (see tesorflow-gpu documentation)
    """
    DEVICE = "/"+str.lower(devicemode)+":0"  # /cpu:0 or /gpu:0
    #Model weights directory
    model_dir = os.path.join(ROOT_DIR, "model_weights")

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=model_dir,
                                  config=config)

    # Path to the model weights file
    weights_path = os.path.join(model_dir, "mask_rcnn_clasts.h5")

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    
    # Segmentation & Measurement bloc
    class_names = ['BG', 'Clast'] # Detection class names
    if str.lower(mode)=="terrestrial":
        image=mpimg.imread(imgpath)
        results = model.detect([image], verbose=0)
        r = results[0]    
        n_detected_objects = np.shape(r['masks'])[2]
        if n_detected_objects>0:
            clasts=pd.DataFrame(np.zeros([n_detected_objects,8]))
            clasts.columns=['clast_ID', 'x', 'y', 'Major_axis', 'Minor_axis', 'Surface_area', 'Equivalent_diameter', 'Score']
            if plot==True:
                ax = get_ax(1)
                visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                            class_names, r['scores'], ax=ax,
                                            title="Predictions")
                fig = plt.gcf()
                fig.set_size_inches(16, 16)
                if saveplot==True:
                    fig.savefig(imgpath[0:-4]+'_mask_r-cnn.png', dpi=100)
                plt.show()
            s=np.zeros(n_detected_objects)
            D=np.zeros(n_detected_objects)
        
            if plot==True:
                plt.imshow(image)
            ax_list=np.zeros([n_detected_objects,2])
            for i in range(0, n_detected_objects):
            
                s[i]=np.sum(r['masks'][:,:,i]*1)*resolution**2
                D[i]=(2*np.sqrt(s[i]/np.pi))
                clasts['Surface_area'].iloc[i]=s[i]
                clasts['Equivalent_diameter'].iloc[i]=D[i]
            
                clast_i=r['masks'][:,:,i]
                contours_clast_i=np.asarray(measure.find_contours(clast_i.astype(int),0.99999))
                x=contours_clast_i[0][:,1]
                y=contours_clast_i[0][:,0]
                a = ellipse.fitEllipse(x,y)
                center = ellipse.ellipse_center(a)
                phi = ellipse.ellipse_angle_of_rotation(a)
                axes = ellipse.ellipse_axis_length(a) #results = radius !!
                R = np.arange(0,2*np.pi, 0.01)
                a, b = axes

                #Ellipse rotation
                xx = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
                yy = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)

                #Main axis rotation
                xx_ax1_norot=np.linspace(0,a,100)
                yy_ax1_norot=np.zeros(np.shape(xx_ax1_norot))
                xx_ax1_rot=center[0]+xx_ax1_norot*np.cos(phi)+yy_ax1_norot*np.sin(phi)
                yy_ax1_rot=center[1]+xx_ax1_norot*np.sin(phi)+yy_ax1_norot*np.cos(phi)

                xx_ax2_norot=np.linspace(0,b,100)
                yy_ax2_norot=np.zeros(np.shape(xx_ax2_norot))
                xx_ax2_rot=center[0]+xx_ax2_norot*np.cos(phi-(np.pi/2))+yy_ax2_norot*np.sin(phi-(np.pi/2))
                yy_ax2_rot=center[1]+xx_ax2_norot*np.sin(phi-(np.pi/2))+yy_ax2_norot*np.cos(phi-(np.pi/2))
            
                ax_list[i,:]=axes
            
                maj_ax=np.max(ax_list[i,:])
                min_ax=np.min(ax_list[i,:])
        
                clasts['clast_ID'].iloc[i]=int(i)
                clasts['x'].iloc[i]=abs(center[0])
                clasts['y'].iloc[i]=np.shape(image)[1]-abs(center[1]) #y coordinates are upside down
                clasts['Major_axis'].iloc[i]=abs(maj_ax*resolution*2) #(*2 for getting diameter values)
                clasts['Minor_axis'].iloc[i]=abs(min_ax*resolution*2)
        
                #Display
                if plot==True:
                    plt.plot(x,y, color = 'red') #Mask R-CNN segmentation
                    plt.plot(xx,yy, color = 'blue') #Ellipse
                    plt.plot(xx_ax1_rot,yy_ax1_rot, color = 'blue', linestyle='dashed') #Major ax
                    plt.plot(xx_ax2_rot,yy_ax2_rot, color = 'blue', linestyle='dashed') #Minor ax
        
            clasts['Score']=r['scores']
            if plot==True:
                fig = plt.gcf()
                fig.set_size_inches(16, 16)
                if saveplot==True:
                    fig.savefig(imgpath[0:-4]+'_ellipses.png', dpi=100)
                plt.show()
        
            print(' ')
            print('Detected clasts:', np.shape(r['masks'])[2])
            print(' ')
            print('Eq diameter D10 = '+ str(np.round(np.quantile(D, 0.1)*100,2))+' cm')
            print('Eq diameter D50 = '+ str(np.round(np.quantile(D, 0.5)*100,2))+' cm')
            print('Eq diameter D90 = '+ str(np.round(np.quantile(D, 0.9)*100,2))+' cm')
            if plot==True:
                plt.hist(D*100, 20)
                plt.xlabel('Grain Size (cm)')
                plt.ylabel('Number of clasts')
                fig = plt.gcf()
                fig.set_size_inches(10, 10)
                if saveplot==True:
                     fig.savefig(imgpath[0:-4]+'_histogram.png', dpi=100)
                plt.show()
            print(' ')
            print('Major Axis D10 = '+ str(round(np.quantile(clasts['Major_axis'], 0.1)*100,2))+' cm')
            print('Major Axis D50 = '+ str(round(np.quantile(clasts['Major_axis'], 0.5)*100,2))+' cm')
            print('Major Axis D90 = '+ str(round(np.quantile(clasts['Major_axis'], 0.9)*100,2))+' cm')
            print(' ')
            print('Minor Axis D10 = '+ str(round(np.quantile(clasts['Minor_axis'], 0.1)*100,2))+' cm')
            print('Minor Axis D50 = '+ str(round(np.quantile(clasts['Minor_axis'], 0.5)*100,2))+' cm')
            print('Minor Axis D90 = '+ str(round(np.quantile(clasts['Minor_axis'], 0.9)*100,2))+' cm')
            print(' ')
            print('-------------------------------------------------------')
            print(' ')
            if saveresults==True:
                clasts.to_csv(imgpath[0:-4]+'individual_clasts.csv', index=False)

    if str.lower(mode)=="uav":
        raster = gdal.Open(imgpath, gdal.GA_ReadOnly)
        geotransform = raster.GetGeoTransform()
        image = np.dstack([raster.GetRasterBand(1).ReadAsArray(), raster.GetRasterBand(2).ReadAsArray(), raster.GetRasterBand(3).ReadAsArray()] )
    
        shpimg=np.shape(image)
        resolution = np.abs(geotransform[1])
        image_x_corner = geotransform[0]
        image_y_corner = geotransform[3]
    
        cropsize=int(metric_cropsize/resolution)
        n_crops= int((shpimg[0]/cropsize)+1) * int((shpimg[1]/cropsize)+1)
        expected_maximum_number_of_detected_objects=n_crops*300    #30 is the empirical expected number of object per image
        total_number_of_detected_objects=0                         #updated in the loop
        clasts=pd.DataFrame(np.zeros([expected_maximum_number_of_detected_objects,9]))
        clasts.columns=['clast_ID', 'x', 'y', 'Major_axis', 'Minor_axis', 'Surface_area', 'Equivalent_diameter', 'Orientation', 'Score']

        for n in range(0, int(shpimg[0]/cropsize)+1): #Lignes
            for m in range(0, int(shpimg[1]/cropsize)+1): #Colonnes
            
                croppedimage=image[n*cropsize:n*cropsize+cropsize, m*cropsize:m*cropsize+cropsize,:]
                pixel_value=np.sum(np.sum(croppedimage))
                if (pixel_value>0):
                    results = model.detect([croppedimage], verbose=0)
                    r = results[0]
                    if plot==True:
                         #Displalying image and masks
                        visualize.display_instances(croppedimage, r['rois'], r['masks'], r['class_ids'], 
                                                    class_names, r['scores'],
                                                    title="Predictions")
                        plt.show()
                
                    n_detected_objects = np.shape(r['masks'])[2]

                    print('---------------------------------------------------------')
                    print('Processing crops: '+str(np.round((((n*int(shpimg[1]/cropsize))+(m+1))/n_crops)*100,3))+' % - '+str((n*int(shpimg[1]/cropsize))+(m+1))+' / '+str(n_crops)+' crops processed')
                    print('Total number of detected objects: '+str(total_number_of_detected_objects+n_detected_objects))
                    print('Current crop: '+str(n_detected_objects))

                    if n_detected_objects>0:
                    
                        crop_x = m*cropsize
                        crop_y = -n*cropsize
                    
                        s=np.zeros(n_detected_objects)
                        D=np.zeros(n_detected_objects)

                        ax_list=np.zeros([n_detected_objects,2])
                        for i in range(0, n_detected_objects):
            
                            s[i]=np.sum(r['masks'][:,:,i]*1)*resolution**2
                            D[i]=(2*np.sqrt(s[i]/np.pi))
                            clasts['Surface_area'].iloc[total_number_of_detected_objects+i]=s[i]
                            clasts['Equivalent_diameter'].iloc[total_number_of_detected_objects+i]=D[i]
            
                            clast_i=r['masks'][:,:,i]
                            contours_clast_i=np.asarray(measure.find_contours(clast_i.astype(int),0.99999))
                        
                            if (np.shape(contours_clast_i)[0]==1):
                                x=contours_clast_i[0][:,1]
                                y=contours_clast_i[0][:,0]
                            

                                a = ellipse.fitEllipse(x,y)
                                center = ellipse.ellipse_center(a)
                                phi = ellipse.ellipse_angle_of_rotation(a)
                                axes = ellipse.ellipse_axis_length(a) #results = radius !!
                                R = np.arange(0,2*np.pi, 0.01)
                                a, b = axes
            
                                #Ellipse rotation
                                xx = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
                                yy = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)

                                #Main axis rotation
                                xx_ax1_norot=np.linspace(0,a,100)
                                yy_ax1_norot=np.zeros(np.shape(xx_ax1_norot))
                                xx_ax1_rot=center[0]+xx_ax1_norot*np.cos(phi)+yy_ax1_norot*np.sin(phi)
                                yy_ax1_rot=center[1]+xx_ax1_norot*np.sin(phi)+yy_ax1_norot*np.cos(phi)
    
                                xx_ax2_norot=np.linspace(0,b,100)
                                yy_ax2_norot=np.zeros(np.shape(xx_ax2_norot))
                                xx_ax2_rot=center[0]+xx_ax2_norot*np.cos(phi-(np.pi/2))+yy_ax2_norot*np.sin(phi-(np.pi/2))
                                yy_ax2_rot=center[1]+xx_ax2_norot*np.sin(phi-(np.pi/2))+yy_ax2_norot*np.cos(phi-(np.pi/2))
            
                                ax_list[i,:]=axes
            
                                maj_ax=np.max(ax_list[i,:])
                                min_ax=np.min(ax_list[i,:])
        
                                clasts['clast_ID'].iloc[total_number_of_detected_objects+i]=int(1+total_number_of_detected_objects+i)
                                clasts['x'].iloc[total_number_of_detected_objects+i]=image_x_corner+(crop_x+abs(center[0]))*resolution
                                clasts['y'].iloc[total_number_of_detected_objects+i]=image_y_corner+(crop_y-abs(center[1]))*resolution #y coordinates are upside down
                                clasts['Major_axis'].iloc[total_number_of_detected_objects+i]=abs(maj_ax*resolution*2)  #(*2 for getting diameter values)
                                clasts['Minor_axis'].iloc[total_number_of_detected_objects+i]=abs(min_ax*resolution*2)
                                clasts['Orientation'].iloc[total_number_of_detected_objects+i]=phi
                                clasts['Score'][total_number_of_detected_objects+i]=r['scores'][i]
                        total_number_of_detected_objects=total_number_of_detected_objects+n_detected_objects
        clasts=clasts[clasts.clast_ID > 0]
        if saveresults==True:
            clasts.to_csv(imgpath[0:-4]+'_window_size='+str(metric_cropsize)+'m_individual_clast_values.csv', index=False)
            print(imgpath[0:-4]+'_window_size='+str(metric_cropsize)+'m_individual_clast_values.csv')
    return clasts