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
from shapely.geometry import Point,Polygon,MultiPolygon,LineString
from math import degrees,atan2,cos,sin

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
from functions import ellipse


def get_ax(rows=1, cols=1, size=16):

    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
    
def matrix2xyz(matrix):
    m = np.tile(range(0,np.shape(matrix)[1]),[np.shape(matrix)[0],1])
    n = np.transpose(np.tile(range(0,np.shape(matrix)[0]),[np.shape(matrix)[1],1]))
    xyz = np.transpose([n[matrix==matrix],m[matrix==matrix],matrix[matrix==matrix]])
    return xyz;

def clasts_detect(mode, imgpath, resolution=0.001, metric_cropsize=1, plot=True, saveplot=False, saveresults=False, devicemode="gpu", devicenumber=0, kstart=0, ksaveint=1):

    """Detects and measures the clasts on a scaled image that can be either terrestrial (i.e. photograph of known resolution) or UAV derived (georeferenced ortho-image)
    mode: "terrestrial", "UAV"
    resolution: resolution of a terrestrial image (default = 0.001 m/pixel)
    imgpath: full path of the image/ortho-image
    metric_cropsize: Window size used for croping a UAV image into tiles (default = 1m)
    plot: (boolean) Display detections and histograms
    saveplot: (boolean) Save the detection and histogram figures
    saveresults: (boolean) Save the clast sizes into a CSV file
    devicemode: "gpu", "cpu" (default = "gpu") (see tesorflow-gpu documentation)
    devicenumber: id number of the device to be used (default = 0)
    kstart: Starting itteration step, (in case of resuming a previously stopped processing)(default = 0)
    ksaveint: Intervalle step between 2 checkpoint saving (default = 1%)
    """
    DEVICE = "/"+str.lower(devicemode)+":"+str(devicenumber)  # /cpu:0 or /gpu:0
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
            clasts=pd.DataFrame(np.zeros([n_detected_objects,10]))
            clasts.columns=['clast_ID', 'x', 'y', 'Ellipse_major_axis', 'Ellipse_minor_axis', 'Clast_length', 'Clast_width', 'Surface_area', 'Score', 'Orientation']
            if plot==True:
                ax = get_ax(1)
                visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                            class_names, r['scores'], ax=ax,
                                            title="Predictions")
                fig = plt.gcf()
                fig.set_size_inches(16, 16)
                plt.show()
            s=np.zeros(n_detected_objects)
    
            if plot==True:
                plt.imshow(image)
#                 ax_list=np.zeros([n_detected_objects,2])
            for i in range(0, n_detected_objects):
    
                s[i]=np.sum(r['masks'][:,:,i]*1)*resolution**2
                clasts['Surface_area'].iloc[i]=s[i]
    
                clast_i=r['masks'][:,:,i]
                contours_clast_i=np.asarray(measure.find_contours(clast_i.astype(int),0.99999))
                x=contours_clast_i[0][:,1]
                y=contours_clast_i[0][:,0]
                aa = ellipse.fit_ellipse(x,y)
                center = ellipse.ellipse_center(aa)
                phi = ellipse.ellipse_angle_of_rotation(aa)
                axes = ellipse.ellipse_axis_length(aa) #results = radius !!
                a, b = axes
    
                if plot==True:
                    R = np.arange(0,2*np.pi, 0.01)
                
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

                #Extended axes for real size measurements using intersections
                xx_ax1_norot_plus=np.linspace(-a*10,a*10,100)
                yy_ax1_norot_plus=np.zeros(np.shape(xx_ax1_norot_plus))
                xx_ax1_rot_plus=center[0]+xx_ax1_norot_plus*np.cos(phi)+yy_ax1_norot_plus*np.sin(phi) #direction 1
                yy_ax1_rot_plus=center[1]+xx_ax1_norot_plus*np.sin(phi)+yy_ax1_norot_plus*np.cos(phi) #direction 2
    
                xx_ax2_norot_plus=np.linspace(-b*10,b*10,100)
                yy_ax2_norot_plus=np.zeros(np.shape(xx_ax2_norot_plus))
                xx_ax2_rot_plus=center[0]+xx_ax2_norot_plus*np.cos(phi-(np.pi/2))+yy_ax2_norot_plus*np.sin(phi-(np.pi/2)) #direction 1
                yy_ax2_rot_plus=center[1]+xx_ax2_norot_plus*np.sin(phi-(np.pi/2))+yy_ax2_norot_plus*np.cos(phi-(np.pi/2))#direction 2
    
                polygon = list(set(zip(x,y)))
                cent=(sum([p[0] for p in polygon])/len(polygon),sum([p[1] for p in polygon])/len(polygon))
                polygon.sort(key=lambda p: math.atan2(p[1]-cent[1],p[0]-cent[0]))
    
                p1 = Polygon(polygon)
                p1 = p1.buffer(0)

                line_ax1 = [(xx_ax1_rot_ext[0], yy_ax1_rot_ext[0]), (xx_ax1_rot_ext[-1], yy_ax1_rot_ext[-1])]
                p2_ax1 = LineString(line_ax1)
                
                line_ax2 = [(xx_ax2_rot_ext[0], yy_ax2_rot_ext[0]), (xx_ax2_rot_ext[-1], yy_ax2_rot_ext[-1])]
                p2_ax2 = LineString(line_ax2)
                
                if (p1.is_valid & p2_ax1.is_valid & p2_ax2.is_valid):
                    intersection_line_ax1 = p1.intersection(p2_ax1)
                    intersection_line_ax2 = p1.intersection(p2_ax2)
                    length = intersection_line_ax1.length
                    width = intersection_line_ax2.length
                else:
                    length = math.nan
                    width = math.nan

                #Storing measurements
                clasts['clast_ID'].iloc[i]=int(i)
                clasts['x'].iloc[i]=abs(center[0])
                clasts['y'].iloc[i]=np.shape(image)[1]-abs(center[1]) #y coordinates are upside down
                clasts['Ellipse_major_axis'].iloc[i]=abs(axes[0]*resolution*2) #(*2 for getting diameter values)
                clasts['Ellipse_minor_axis'].iloc[i]=abs(axes[1]*resolution*2)
                clasts['Clast_length'].iloc[i]=length*resolution
                clasts['Clast_width'].iloc[i]=width*resolution
                clasts['Orientation'].iloc[i]=np.rad2deg(phi)+90

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

            if plot==True:
                plt.hist(clasts['Clast_length']*100, 20)
                plt.xlabel('Grain Size (cm)')
                plt.ylabel('Number of clasts')
                fig = plt.gcf()
                fig.set_size_inches(10, 10)
                if saveplot==True:
                    fig.savefig(imgpath[0:-4]+'_histogram.png', dpi=100)
                plt.show()

            print(' ')
            print('Detected clasts:', np.shape(r['masks'])[2])
            print(' ')
            print('Ellipse Major Axis D10 = '+ str(round(np.quantile(clasts['Ellipse_major_axis'], 0.1)*100,2))+' cm')
            print('Ellipse Major Axis D50 = '+ str(round(np.quantile(clasts['Ellipse_major_axis'], 0.5)*100,2))+' cm')
            print('Ellipse Major Axis D90 = '+ str(round(np.quantile(clasts['Ellipse_major_axis'], 0.9)*100,2))+' cm')
            print(' ')
            print('Ellipse Minor Axis D10 = '+ str(round(np.quantile(clasts['Ellipse_minor_axis'], 0.1)*100,2))+' cm')
            print('Ellipse Minor Axis D50 = '+ str(round(np.quantile(clasts['Ellipse_minor_axis'], 0.5)*100,2))+' cm')
            print('Ellipse Minor Axis D90 = '+ str(round(np.quantile(clasts['Ellipse_minor_axis'], 0.9)*100,2))+' cm')
            print(' ')
            print('Clast Length D10 = '+ str(round(np.quantile(clasts['Clast_length'], 0.1)*100,2))+' cm')
            print('Clast Length D50 = '+ str(round(np.quantile(clasts['Clast_length'], 0.5)*100,2))+' cm')
            print('Clast Length D90 = '+ str(round(np.quantile(clasts['Clast_length'], 0.9)*100,2))+' cm')
            print(' ')
            print('Clast Width D10 = '+ str(round(np.quantile(clasts['Clast_width'], 0.1)*100,2))+' cm')
            print('Clast Width D50 = '+ str(round(np.quantile(clasts['Clast_width'], 0.5)*100,2))+' cm')
            print('Clast Width D90 = '+ str(round(np.quantile(clasts['Clast_width'], 0.9)*100,2))+' cm')
            print(' ')
            print('-------------------------------------------------------')
            print(' ')
            if saveresults==True:
                clasts.to_csv(imgpath[0:-4]+'individual_clasts.csv', index=False, float_format='%.5f')

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
        clasts=pd.DataFrame(np.zeros([expected_maximum_number_of_detected_objects,10]))
        clasts.columns=['clast_ID', 'x', 'y', 'Ellipse_major_axis', 'Ellipse_minor_axis', 'Clast_length', 'Clast_width', 'Surface_area', 'Score', 'Orientation']

        work_tiles_mat = np.zeros([int(shpimg[0]/cropsize)+1, int(shpimg[1]/cropsize)+1])
        for n in range(0, int(shpimg[0]/cropsize)+1): #Lines
            for m in range(0, int(shpimg[1]/cropsize)+1): #Columns
                croppedimage=image[n*cropsize:n*cropsize+cropsize, m*cropsize:m*cropsize+cropsize,:]
                if (np.shape(croppedimage[croppedimage!=255])[0]==0):
                    work_tiles_mat[n,m]=0
                else:
                    work_tiles_mat[n,m]=1
        work_tiles_nm = matrix2xyz(work_tiles_mat)
        work_tiles_nm = work_tiles_nm[work_tiles_nm[:,2]==1,:]
        k_save = np.asarray(range(int(((np.shape(work_tiles_nm)[0]-1)/100)*ksaveint),np.shape(work_tiles_nm)[0]-int(((np.shape(work_tiles_nm)[0]-1)/100)*ksaveint)+1,int(((np.shape(work_tiles_nm)[0]-1)/100)*ksaveint)))
          
        print('List of checkpoint k: '+str(k_save))
        
        for k in range(kstart, np.shape(work_tiles_nm)[0]):
            n = int(work_tiles_nm[k,0])
            m = int(work_tiles_nm[k,1])
            
            croppedimage=image[n*cropsize:n*cropsize+cropsize, m*cropsize:m*cropsize+cropsize,:]
        
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
            print('Processing crops: '+'row = '+str(int(n))+', col = '+str(int(m))+' - '+str(np.round(((k+1)/(np.shape(work_tiles_nm)[0]+1))*100,3))+' % - '+str(int(k+1))+' / '+str(np.shape(work_tiles_nm)[0]+1)+' crops processed')
            print('Total number of detected objects: '+str(total_number_of_detected_objects+n_detected_objects))
            print('Current crop: '+str(n_detected_objects))
        
            if n_detected_objects>0:
        
                crop_x = m*cropsize
                crop_y = -n*cropsize
        
                s=np.zeros(n_detected_objects)
        
                for i in range(0, n_detected_objects):
        
                    s[i]=np.sum(r['masks'][:,:,i]*1)*resolution**2
                    clasts['Surface_area'].iloc[total_number_of_detected_objects+i]=s[i]
        
                    clast_i=r['masks'][:,:,i]
                    contours_clast_i=np.asarray(measure.find_contours(clast_i.astype(int),0.99999))
        
                    if (np.shape(contours_clast_i)[0]==1):
                        x=contours_clast_i[0][:,1]
                        y=contours_clast_i[0][:,0]
        
                        a = ellipse.fit_ellipse(x,y)
                        center = ellipse.ellipse_center(a)
                        phi = ellipse.ellipse_angle_of_rotation(a)
                        axes = ellipse.ellipse_axis_length(a) #results = radius !!
                        a, b = axes
        
                        #Extended axes for real size measurements using intersections
                        xx_ax1_norot_ext=np.linspace(-a*10,a*10,100)
                        yy_ax1_norot_ext=np.zeros(np.shape(xx_ax1_norot_ext))
                        xx_ax1_rot_ext=center[0]+xx_ax1_norot_ext*np.cos(phi)+yy_ax1_norot_ext*np.sin(phi) #direction 1
                        yy_ax1_rot_ext=center[1]+xx_ax1_norot_ext*np.sin(phi)+yy_ax1_norot_ext*np.cos(phi) #direction 2

                        xx_ax2_norot_ext=np.linspace(-b*10,b*10,100)
                        yy_ax2_norot_ext=np.zeros(np.shape(xx_ax2_norot_ext))
                        xx_ax2_rot_ext=center[0]+xx_ax2_norot_ext*np.cos(phi-(np.pi/2))+yy_ax2_norot_ext*np.sin(phi-(np.pi/2)) #direction 1
                        yy_ax2_rot_ext=center[1]+xx_ax2_norot_ext*np.sin(phi-(np.pi/2))+yy_ax2_norot_ext*np.cos(phi-(np.pi/2))#direction 2
        
                        polygon = list(set(zip(x,y)))
                        cent=(sum([p[0] for p in polygon])/len(polygon),sum([p[1] for p in polygon])/len(polygon))
                        polygon.sort(key=lambda p: math.atan2(p[1]-cent[1],p[0]-cent[0]))
            
                        p1 = Polygon(polygon)
                        p1 = p1.buffer(0)
        
                        line_ax1 = [(xx_ax1_rot_ext[0], yy_ax1_rot_ext[0]), (xx_ax1_rot_ext[-1], yy_ax1_rot_ext[-1])]
                        p2_ax1 = LineString(line_ax1)
        
                        line_ax2 = [(xx_ax2_rot_ext[0], yy_ax2_rot_ext[0]), (xx_ax2_rot_ext[-1], yy_ax2_rot_ext[-1])]
                        p2_ax2 = LineString(line_ax2)
        
                        if (p1.is_valid & p2_ax1.is_valid & p2_ax2.is_valid):
                            intersection_line_ax1 = p1.intersection(p2_ax1)
                            intersection_line_ax2 = p1.intersection(p2_ax2)
                            length = intersection_line_ax1.length
                            width = intersection_line_ax2.length
                        else:
                            length = math.nan
                            width = math.nan
        
                        #Storing measurements
                        clasts['clast_ID'].iloc[total_number_of_detected_objects+i]=int(total_number_of_detected_objects+i)
                        clasts['x'].iloc[total_number_of_detected_objects+i]=image_x_corner+(crop_x+abs(center[0]))*resolution
                        clasts['y'].iloc[total_number_of_detected_objects+i]=image_y_corner+(crop_y-abs(center[1]))*resolution #y coordinates are upside down
                        clasts['Ellipse_major_axis'].iloc[total_number_of_detected_objects+i]=abs(axes[0]*resolution*2) #(*2 for getting diameter values)
                        clasts['Ellipse_minor_axis'].iloc[total_number_of_detected_objects+i]=abs(axes[1]*resolution*2)
                        clasts['Clast_length'].iloc[total_number_of_detected_objects+i]=length*resolution
                        clasts['Clast_width'].iloc[total_number_of_detected_objects+i]=width*resolution
                        clasts['Orientation'].iloc[total_number_of_detected_objects+i]=np.rad2deg(phi)+90
                        clasts['Score'][total_number_of_detected_objects+i]=r['scores'][i]
                total_number_of_detected_objects=total_number_of_detected_objects+n_detected_objects
            if not not(k_save[k_save==k]):
                if saveresults==True:
                    clasts_temp=clasts.copy()
                    clasts_temp=clasts_temp[clasts_temp.clast_ID > 0]
                    clasts_temp.to_csv(imgpath[0:-4]+'_checkpoint_save_k='+str(k)+'_window_size='+str(metric_cropsize)+'m_individual_clast_values.csv', index=False, float_format='%.5f')
                    print(imgpath[0:-4]+'_checkpoint_save_k='+str(k)+'_window_size='+str(metric_cropsize)+'m_individual_clast_values.csv')
        clasts=clasts[clasts.clast_ID > 0]
        if saveresults==True:
            clasts.to_csv(imgpath[0:-4]+'_window_size='+str(metric_cropsize)+'m_individual_clast_values.csv', index=False, float_format='%.5f')
            print(imgpath[0:-4]+'_window_size='+str(metric_cropsize)+'m_individual_clast_values.csv')
    return clasts