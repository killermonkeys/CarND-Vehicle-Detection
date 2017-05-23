import numpy as np
import matplotlib.image as mpimg
import cv2
import vehicle_detect
import lanes_detect

from moviepy.editor import VideoFileClip
import multiprocessing as mp 
from scipy.ndimage.measurements import label
from moviepy.editor import VideoClip

# detect windows in frames, returns windows list 
# this is used as a map on frames, so doesn't need to know about start_s and end_s
def detect_vehicles_in_frame(image):
    image = image.astype(np.float32)/255
    hot_windows = vehicle_detect.detect_multiScale(image, clf, X_scaler, ystart=400, ystop=None,
                        color_space=color_space, start_scale=1.3, scale=1.3, max_layers=5,
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, cells_per_step=2,
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    
    print('.', end='', flush=True)
    return hot_windows


# get distortion constants to transform frames
import pickle
dist_pickle = pickle.load( open("dist.pickle", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# wraps arguments for lane detection
def detect_lanes_in_frame(image):
    print(',', end='', flush=True)
    return lanes_detect.find_lanes(image, mtx, dist)
    
# used as a generator for frames, needs to know about start_s and end_s (t+start_s becomes an offset)
def make_frame(t):
    image = clip1.get_frame(t)
    frame_no = int(round(t*clip1.fps))
    
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    
    lookback = 6
    if frame_no < lookback:
      lookback = frame_no
    hot_window = []
    for i in range(lookback):
      hot_window.extend(windows[frame_no-i])
    
    # Add heat to each box in box list
    heat = vehicle_detect.add_heat(heat,hot_window)
    # Apply threshold to help remove false positives
    heat = vehicle_detect.apply_threshold(heat,2)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = vehicle_detect.draw_labeled_bboxes(image, labels)
    draw_img = lanes_detect.draw_lines(draw_img, lines[frame_no][0], lines[frame_no][1])
    return draw_img
    



# load pickled trained classifer and normalizer
from sklearn.externals import joblib
clf = joblib.load('clf.pickle')
X_scaler = joblib.load('X_scaler.pickle')

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off


# video... yes I could take this as args but lazy
project_video = 'project_video.mp4'
output_video = project_video.split('.')[0] + '_output.mp4'

#These values are globals to make it easy to debug frame extraction and generation
clip1 = VideoFileClip(project_video)
start_s = 0
end_s = clip1.duration
clip1 = clip1.subclip(start_s,end_s)

if __name__ == '__main__':
    #get the windows
    with mp.Pool(processes=7) as pool:     
        windows = pool.map(detect_vehicles_in_frame, clip1.iter_frames())
        pool.close()
        pool.join()
        pool.terminate()
    #windows_result = pool.map_async(detect_vehicles_in_frame, clip1.iter_frames())
    #windows = windows_result.get()
    print('detection step complete. ', len(windows), ' windows found')
    
    lines = []
    for frame in clip1.iter_frames():
        lines.append(detect_lanes_in_frame(frame))
    print('lines: ', len(lines))
    
    lanes_detect.find_best_lines(lines,1280,720)
    print('best lines complete.')
    clip_out = VideoClip(make_frame, duration=(end_s-start_s)) 
    clip_out.write_videofile(output_video,fps=clip1.fps, threads=4)