## Writeup for Vehicle Detection Project
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image11]: ./output_images/grid1.png
[image12]: ./output_images/detectmultiscale1.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 2nd code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I used GDDI and KDDI images, with 8792 vehicles and 8968 non-vehicles.


#### 2. Explain how you settled on your final choice of HOG parameters.

I am familiar with HOG features from previous work and had already used HOG during the lesson. So I took my code from the lessons and moved it here. In the lesson I used: color_space=YCrCb, channels=all, orient=9, pix_per_cell=16, cell_per_block=2. I also used histogram features of 16,16 spatial binning with 16 bins. 

All this code was based on the code and library functions from the lesson.

After going through the pipeline a few times, I increased the number of orientations to 11 because I was getting too many false negatives.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The 3rd cell of the notebook has the training step. I trained the classifier using a grid search ranging from 'C':[0.001, 0.01, 1, 10, 100, 1000] and a test/train split of 0.2 with (default) 3-fold cross validation. The total feature vector length was 2000.

After training the classifier several times, I stopped using the gridsearch and started using just C=10. The final accuracy for the classifier was 0.9963, which is extremely good performance.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The 7th code cell has my singlescale and mutltiscale code. For the singlescale, I used HOG subsampling and scaled the image to match the size of the detection window 

For multiscale I used an image pyramid with a starting scale of 1.3 and a scale factor of 1.3, with a limit of 5 layers. This meant that the first layer had a window size of 83x83 px, then 108x108px, 141x141px, 183x183px, 237x237px. Here is the smallest set of windows:

![alt text][image11]

As you can see in the multiscale implementation, the windows stop at the right and bottom edges if they cannot tile across. Ideally they would not do this, but I did not change the implementation because I did not feel that the right and bottom edge were likely to have vehicles that needed to be tracked. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here is the output from my classifier.

![alt text][image12]

As you can see, it misses the right edge. 

The majority of the work I did to optimize the classifer was to parallelize the detection step in the video version, but even so my implementation was not very fast. This is likely because I was using a multiscale approach that would have high accuracy but slowly. To increase speed, the most important optimzation would be to reduce the number of scales, followed by a better implementation of HoG such as OpenCV's. 

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
My pipeline code is in detect_pipeline.py, this integrates the lane detection and vehicle detection code, and is parallelized. 

Here's a [link to my video result](./project_video_output.mp4)

I have no false positives, although I do occasionally detect vehicles in the oncoming traffic, these are not false. I also see false negatives at the frame edges as discussed. It takes roughly 3 minutes to generate the video on my machine. This is obviously post-processed, so streaming frames would be a different task.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

For each frame, I recorded the detection bounding boxes (along with generating the lane detection data). I then iterate over the generated data and have a 6 frame lookback, summing the hot windows for the previous 6 frames. I create a heatmap of these and threshold it to determine vehicle positions in that frame. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

This is the reference implementation from the lesson. The main changes I made are to parallelize the detection step using Python multiprocessing.  


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest issue was the speed. I spent a lot of time optimizing the speed within python but I believe that given the approach a CNN would be faster with nearly equal accuracy to a SVM. Because vehicle detection is inherently multiscale, it's important that it can process across the whole image in near-real time. The fastest processing I saw was approximately 7fps, which is required all the compute power of a high end processor. This included the lane finding step as well, but it is still not as fast as CNNs. 

I also ran my pipeline with the other videos from the lane finding lesson. The challenge video was reasonably OK but the harder challenge video caused many false positives, mostly due to the glare, light/shadow, and trees. Because this data was so noisy, it points out the importance of good input data.