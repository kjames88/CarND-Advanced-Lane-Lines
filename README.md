##Advanced Lane Lines##

###Methodology###

The pipeline used is based on the lectures, with tuning used to get the desired results.  The following steps are used:

1. Calibrate the camera using provided checkerboard images and OpenCV corner finding and distortion computation
2. Compute the perspective transform by identifying four points on a test image (car on freeway) and specifying a corresponding set of points in an undistored view then using OpenCV to generate the warping matrix
3. Start processing video frames as below
4. Undistort the frame
5. Switch color space to HLS
6. Threshold S channel
7. Compute gradient using grayscale
8. Combine gradient angle and absolute Sobel y direction gradient with S channel to generate a binary image that exposes the lane lines in the frame
9. Warp the binary image to create an overhead view such that lane lines are approximately parallel rather than converging
10. Identify left and right lane line points starting from a pixel histogram over image columns
11. Search horizontal windows around the histogram peaks and take the average x position of set pixels (1 values)
12. Repeat this process in vertical windows, adjusting the center x position of each lane when sufficient pixels are found
13. Fit a degree 2 polynomial to the identified points for each lane line
14. Use the previously identified lane line polynomials to guide search for pixels in subsequent frames
15. If the above optimization fails to identify a line then revert to the histogram-based search in the next frame
16. Compute radius of curvature for the polynomial fit lines
17. Color identified lane line pixels in a new image
17. Draw a filled polygon corresponding to the lane lines
18. Inverse transform the new image and overlay on the original frame

####Calibration####

![Uncalibrated]
(images/uncalibrated.png)

![Calibrated]
(images/calibrated_chessboard.png)

####Binary Image Generation####

![Color]
(images/color.png)

![Binary]
(images/binary.png)

####Image Warping####

![Warped]
(images/warped.png)

####Finding Lines####

![Lines]
(images/lines.png)

####Lines on Road####

![Road]
(images/overlay1.png)

####Complete Frame####

![Final]
(images/final.png)

###Notes and Refinements###

The process outlined above corresponds to the lectures.  Following are some specifics involved in making the project functional and some findings along the way.

####Binary Image Generation####

The S channel in HLS color space works well as a starting point.  I thresholded this high enough to remove most of the scenery while leaving most of the lane lines.  While the S channel is the primary detector, a Red channel threshold is ANDd with the S channel result.  This modification was made to prevent shadows from registering in the final result.  In the interest of preserving lane-like lines, gradient analysis is also used.  Sobel x and y gradients are computed on grayscale.  Arctan is then used to compute gradient angle (in this case a large kernel is used in the gradient function).  I found that this output needed to be thresholded to a narrow range to avoid adding too much noise.  I also mixed in a thresholded y-gradient (small kernel) to further limit the retained pixels to more vertical lines.  The two Sobel results are ANDd and then ORd with the S and R channels to create the binary image used for lane line detection. 

####Perspective Transform####

It took a few iterations to get a decent result on the perspective transform.  While it is easy enough to identify four points on the camera image, deciding on appropriate correspondence locations in the warped image is not so easy.  I first used a long range in y (down the road), I found that the warped image lost focus significantly on the upper half of the image.  This gave the effect of diverging lines, and was not consistent with expectations.  After trying to choose coordinates more exactly, I finally shortened the y range and then carefully recorded coordinates on a zoomed image.  I kept the target scale consistent but did not shift the polygon within the target frame.  That is, I used the bottom edge as an anchor and then squared out the sides.  Despite starting with the same idea, it took a few attempts to get a good warped result.

####Lane Following####

There are two tricky spots on the test video that cause the lane detection to go out of bounds.  While the fallback to use the histogram and sliding window method was always present, I had to add a sanity check on the optimized search to ensure that it returned a reasonable number of candidate pixels and that the vertical span of those pixels was sufficient to suggest a line segment.  Otherwise, lines would be detected, preventing the fallback from activating, even though the "lines" were not usable.

####Lane Line and Radius Smoothing####

I added a small filter to average the lane line polynomial parameters and the computed radius for each line.  This makes the lane identifying polygon smoother and keeps the radius results in a better range.

###Results###

The lane is tracked reasonably well.  The computed radius values jump around too much and are sometimes quite different between left and right (displayed is averaged).  More work may be needed on the radius, although smoothing the line parameters helps.  The magnitude is within reason, but the result is not satisfying (yet).

###Pitfalls###

This pipeline can be expected to have difficulty if the road surface is very uneven.  The perspective transform distorts when the car's orientation to the road is not consistent with the designed-in correspondence.  The effect is observed over the first light concrete stripe.  Prior to readjusting the correspondence, the lane projection was severely distorted, and analysis showed major dispersion of pixels toward the top of the frame.  This fix improved the precision of the coordinate selection, but if the road surface turns sharply up or down, failure of the transform can be expected.






