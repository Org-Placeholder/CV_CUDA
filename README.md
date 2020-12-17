# CUDA implementations of various Computer Vision Algorithms 

Project's executable for windows can be downloaded [here](https://github.com/Attendo-App/CV_CUDA/releases/tag/v1.0). However, due to discrepancies in OpenCV and CUDA versions,
The executable might not run. In such cases and for other Operating Systems, you can build the project by setting up the Visual Studio project for yourself.

### Algorithms implemented :computer: 
  - Sobel Filter
  - Canny Edge Detection
  - Mean Blur (and its seperable version)
  - Gaussian Blur (and its seperable version)
  - Noise Addition
  - Noise Reduction
  - Bokeh Blur (With three shapes; circle, ring and hexagon; and dynamically decided sizes)

### Visual Studio setup guide :vs:
  - Install CUDA toolkit.
  - Install OpenCV (This is used for reading and displaying images and videos)
  - Setup the corresponding path variables in the Visual Studio project. Follow this [link](https://medium.com/@subwaymatch/adding-opencv-4-2-0-to-visual-studio-2019-project-in-windows-using-pre-built-binaries-93a851ed6141#:~:text=To%20use%20OpenCV%20with%20Visual,%2B%2B%20during%20Visual%20Studio%20install) if you have issues in doing so. 

### Trying it out :heavy_check_mark:
Once built, the executable shall run and present you with all options and instructions needed to get the demonstrations of all filters and algorithms. Hope this helps! :smiley:
