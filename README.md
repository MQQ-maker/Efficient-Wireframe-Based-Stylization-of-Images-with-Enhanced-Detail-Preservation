# Efficient-Wireframe-Based-Stylization-of-Images-with-Enhanced-Detail-Preservation
# 1.Steps for Wireframe Stylization of Grayscale Images:  
Input a grayscale image from the dataset, and an example is shown below.
image_lena = cv2.imread('peppers_gray.tif', cv2.IMREAD_GRAYSCALE)
Run gray_picture.py.

# 2.Wireframe Stylization of Color Images: 
Use a color extraction algorithm to extract the main colors from the color image and determine the drawing colors.  
After determining the colors, input the color image and the hexadecimal color codes into color_picture.py.An example is shown below：
image_lena_color = cv2.imread('PeppersRGB.tif')
Run color_picture.py.


