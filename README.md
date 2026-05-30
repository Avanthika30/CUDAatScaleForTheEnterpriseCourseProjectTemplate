# GPU Accelerated Image Processing Suite Using CUDA on Google Colab

## AIM:
To develop a GPU-accelerated image processing application using CUDA and Python that performs grayscale conversion and Sobel edge detection, and to compare the execution performance of CPU and GPU implementations.

## Concept

Graphics Processing Units (GPUs) are designed to execute thousands of parallel threads simultaneously, making them highly suitable for image processing tasks. Traditional CPU-based image processing executes operations sequentially or with limited parallelism, whereas GPU computing enables large-scale parallel execution.

In this project, CUDA is used through the Numba library in Python to accelerate image processing operations. The image is processed using both CPU and GPU implementations, and the execution times are compared to evaluate the performance benefits of GPU computing.

### The project demonstrates the following operations:

RGB to Grayscale Conversion
Sobel Edge Detection
CPU vs GPU Performance Comparison
Software and Hardware Requirements
Software
Python 3
Google Colab
CUDA-enabled Runtime
Numba
OpenCV
NumPy
Pandas
Matplotlib
Hardware
NVIDIA GPU provided by Google Colab
## Procedure
1.Enable GPU runtime in Google Colab.

2.Install required Python libraries.

3.Upload the input image.

4.Load the image using OpenCV.

5.Convert the image to grayscale using CPU implementation.

6.Create a CUDA kernel for grayscale conversion.

7.Execute grayscale conversion on the GPU.

8.Compare CPU and GPU execution times.

9.Perform Sobel edge detection on the CPU.

10.Create a CUDA kernel for Sobel edge detection.
11.Execute Sobel edge detection on the GPU.

12.Measure and compare execution times.

13.Save the generated output images.

14.Store timing results in a CSV file.

15.Analyze the performance improvement obtained using GPU acceleration.

## PROGRAM: 

```
!pip install numba opencv-python
```
```
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numba import cuda
import math
import time
```
### Upload image:
```
from google.colab import files

uploaded = files.upload()
```
```
img = cv2.imread("lena.jpg")

img_rgb = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.axis("off")
```
### GPU Grayscale Kernel:
```
@cuda.jit
def grayscale_kernel(
    input_img,
    output_img):

    x, y = cuda.grid(2)

    if x < input_img.shape[0] and y < input_img.shape[1]:

        r = input_img[x,y,0]
        g = input_img[x,y,1]
        b = input_img[x,y,2]

        gray = (
            0.299*r +
            0.587*g +
            0.114*b
        )

        output_img[x,y] = gray
```
### CPU Grayscale:
```
start = time.time()

gray_cpu = cv2.cvtColor(
    img_rgb,
    cv2.COLOR_RGB2GRAY)

cpu_gray_time = (
    time.time() - start
) * 1000

print(cpu_gray_time,"ms")
```
### GPU Grayscale:
```
gpu_input = cuda.to_device(img_rgb)

gpu_output = cuda.device_array(
    (img_rgb.shape[0],
     img_rgb.shape[1]),
     dtype=np.uint8)

threads = (16,16)

blocks = (
    math.ceil(img_rgb.shape[0]/16),
    math.ceil(img_rgb.shape[1]/16)
)

start = time.time()

grayscale_kernel[
    blocks,
    threads
](
    gpu_input,
    gpu_output
)

cuda.synchronize()

gpu_gray_time = (
    time.time() - start
) * 1000

gray_gpu = gpu_output.copy_to_host()

print(gpu_gray_time,"ms")
```
### Display Results:
```
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(gray_cpu,cmap='gray')
plt.title("CPU")

plt.subplot(1,2,2)
plt.imshow(gray_gpu,cmap='gray')
plt.title("GPU")

plt.show()
```
### Sobel CPU:
```
start = time.time()

sobel_cpu = cv2.Sobel(
    gray_cpu,
    cv2.CV_8U,
    1,
    1)

cpu_sobel_time = (
    time.time()-start
)*1000

print(cpu_sobel_time)
```
### GPU Sobel Kernel
```
@cuda.jit
def sobel_kernel(
    input_img,
    output_img):

    x,y = cuda.grid(2)

    if (1 <= x < input_img.shape[0]-1 and
        1 <= y < input_img.shape[1]-1):

        gx = (
            -input_img[x-1,y-1]
            -2*input_img[x,y-1]
            -input_img[x+1,y-1]
            +input_img[x-1,y+1]
            +2*input_img[x,y+1]
            +input_img[x+1,y+1]
        )

        gy = (
            -input_img[x-1,y-1]
            -2*input_img[x-1,y]
            -input_img[x-1,y+1]
            +input_img[x+1,y-1]
            +2*input_img[x+1,y]
            +input_img[x+1,y+1]
        )

        output_img[x,y] = min(
            255,
            int(math.sqrt(
                gx*gx+gy*gy
            ))
        )
```
### Run Sobel GPU:
```
gpu_gray = cuda.to_device(gray_gpu)

gpu_edge = cuda.device_array_like(
    gray_gpu)

start = time.time()

sobel_kernel[
    blocks,
    threads
](
    gpu_gray,
    gpu_edge
)

cuda.synchronize()

gpu_sobel_time = (
    time.time()-start
)*1000

sobel_gpu = gpu_edge.copy_to_host()

print(gpu_sobel_time)
```
### Save Outputs:
```
cv2.imwrite(
    "gray_cpu.png",
    gray_cpu)

cv2.imwrite(
    "gray_gpu.png",
    gray_gpu)

cv2.imwrite(
    "sobel_cpu.png",
    sobel_cpu)

cv2.imwrite(
    "sobel_gpu.png",
    sobel_gpu)
```
## OUTPUT:
<img width="515" height="350" alt="image" src="https://github.com/user-attachments/assets/f275f4bc-9c66-4372-b580-2604db026415" />

<img width="831" height="300" alt="image" src="https://github.com/user-attachments/assets/3923afef-1786-4fe3-9e75-25402b9fb162" />

<img width="376" height="135" alt="image" src="https://github.com/user-attachments/assets/f27d0597-4d67-4b82-bd6d-bf357c042925" />

## Conclusion

This project successfully demonstrated the use of GPU computing for image processing using CUDA on Google Colab. Grayscale conversion and Sobel edge detection were implemented and executed using parallel processing techniques. The project highlighted how GPUs can process image data more efficiently by utilizing thousands of parallel threads compared to traditional CPU execution.

The implementation showed that CUDA-based processing is suitable for computationally intensive image operations and can significantly reduce execution time for large datasets. The project also provided practical experience with GPU programming, CUDA kernels, memory management, and performance analysis in a cloud-based environment.

Overall, the project validates the effectiveness of GPU acceleration for image processing applications and demonstrates the potential of CUDA-enabled computing for real-time image analysis, computer vision, scientific computing, and high-performance computing tasks.

## Result

The GPU-Accelerated Image Processing Suite was successfully implemented using CUDA and Python on Google Colab. The application performed grayscale conversion and Sobel edge detection on input images and generated the corresponding output images. CPU and GPU execution times were recorded and compared, demonstrating the advantages of parallel processing using GPU hardware. The generated outputs and performance measurements confirm the successful execution of the project objectives.

