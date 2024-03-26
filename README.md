# LidarPrior4GaussianSplatting
Using prior LiDAR point cloud information to enhance the accuracy and robustness of Gaussian Splatting
## Setup

Our default, provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate gaussian_splatting
```
I suggest using CUDA 11.8 and manually installing these two items:
```shell
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```
## The experimental results
### Underground Parking Lot A
From left to right: Ground truth of the ground, model rendered image, LiDAR point cloud to image projection.
| ![GT](readme_images/case1/1280X1280.JPEG) | ![Model Prediction](readme_images/case1/1280X1280%20(1).JPEG) | ![Point Cloud to Image Projection](readme_images/case1/1280X1280%20(2).JPEG) |
| --- | --- | --- |
| ![GT](readme_images/case1/1280X1280%20(3).JPEG) | ![Model Prediction](readme_images/case1/1280X1280%20(4).JPEG) | ![Point Cloud to Image Projection](readme_images/case1/1280X1280%20(5).JPEG) |
|  |  |  |
| ![GT](readme_images/case1/1280X1280%20(6).JPEG) | ![Model Prediction](readme_images/case1/1280X1280%20(7).JPEG) | ![Point Cloud to Image Projection](readme_images/case1/1280X1280%20(8).JPEG) |
|  |  |  |
| ![GT](readme_images/case1/1280X1280%20(9).JPEG) | ![Model Prediction](readme_images/case1/1280X1280%20(10).JPEG) | ![Point Cloud to Image Projection](readme_images/case1/1280X1280%20(11).JPEG) |
|  |  |  |
| ![GT](readme_images/case1/1280X1280%20(12).JPEG) | ![Model Prediction](readme_images/case1/1280X1280%20(13).JPEG) | ![Point Cloud to Image Projection](readme_images/case1/1280X1280%20(14).JPEG) |
|  |  |  |
| ![GT](readme_images/case1/1280X1280%20(15).JPEG) | ![Model Prediction](readme_images/case1/1280X1280%20(16).JPEG) | ![Point Cloud to Image Projection](readme_images/case1/1280X1280%20(17).JPEG) |
|  |  |  |
### Underground Parking Lot B
| ![GT](readme_images/case2/1.jpeg) | ![Model Prediction](readme_images/case2/2.jpeg) | ![Point Cloud to Image Projection](readme_images/case2/3.jpeg) |
| --- | --- | --- |
| ![GT](readme_images/case2/4.jpeg) | ![Model Prediction](readme_images/case2/5.jpeg) | ![Point Cloud to Image Projection](readme_images/case2/6.jpeg) |
|  |  |  |
| ![GT](readme_images/case2/7.jpeg) | ![Model Prediction](readme_images/case2/8.jpeg) | ![Point Cloud to Image Projection](readme_images/case2/9.jpeg) |
|  |  |  |
| ![GT](readme_images/case2/10.jpeg) | ![Model Prediction](readme_images/case2/11.jpeg) | ![Point Cloud to Image Projection](readme_images/case2/12.jpeg) |
|  |  |  |
| ![GT](readme_images/case2/13.jpeg) | ![Model Prediction](readme_images/case2/14.jpeg) | ![Point Cloud to Image Projection](readme_images/case2/15.jpeg) |
|  |  |  |
| ![GT](readme_images/case2/16.jpeg) | ![Model Prediction](readme_images/case2/17.jpeg) | ![Point Cloud to Image Projection](readme_images/case2/18.jpeg) |
|  |  |  |
