# Saliency-Maps-NR-IQA-Classification-Models
**Understanding similarity of saliency maps between NR IQA models and computer vision classification models.**

The term perception is very different for humans and computers. Understanding a scene comes naturally to us without any effort, but for computers, there is no such thing as conscious; it is just a bunch of filters leading to a prediction. We will observe if there is a correlation between the performance of the image classification model and quality rating given by an NR IQA algorithm not only in terms of accuracy but also plotting saliency maps and local patch qualities, respectively. As NR-IQA models are trained on human judgements/perception, they help to understand the similarity in the perception of an image as humans rate it and the classification model while they classify it.

---

## Similarity of Saliency Maps
### Quality-Map and Saliency-Map
Local patch qualities and pixel attribution map of PaQ-2-PiQ and ResNet1 models respectively on a resized image.

![](test_images/severity%3D0/kitchen.png)
![](saliency_maps/paq2piq/severity%3D0/kitchen.png)
![](saliency_maps/resnet18/severity%3D0/kitchen.png)

### Varying the brightness of an image
On brightness the image:

![](test_images/severity%3D2/brightness/kitchen.png)
![](saliency_maps/paq2piq/severity%3D2/brightness/kitchen.png)
![](saliency_maps/resnet18/severity%3D2/brightness/kitchen.png)

### Varying the contrast of an image
On contrast the image:

![](test_images/severity%3D2/contrast/kitchen.png)
![](saliency_maps/paq2piq/severity%3D2/contrast/kitchen.png)
![](saliency_maps/resnet18/severity%3D2/contrast/kitchen.png)

### Varying the saturation of an image
On saturating the image:

![](test_images/severity%3D2/saturate/kitchen.png)
![](saliency_maps/paq2piq/severity%3D2/saturate/kitchen.png)
![](saliency_maps/resnet18/severity%3D2/saturate/kitchen.png)

### Varying the defocus-blur of an image
On defocus-blur the image:

![](test_images/severity%3D2/defocus_blur/kitchen.png)
![](saliency_maps/paq2piq/severity%3D2/defocus_blur/kitchen.png)
![](saliency_maps/resnet18/severity%3D2/defocus_blur/kitchen.png)

### Varying the motion-blur of an image
On motion-blur the image:

![](test_images/severity%3D2/motion_blur/kitchen.png)
![](saliency_maps/paq2piq/severity%3D2/motion_blur/kitchen.png)
![](saliency_maps/resnet18/severity%3D2/motion_blur/kitchen.png)

### Varying the zoom-blur of an image
On zoom-blur the image:

![](test_images/severity%3D2/zoom_blur/kitchen.png)
![](saliency_maps/paq2piq/severity%3D2/zoom_blur/kitchen.png)
![](saliency_maps/resnet18/severity%3D2/zoom_blur/kitchen.png)

### Varying the jpeg-compression of an image
On jpeg-compression the image:

![](test_images/severity%3D2/jpeg_compression/kitchen.png)
![](saliency_maps/paq2piq/severity%3D2/jpeg_compression/kitchen.png)
![](saliency_maps/resnet18/severity%3D2/jpeg_compression/kitchen.png)