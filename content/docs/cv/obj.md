---
title : "Object Detection"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 31
---
# Object Localization
- Object Localization {{< math >}}$ \rightarrow ${{</ math>}} 1 obj
- Object Detection {{< math >}}$ \rightarrow ${{</ math>}} multiple objs

## Bounding Box
Idea: to capture the obj in the img with a box

Params: 
- {{< math >}}$ b_x, b_y ${{</ math>}} = central point
- {{< math >}}$ b_h, b_w ${{</ math>}} = full height/width

New target label (in place of image classification):
$$
y=\begin{bmatrix}
p_c; b_x; b_y; b_h; b_w; c_1; \cdots; c_n
\end{bmatrix}
$$
- {{< math >}}$ p_c ${{</ math>}}: "is there any object in this box?"
    - if {{< math >}}$ p_c=0 ${{</ math>}}, we ignore the remaining params
- {{< math >}}$ c_i$: class label $i$ (e.g. $c_1$: cat, $c_2 ${{</ math>}}: dog, ...)
- {{< math >}}$ n ${{</ math>}}: #classes

## Landmark Detection
Idea: to capture the obj in the img with points

Params: 
- {{< math >}}$ (l_{ix},l_{iy}) ${{</ math>}}: each landmark point

New target label:

$$
y=\begin{bmatrix}p_c; l_{1x}; l_{1y}; \cdots; l_{mx}; l_{my}; c_1; \cdots; c_n\end{bmatrix}
$$
- {{< math >}}$ m ${{</ math>}}: #points

Problem: The labels MUST be consistent!
- Always start from the exact same location of the object!
    - e.g. if you start with the left corner of the left eye for one image, you should always start with the left corner of the left eye for all images.
- #points should always be the same!

## Sliding Window
Idea: to apply a sliding window with a fixed size to scan every part of the img left-right and top-bottom (just like CONV), and feed each part to CNN
- In order to capture the same type of objects in different sizes and positions in the img, shrink the img (i.e. enlarge the sliding window) and scan again, and repeat.

Problem: HUGE computational cost!

Solution: (contemporary)
1. Convert FC layer into CONV layer
2. Share the former FC info with latter convolutions
    1. First run: CNN.
    2. Second run: CNN with bigger size of the same image (due to sliding window).
        - The FC info from the first run is shared in the second run.
    3. Latter runs: CNN with bigger sizes of the same image (due to sliding window).
        - The FC info from all previous runs is shared in this run, thus saving computation power and memories.  

## IoU (Intersection over Union)

Problem: although some boxes capture the object, they may not be of good quality (e.g., too large, too small, etc.)

Solution: use Intersection over Union between prediction box and actual box

$$
\text{IoU}=\frac{\text{area of intersection}}{\text{area of union}}
$$
- If {{< math >}}$ \text{IoU}\geq 0.5 ${{</ math>}}, then the prediction box is correct. (Other threshold values are also okay, but 0.5 was conventional.)

&nbsp;

# 2-Stage Detectors
Idea: Proposal (where) + Checking (what)

## R-CNN
Name: [Regions with CNN Features](https://arxiv.org/pdf/1311.2524.pdf)

Algorithm:
1. Extract {{< math >}}$ \approx ${{</ math>}}2000 bottom-up region proposals from input image.
2. Compute features for each proposal with CNN.
3. Classify each region with class-specific linear SVMs.

## Fast R-CNN
[Link](https://arxiv.org/pdf/1504.08083.pdf)

Improvements:
- Use a RoI (Region of Interest) pooling layer to extract a fixed-length feature vector for each object proposal.
- 2 outputs: softmax probabilities + per-class bounding-box regression offsets.

Pros:
- Higher Mean Average Precision
- Faster region extractor method

### Faster R-CNN
[Link](https://arxiv.org/pdf/1506.01497.pdf)

Improvements:
- Use Region Proposal Network as part of the NN that shares the same convolutional features with the detection network, in place of search algorithms.
- End-to-end training (of both localization and classification), in place of training separate models for region proposal and ojbect detection.

# 1-Stage Detectors
Idea: 

## MultiBox

## YOLO
Name: You Only Look Once

- **Grids**: divide the image into grids & use each grid as a bounding box
    - when {{< math >}}$ p_c=0 ${{</ math>}}, we ignore the entire grid
    - {{< math >}}$ p_c=1$ only when the central point of the object $\in ${{</ math>}} the grid
    - target output: {{< math >}}$ Y.\text{shape}=n_{\text{grid}}\times n_{\text{grid}}\times y.\text{length} ${{</ math>}}  
<br/>
- **Non-Max Suppression**: what happens when the grid is too small to capture the entire object?
    <center><img src="../../images/DL/nms.jpg" width="500"/></center>
    
    1. Discard all boxes with {{< math >}}$ p_c\leq 0.6 ${{</ math>}}
    2. Pick the box with the largest {{< math >}}$ p_c ${{</ math>}} as the prediction
    3. Discard any remaining box with {{< math >}}$ \text{IoU}\geq 0.5 ${{</ math>}} with the prediction
    4. Repeat till there is only one box left.  
<br/>
- **Anchor Boxes**: what happens when two objects overlap? (e.g. a hot girl standing in front of a car)
    <center><img src="../../images/DL/anchor.jpg" width="300"/></center>
    
    1. Predefine Anchor boxes for different objects
    2. Redefine the target value as a combination of Anchor 1 + Anchor 2
    
        $$\begin{equation}
        y=\begin{bmatrix}
        p_{c1} \\
        \vdots \\ 
        p_{c2} \\
        \vdots 
        \end{bmatrix}
        \end{equation}$$
    
    3. Each object in the image is assigned to grid cell that contains object's central point & anchor box for the grid cell with the highest {{< math >}}$ \text{IoU} ${{</ math>}}  
<br/>   
- **General Procedure**:

    1. Divide the images into grids and label the objects
    2. Train the CNN
    3. Get the prediction for each anchor box in each grid cell
    4. Get rid of low probability predictions
    5. Get final predictions through non-max suppression for each class