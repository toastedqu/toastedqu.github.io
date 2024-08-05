---
title : "Models"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: true
images: []
weight: 200
---
!-- # Convolutional Neural Networks {#cnn} -->
## Basics of CNN

- <a name="cnn"></a>**Intuition of CNN**
    
    <center><img src="../../images/DL/cnn.gif" width="500"/></center>
    <br/>
    - CNN is mostly used in Computer Vision (image classification, object detection, neural style transfer, etc.)  
    
    - **Input**: images $ \rightarrow$ volume of numerical values in the shape of **width $\times$ height $\times$ color-scale** (color-scale=3 $\rightarrow$ RGB; color-scale=1 $\rightarrow $ BW)  
    
        In the gif above, the input shape is $ 5\times5\times3$, meaning that the image is colored and the image size $5\times5$. The "$7\times7\times3 $" results from **padding**, which will be discussed below.
    
    - **Convolution**: 
        1. For each color layer of the input image, we apply a 2d **filter** that **scans** through the layer in order.
        2. For each block that the filter scans, we **multiply** the corresponding filter value and the cell value, and we **sum** them up.
        3. We **sum** up the output values from all layers of the filter (and add a bias value to it) and **output** this value to the corresponding output cell. 
        4. (If there are multiple filters, ) After the first filter finishes scanning, the next filter starts scanning and outputs into a new layer.  
    <br/>
    - In the gif above, 
        1. Apply 2 filters of the shape $ 3\times3\times3 $.
        2. 1st filter - 1st layer - 1st block: 
        
            $$\begin{equation}
            0+0+0+0+0+0+0+(1\times-1)+0=-1
            \end{equation}$$
            
            1st filter - 2nd layer - 1st block:
            
            $$\begin{equation}
            0+0+0+0+(2\times-1)+(1\times1)+0+(2\times1)+0=1
            \end{equation}$$
            
            1st filter - 3rd layer - 1st block:
            
            $$\begin{equation}
            0+0+0+0+(2\times1)+0+0+(1\times-1)+0=1
            \end{equation}$$
            
        3. Sum up + bias $ \rightarrow $ 1st cell of 1st output layer
            
            $$\begin{equation}
            -1+1+1+1=2
            \end{equation}$$
    
        4. Repeat till we finish scanning  
<br/>
- **Edge Detection & Filter**

    - Sample filters
    
        <center><img src="../../images/DL/edgedetect.png" width="500"/></center>
        
        - Gray Scale: 1 = lighter, 0 = gray, -1 = darker  
    <br/>
    - Notice that we don't really need to define any filter values. Instead, we are supposed to train the filter values.  
    All the convolution operations above are just the same as the operations in ANN. Filters here correspond to $ W $ in ANN.  
    
- **Padding**

    - Problem: corner cells & edge cells are detected much fewer times than the middle cells $ \rightarrow $ info loss of corner & edge
    
    - Solution: pad the edges of the image with "0" cells (as shown in the gif above)
    
- **Stride**: the step size the filter takes ($ s=2 $ in the gif above)

- <a name="formula"></a>**General Formula of Convolution**: 

    $$\begin{equation}
    \text{Output Size}=\left\lfloor\frac{n+2p-f}{s}+1\right\rfloor\times\left\lfloor\frac{n+2p-f}{s}+1\right\rfloor
    \end{equation}$$
    
    - $ n\times n $: image size
    - $ f\times f $: filter size
    - $ p $: padding
    - $ s $: stride
    - Floor: ignore the computation when the filter sweeps the region outside the image matrix  
<br/>
- <a name="layers"></a>**CNN Layers**:

    - **Convolution** (CONV): as described above
    
    - <a name="pool"></a>**Pooling** (POOL): to reduce #params & computations (most common pooling size = $ 2\times2 $)
    
        - Max Pooling
        
            <center><img src="../../images/DL/maxpool.png" height="200"/></center>
            
            1. Divide the matrix evenly into regions
            2. Take the max value in that region as output value  
            <br/>
        - Average Pooling
        
            <center><img src="../../images/DL/avgpool.png" height="190"/></center>
            
            1. Divide the matrix evenly into regions
            2. Take the average value of the cells in that region as output value  
            <br/>
        - Stochastic Pooling
        
            <center><img src="../../images/DL/stochasticpool.png" height="200"/></center>
            
            1. Divide the matrix evenly into regions
            2. Normalize each cell based on the regional sum:
            
                $$\begin{equation}
                p_i=\frac{a_i}{\sum_{k\in R_j}{a_k}}
                \end{equation}$$
                
            3. Take a random cell based on multinomial distribution as output value  
        <br/>
    - <a name="fc"></a>**Fully Connected** (FC): to flatten the 2D/3D matrices into a single vector (each neuron is connected with all input values)
    
        <center><img src="../../images/DL/fullyconnected.png" width="300"/></center>

## CNN Examples

<a name="lenet"></a>**LeNet-5**: LeNet-5 Digit Recognizer
        
<center><img src="../../images/DL/cnneg.png"/></center>  

|  Layer  |  Shape  | Total Size | #params |
| :-----: | :-----: | :--------: | :-----: |
| INPUT | 32 x 32 x 3 | 3072 | 0 |
| CONV1 (Layer 1) | 28 x 28 x 6 | 4704 | 156 |
| POOL1 (Layer 1) | 14 x 14 x 6 | 1176 | 0 |
| CONV2 (Layer 2) | 10 x 10 x 16 | 1600 | 416 |
| POOL2 (Layer 2) | 5 x 5 x 16 | 400 | 0 |
| FC3 (Layer 3) | 120 x 1 | 120 | 48001 |
| FC4 (Layer 4) | 84 x 1 | 84 | 10081 |
| Softmax | 10 x 1 | 10 | 841 |

- Calculation of #params for CONV: $ (f\times f+1)\times n_f $
    - $ f $: filter size
    - $ +1 $: bias
    - $ n_f $: #filter
 
<br/>
<a name="alexnet"></a>**AlexNet**: winner of 2012 ImageNet Large Scale Visual Recognition Challenge  

<center><img src="../../images/DL/alexnet.png"/></center><br/>  
    
|  Layer  |  Shape  | Total Size | #params |
| :-----: | :-----: | :--------: | :-----: |
| INPUT | 227 x 227 x 3 | 154587 | 0 |
| CONV1 (Layer 1) | 55 x 55 x 96 | 290400 | 11712 |
| POOL1 (Layer 1) | 27 x 27 x 96 | 69984 | 0 |
| CONV2 (Layer 2) | 27 x 27 x 256 | 186624 | 6656 |
| POOL2 (Layer 2) | 13 x 13 x 256 | 43264 | 0 |
| CONV3 (Layer 3) | 13 x 13 x 384 | 64896 | 3840 |
| CONV4 (Layer 3) | 13 x 13 x 384 | 64896 | 3840 |
| CONV5 (Layer 3) | 13 x 13 x 256 | 43264 | 2560 |
| POOL5 (Layer 3) | 6 x 6 x 256 | 9216 | 0 |
| FC5 (Flatten) | 9216 x 1 | 9216 | 0 |
| FC6 (Layer 4) | 4096 x 1 | 4096 | 37748737 |
| FC7 (Layer 5) | 4096 x 1 | 4096 | 16777217 |
| Softmax | 1000 x 1 | 1000 | 4096000 |

- Significantly bigger than LeNet-5 (60M params to be trained)
- Require multiple GPUs to speed the training up<br/><br/>  
    
<a name="vgg"></a>**VGG**: made by Visual Geometry Group from Oxford  

<center><img src="../../images/DL/vgg.png"/></center>  

- Too large: 138M params<br/><br/>  

**Inception**  

- <a name="res"></a>**ResNets** 

    - Residual Block

        <center><img src="../../images/DL/resnet.png" width="500"/></center>

        $$\begin{equation}
        a^{[l+2]}=g(z^{[l+2]}+a^{[l]})
        \end{equation}$$
    
        Intuition: we add activation values from layer $ l$ to the activation in layer $l+2 $ 

    - Why ResNets?
    
        - ResNets allow parametrization for the identity function $ f(x)=x $
        - ResNets are proven to be more effective than plain networks:
        
            <center><img src="../../images/DL/resnetperf.png" width="500"/></center>
            
        - ResNets add more complexity to the NN in a very simple way
        - The idea of ResNets further inspired the development of RNN  
<br/>
- <a name="nin"></a>**1x1 Conv** (i.e. Network in Network [NiN])  

    - WHY??? This sounds like the stupidest idea ever!!
    - Watch this.
        
        <center><img src="../../images/DL/1x1pt1.png" height="300"/></center><br/>
        
        <center>In a normal CNN layer like this, we need to do in total 210M calculations.</center><br/>
        
        <center><img src="../../images/DL/1x1pt2.png" height="300"/></center><br/>
    
        <center>However, if we add a 1x1 Conv layer in between, we only need to do in total 17M calculations.</center><br/>
    
    - Therefore, 1x1 Conv is significantly more useful than what newbies expect. When we would like to keep the matrix size but reduce #layers, using 1x1 Conv can significantly reduce #computations needed, thus requiring less computing power.  
<br/>
- <a name="inception"></a>**The Inception**: We need to go deeper!

    - Inception Module
    
        <center><img src="../../images/DL/incepm.png" width="400"/></center><br/>
        
    - Inception Network
    
        <center><img src="../../images/DL/incep.png" width="500"/></center>     

<a name="conv1d"></a>**Conv1D & Conv3D**:

Although CNN (Conv2D) is undoubtedly most useful in Computer Vision, there are also some other forms of CNN used in other fields:

- **Conv1D**: e.g. text classification, heartbeat detection, etc.

    <center><img src="../../images/DL/conv1d.png" width="400"/></center>
    
    - use a 1D filter to convolve a 1D input vector
    - e.g. $ 14\times1\xrightarrow{5\times1,16}10\times16\xrightarrow{5\times16,32}6\times32 $
    - However, this is almost never used since we have **RNN**  
<br/>
- **Conv3D**: e.g. CT scan, etc.

    <center><img src="../../images/DL/conv3d.png" width="400"/></center>
    
    - use a 3D filter to convolve a 3D input cube
    - e.g. $ 14\times14\times14\times1\xrightarrow{5\times5\times5\times1,16}10\times10\times10\times16\xrightarrow{5\times5\times5\times16,32}6\times6\times6\times32 $

## Object Detection

- Object Localization $ \rightarrow$ 1 obj; Detection $\rightarrow $ multiple objs.

- **Bounding Box**: to capture the obj in the img with a box
    - Params: 
        - $ b_x, b_y $ = central point
        - $ b_h, b_w $ = full height/width
    - New target label (in place of image classification output):
        
        $$\begin{equation}
        y=\begin{bmatrix}
        p_c \\ b_x \\ b_y \\ b_h \\ b_w \\ c_1 \\ \vdots \\ c_n
        \end{bmatrix}
        \end{equation}$$
    
        - $ p_c $: "is there any object in this box?"
            - if $ p_c=0 $, we ignore the remaining params
        - $ c_i$: class label $i$ (e.g. $c_1$: cat, $c_2$: dog, $c_3 $: bird, etc.)  
<br/>      
- **Landmark Detection**: to capture the obj in the img with points
    - Params: $ (l_{ix},l_{iy}) $ = each landmark point
    - New target label:
    
        $$\begin{equation}
        y=\begin{bmatrix}
        p_c \\ l_{1x} \\ l_{1y} \\ \vdots \\ l_{nx} \\ l_{ny} \\ c_1 \\ \vdots \\ c_n
        \end{bmatrix}
        \end{equation}$$
        
    - THE LABELS MUST BE CONSISTENT!
        - Always start from the exact same location of the object! (e.g. if you start with the left corner of the left eye for one image, you should always start with the left corner of the left eye for all images.)
        - #landmarks should be the same! 

    <br/>
    I personally have a very awful experience with Landmark Detection. When the algorithms of object detection were not yet well-known in the IT industry, I worked on a project of digital screen defects detection in a Finnish company. Since digital screen defects are 1) black & white 2) in very simple geometric shapes, the usage of bounding boxes could have significantly reduced the complexity of both data collection and NN model building.<br/>  
    However, the team insisted to use landmark detection. Due to 1) that screen defects are unstructured 2) that the number of landmark points for two different screen defects can hardly be the same, the dataset was basically unusable, and none of the models we built could learn accurate patterns from it, leading to an unfortunate failure.<br/>  
    I personally would argue that bounding box is much better than landmark detection in most practical cases.
    
- **Sliding Window**

    <center><img src="../../images/DL/sliding.gif" width="500"/></center><br/>
    
    - Apply a sliding window with a fixed size to scan every part of the img left-right and top-bottom (just like CONV), and feed each part to CNN
    - In order to capture the same type of objects in different sizes and positions in the img, shrink the img (i.e. enlarge the sliding window) and scan again, and repeat.<br/>

    - Problem: HUGE computational cost!
    - Solution: (contemporary)
        1. Convert FC layer into CONV layer
        
            <center><img src="../../images/DL/slidingfc.jpg" width="700"/></center><br/>
        
        2. Share the former FC info with latter convolutions
        
            <center><img src="../../images/DL/sliding.png" width="700"/></center><br/>
            
            1. First run of the CNN.
            2. Second run of the same CNN with a bigger size of the same img (due to sliding window). Notice that the FC info from the first run is shared in the second run.
            3. Latter runs of the same CNN with bigger sizes of the same img (due to sliding window). Notice that the FC info from all previous runs is shared in this run, thus saving computation power and memories.  
<br/>           
- **Intersection over Union**

    <center><img src="../../images/DL/iou.png" width="200"/></center>  
    <center>Is the purple box a good prediction of the car location?</center>
    
    Intersection over Union is defined as:
    
    $$\begin{equation}
    \text{IoU}=\frac{\text{area of intersection}}{\text{area of union}}
    \end{equation}$$
    
    In this case, area of intersection is the intersection between the red and purple box, and area of union is the total area covered by the red and purple box.  
    If $ \text{IoU}\leq 0.5 $, then the prediction box is correct. (Other threshold values are also okay but 0.5 is conventional.)
    
- <a name="yolo"></a>**YOLO (You Only Look Once)**

    <center><img src="../../images/DL/yolo.jpg" width="300"/></center>

    - **Grids**: divide the image into grids & use each grid as a bounding box
        - when $ p_c=0 $, we ignore the entire grid
        - $ p_c=1$ only when the central point of the object $\in $ the grid
        - target output: $ Y.\text{shape}=n_{\text{grid}}\times n_{\text{grid}}\times y.\text{length} $  
    <br/>
    - **Non-Max Suppression**: what happens when the grid is too small to capture the entire object?
        <center><img src="../../images/DL/nms.jpg" width="500"/></center>
        
        1. Discard all boxes with $ p_c\leq 0.6 $
        2. Pick the box with the largest $ p_c $ as the prediction
        3. Discard any remaining box with $ \text{IoU}\geq 0.5 $ with the prediction
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
        
        3. Each object in the image is assigned to grid cell that contains object's central point & anchor box for the grid cell with the highest $ \text{IoU} $  
    <br/>   
    - **General Procedure**:
    
        1. Divide the images into grids and label the objects
        2. Train the CNN
        3. Get the prediction for each anchor box in each grid cell
        4. Get rid of low probability predictions
        5. Get final predictions through non-max suppression for each class  
<br/>
- <a name="rcnn"></a>**R-CNN**

    TO BE CONTINUED

## Face Recognition

- Face Verification vs Face Recognition
    - Verification
        - Input image, name/ID
        - Output whether the input image is that of the claimed person (1:1)
    - Recognition
        - Input image
        - Output name/ID if the image is any of the $ K $ ppl in the database (1:K)  
<br/>
- <a name="sn"></a>**Siamese Network**

    - **One Shot Learning**: learn a similarity function

        The major difference between normal image classification and face recognition is that we don't have enough training examples. Therefore, rather than learning image classification, we

        1. Calculate the degree of diff between the imgs as $ d $
        2. If $ d\leq\tau$: same person; If $d>\tau $: diff person  
    <br/>
    - Preparation & Objective:
        - Encode $ x^{(i)}$ as $f(x^{(i)}) $ (defined by the params of the NN)
        - Compute $ d(x^{(i)},x^{(j)})=\left\lVert{f(x^{(i)})-f(x^{(j)})}\right\lVert_ 2^2 $            
            - i.e. distance between the two encoding vectors
            - if $ x^{(i)},x^{(j)}$ are the same person, $\left\lVert{f(x^{(i)})-f(x^{(j)})}\right\lVert_ 2^2 $ is small
            - if $ x^{(i)},x^{(j)}$ are different people, $\left\lVert{f(x^{(i)})-f(x^{(j)})}\right\lVert_ 2^2 $ is large  
    <br/>
    - **Method 1: <a name="tl"></a>Triplet Loss** 
        - <u>Learning Objective</u>: distinguish between Anchor image & Positive/Negative images (i.e. **A vs P / A vs N**)
        
            1. <u>Initial Objective</u>: $ \left\lVert{f(A)-f(P)}\right\lVert_ 2^2 \leq \left\lVert{f(A)-f(N)}\right\lVert_ 2^2 $  
            
                <u>Intuition</u>: We want to make sure the difference of A vs P is smaller than the difference of A vs N, so that this Anchor image is classified as positive (i.e. recognized)
                
            2. <u>Problem</u>: $ \exists\ "0-0\leq0" $, in which case we can't tell any difference
            
            3. <u>Final Objective</u>: $ \left\lVert{f(A)-f(P)}\right\lVert_ 2^2-\left\lVert{f(A)-f(N)}\right\lVert_ 2^2+\alpha\leq0 $  
            
                <u>Intuition</u>: We apply a margin $ \alpha $ to solve the problem and meanwhile make sure "A vs N" is significantly larger than "A vs P"  
            
        - <u>Loss Function</u>:
        
            $$\begin{equation}
            \mathcal{L}(A,P,N)=\max{(\left\lVert{f(A)-f(P)}\right\lVert_ 2^2-\left\lVert{f(A)-f(N)}\right\lVert_ 2^2+\alpha, 0)}
            \end{equation}$$
        
            - <u>Intuition</u>: As long as this thing is less than 0, the loss is 0 and that's a successful recognition!  
        <br/>
        - <u>Training Process</u>:
            - Given 10k imgs of 1k ppl: use the 10k images to generate triplets $ A^{(i)}, P^{(i)}, N^{(i)} $
            - Make sure to have multiple imgs of the same person in the training set
            - <strike>random choosing</strike>
            - Choose triplets that are quite "hard" to train on
        
            <center><img src="../../images/DL/andrew.png" width="300"/></center>
            
    - **Method 2: <a name="bc"></a>Binary Classification**
    
        - <u>Learning Objective</u>: Check if two imgs represent the same person or diff ppl
            - $ y=1 $: same person
            - $ y=0 $: diff ppl
            
        - <u>Training output</u>:
        
            $$\begin{equation}
            \hat{y}=\sigma\Bigg(\sum_{k=1}^{128}{w_i \Big|f(x^{(i)})_ k-f(x^{(j)})_ k\Big|+b}\Bigg)
            \end{equation}$$
        
            <center><img src="../../images/DL/binary.png" width="500"/></center>
        
            - Precompute the output vectors $ f(x^{(i)})\ \&\ f(x^{(j)}) $ so that you don't have to compute them again during each training process  
<br/>
- <a name="nst"></a>**Neural Style Transfer**
    - <u>Intuition</u>: **Content(C) + Style(S) = Generated Image(G)**
    
        <center><img src="../../images/DL/csg.png" width="500"/></center>
        <center>Combine Content image with Style image to Generate a brand new image</center>  
    <br/>
    - <u>Cost Function</u>: 
    
        $$\begin{equation}
        \mathcal{J}(G)=\alpha\mathcal{J}_ \text{content}(C,G)+\beta\mathcal{J}_ \text{style}(S,G)
        \end{equation}$$
        
        - $ \mathcal{J} $: the diff between C/S and G
        - $ \alpha,\beta $: weight params
        - Style: correlation between activations across channels
            
            <center><img src="../../images/DL/corr.png" width="500"/></center>
            
            When there is some pattern in one patch, and there is another pattern that changes similarly in the other patch, they are **correlated**.  
            
            e.g. vertical texture in one patch $ \leftrightarrow $ orange color in another patch  
            
            The more often they occur together, the more correlated they are.
            
        - Content Cost Function:
        
            $$\begin{equation}
            \mathcal{J}_ \text{content}(C,G)=\frac{1}{2}\left\lVert{a^{[l](C)}-a^{[1](G)}}\right\lVert^2
            \end{equation}$$
            
            - Use hidden layer $ l $ to compute content cost
            - Use pre-trained CNN (e.g. VGG)
            - If $ a^{[l](C)}\ \&\ a^{[l](G)} $ are similar, then both imgs have similar content  
        <br/>
        - Style Cost Function:
        
            $$\begin{equation}
            \mathcal{J}_ \text{style}(S,G)=\sum_l{\lambda^{[l]}\mathcal{J}_ \text{style}^{[l]}(S,G)}
            \end{equation}$$
            
            - Style Cost per layer:
            
                $$\begin{equation}
                \mathcal{J}^{[l]}_ \text{style}(S,G)=\frac{1}{(2n_h^{[l]}n_w^{[l]}n_c^{[l]})^2}\left\lVert{G^{[l](S)}-G^{[1](G)}}\right\lVert^2_F
                \end{equation}$$
                
                - the first term is simply a normalization param 
            
            - Style Matrix:
            
                $$\begin{equation}
                G_{kk'}^{[l]}=\sum_{i=1}^{n_H^{[l]}}{\sum_{j=1}^{n_W^{[l]}}{a_{i,j,k}^{[l]}\cdot a_{i,j,k'}^{[l]}}}
                \end{equation}$$
            
                - $ a_{i,j,k}^{[l]}$: activation at height $i$, width $j$, channel $k $
                - $ G^{[l]}.\text{shape}=n_c^{[l]}\times n_c^{[l]} $
                - <u>Intuition</u>: sum up the multiplication of the two activations on the same cell in two different channels
                
        - Training Process:
        
            - Intialize $ G $ randomly (e.g. 100 x 100 x 3)
            - Use GD to minimize $ \mathcal{J}(G)$: $G := G-\frac{\partial{\mathcal{J}(G)}}{\partial{G}} $


<!-- # Recurrent Neural Networks {#rnn} -->

## Basics of RNN

### <strong>Intuition of Sequence Models</strong>  

These are called sequence modeling:

- Speech recognition
- Music generation
- Sentiment classification
- DNA sequence analysis
- Machine translation
- Video activity recognition
- Name entity recognition
- ......

Forget about the tedious definitions. As a basic intuition of what we are doing in sequence modeling, here is a very simple example:

- We have a sentence: "Pewdiepie and MrBeast are two of the greatest youtubers in human history."
- We want to know: where are the "names" in this sentence? (i.e. name entity recognition)
- We convert the input sentence into $ X$: $x^{\langle 1 \rangle}x^{\langle 2 \rangle}...x^{\langle t \rangle}...x^{\langle 12 \rangle} $

    where $ x^{\langle t \rangle} $ represents each word in the sentence.  
    
    But how does it represent a word? Notice that we used the capitalized $ X$ for a single sentence. Actually, $X.\text{shape}=5000\times12$, and $x.\text{shape}=5000\times1 $. Why?
    
    We first make a vocabulary list like $ \text{list}=[\text{a; and; ...; history; ...; MrBeast; ...}] $.
    
    Then, we convert each word into a one-hot vector representing the index of the word in the dictionary, e.g.:
    
    $$\begin{equation}
    x^{\langle 1 \rangle}=\begin{bmatrix}
    0 \\ \vdots \\ 1 \\ \vdots \\ 0
    \end{bmatrix}\longleftarrow 425,\ 
    x^{\langle 2 \rangle}=\begin{bmatrix}
    0 \\ \vdots \\ 1 \\ \vdots \\ 0
    \end{bmatrix}\longleftarrow 3578,\ \cdots\cdots
    \end{equation}$$
    
- We then label the output as $ y: 1\ 0\ 1\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0 $ and train our NN on this.

- Accordingly, we can use most of the sequences in our daily life as datasets and build our NN models on them to solve such ML problems.  

### <strong>Intuition of RNN</strong>

We have very briefly mentioned that Conv1D can be used to scan through a sequence, extract features and make predictions. Then why don't we just stick to Conv1D or use normal ANNs?

1. The scope of sequence modeling is not necessarily recognition or classification, meaning that our inputs & outputs can be in very diff lengths for diff examples. 
2. Neither ANNs nor CNNs share features learned across diff positions of a text or a sequence, whereas context matters quite a lot in most sequence modeling problems.

Therefore, we need to define a brand new NN structure that can perfectly align with sequence modeling - RNN:

<center><img src="../../images/DL/rnn.jpg" height="200"/></center>
<br/>
Forward propagation:
- $ a^{\langle 0 \rangle}=\textbf{0} $
- $ a^{\langle t \rangle}=g(W_{a}[a^{\langle t-1 \rangle}; x^{\langle t \rangle}]+b_a)\ \ \ \ \|\ g:\ \text{tanh/ReLU} $  
    
    where $ W_a=[W_{aa}\ W_{ax}]$ with a shape of $(100,10100)$ if we assume a dictionary of 10000 words (i.e. $x^{\langle t \rangle}.\text{shape}=(10000,100) $) and the activation length of 100.

- $ \hat{y}^{\langle t \rangle}=g(W_{y}a^{\langle t \rangle}+b_y)\ \ \ \ \|\ g:\ \text{sigmoid} $  

Backward propagation:
- $ \mathcal{L}^{\langle t \rangle}(\hat{y}^{\langle t \rangle},y^{\langle t \rangle})=-\sum_i{y_i^{\langle t \rangle}\log{\hat{y}_ i^{\langle t \rangle}}}\ \ \ \ \|\  $Same loss function as LogReg  

### <strong>RNN Types</strong>

<center><img src="../../images/DL/rnntypes.png" width="550"/></center>  
<br/>
There is nothing much to explain here. The images are pretty clear.  

### <strong>Language Model</strong>

- <u>Intuition of Softmax & Conditional Probability</u>

    The core of RNN is to calculate the likelihood of a sequence: $ P(y^{\langle 1 \rangle},y^{\langle 2 \rangle},...,y^{\langle t \rangle}) $ and output the one with the highest probability.

    For example, the sequence "<u>the apple and pair salad</u>" has a much smaller possibility to occur than the sequence "<u>the apple and pear salad</u>". Therefore, RNN will output the latter. This seems much like **Softmax**, and indeed it is. 

    Recall from the formula of conditional probability, we can separate the likelihood into:

    $$\begin{equation}
    P\big(y^{\langle 1 \rangle},y^{\langle 2 \rangle},...,y^{\langle t \rangle}\big)=P\big(y^{\langle 1 \rangle}\big)P\big(y^{\langle 2 \rangle}|y^{\langle 1 \rangle}\big)...P\big(y^{\langle t \rangle}|y^{\langle 1 \rangle},y^{\langle 2 \rangle},...,y^{\langle t-1 \rangle}\big)
    \end{equation}$$

    For example, to generate the sentence "I like cats.", we calculate:

    $$\begin{equation}
    P\big(\text{"I like cats"}\big)=P\big(\text{"I"}\big)P\big(\text{"like"}|\text{"I"}\big)P\big(\text{"cats"}|\text{"I like"}\big)
    \end{equation}$$  

- <u>Language Modeling Procedure</u>  
    1. Data Preparation
        * Training set: large corpus of English text (or other languages)
        * **Tokenize**: mark every word into a token
            * \<EOS>: End of Sentence token
            * \<UNK>: Unknown word token
        * e.g. "I hate Minecraft and kids." $ \Rightarrow $ "I hate \<UNK> and kids. \<EOS>"  
    <br/>
    2. Training
        <center><img src="../../images/DL/rnnlm.png" width="700"/></center>  
        <br/>
        We use the sentence "I hate Minecraft and kids. \<EOS>" as one training example.  
        At the beginning, we initialize $ a^{<0>}$ and $x^{<1>}$ as $\vec{0} $ and let the RNN try to guess the first word.  
        At each step, we use the original word at the same index $ y^{\<i-1>}$ and the previous activation $a^{\<i-1>}$ to let the RNN try to guess the next word $\hat{y}^{\<i>} $ from Softmax regression.  
        During the training process, we try to minimize the loss function $ \mathcal{L}(\hat{y},y) $ to ensure the training is effective to predict the sentence correctly.
    <br/>
    3. Sequence Sampling
        <center><img src="../../images/DL/rnnsample.png" width="700"/></center>  
        <br>
        After the RNN is trained, we can use it to generate a sentence by itself. In each step, the RNN will take the previous word it generated $ \hat{y}^{\<i-1>}$ as $x^{\<i>}$ to generate the next word $\hat{y}^{\<i>} $.
        
- <u>Character-level LM</u>
    - Dictionary
        - Normal LM: [a, abandon, etc., zoo, \<UNK>]
        - Char-lv LM: [a, b, c, etc., z]
    - Pros & Cons
        - Pros: never need to worry about unknown words \<UNK>
        - Cons: sequence becomes much much longer; the RNN doesn't really learn anything about the words.  
<br>
- <u>Problems with current RNN</u>  
    One of the most significant problems with our current simple RNN is **vanishing gradients**. As shown in the figures above, the next word always has a very strong dependency on the previous word, and the dependency between two words weakens as the distance between them gets longer. In other words, the current RNN are very bad at catching long-line dependencies, for example,
    
    <center>the <strong>cat</strong>, which already ......, <strong>was</strong> full.</center>
    <center>the <strong>cats</strong>, which already ......, <strong>were</strong> full.</center>
    <br>
    "be" verbs have high dependencies on the "subject", but RNN doesn't know that. Since the distance between these two words are too long, the gradient on the "subject" nouns would barely affect the training on the "be" verbs.

## RNN Variations

| RNN | GRU | LSTM |
|:---:|:---:|:----:|
| <img src="../../images/DL/rnnblock.png" width="330"/> | <img src="../../images/DL/gru.png" width="330"/> | <img src="../../images/DL/lstm.png" width="330"/> |

As shown above, there are currently 3 most used RNN blocks. The original RNN block activates the linear combination of $ a^{\<t-1>}$ and $x^{\<t>}$ with a $\text{tanh} $ function and then passes the output value onto the next block.

However, because of the previously mentioned problem with the original RNN, scholars have created some variations, such as GRU & LSTM.  

### <strong>GRU</strong> (Gated Recurrent Unit)

<center><img src="../../images/DL/gru.png" width="400"/></center>  
<br>
As the name implies, GRU is an advancement of normal RNN block with "gates". There are 2 gates in GRU:

- **R gate**: (Remember) determine whether to remember the previous cell
- **U gate**: (Update) determine whether to update the computation with the candidate

Computing process of GRU:

1. Compute R gate:

    $$\begin{equation}
    \Gamma_r=\sigma\big(w_r\big[a^{<t-1>};x^{<t>}\big]+b_r\big)
    \end{equation}$$

2. Compute U gate:

    $$\begin{equation}
    \Gamma_u=\sigma\big(w_u\big[a^{<t-1>};x^{<t>}\big]+b_u\big)
    \end{equation}$$
    
3. Compute Candidate:

    $$\begin{equation}
    \tilde{c}^{<t>}=\tanh{\big(w_c\big[\Gamma_r * a^{<t-1>};x^{<t>}\big]+b_c\big)}
    \end{equation}$$
    
    When $ \Gamma_r=0$, $\tilde{c}^{\<t>}=\tanh{\big(w_cx^{\<t>}+b_c\big)} $, the previous word has no effect on the word choice of this cell.
    
4. Compute Memory Cell:

    $$\begin{equation}
    c^{<t>}=\Gamma_u \cdot \tilde{c}^{<t>} + (1-\Gamma_u) \cdot c^{<t-1>}
    \end{equation}$$
    
    When $ \Gamma_u=1$, &emsp;$c^{\<t>}=\tilde{c}^{\<t>} $. The candidate updates.  
    When $ \Gamma_u=0$, &emsp;$c^{\<t>}=c^{\<t-1>} $. The candidate does not update.
    
5. Output:

    $$\begin{equation}
    a^{<t>}=c^{<t>}
    \end{equation}$$ 

### <strong>LSTM</strong> (Long Short-Term Memory)

<center><img src="../../images/DL/lstm.png" width="400"/></center>  
<br>
LSTM is an advancement of GRU. While GRU relatively saves more computing power, LSTM is more powerful. There are 3 gates in LSTM:

- **F gate**: (Forget) determine whether to forget the previous cell
- **U gate**: (Update) determine whether to update the computation with the candidate
- **O gate**: (Update) Compute the normal activation

Computing process of GRU:

1. Compute F gate:

    $$\begin{equation}
    \Gamma_f=\sigma\big(w_f\big[a^{<t-1>};x^{<t>}\big]+b_f\big)
    \end{equation}$$

2. Compute U gate:

    $$\begin{equation}
    \Gamma_u=\sigma\big(w_u\big[a^{<t-1>};x^{<t>}\big]+b_u\big)
    \end{equation}$$
    
3. Compute O gate:

    $$\begin{equation}
    \Gamma_o=\sigma\big(w_o\big[a^{<t-1>};x^{<t>}\big]+b_o\big)
    \end{equation}$$
    
4. Compute Candidate:

    $$\begin{equation}
    \tilde{c}^{<t>}=\tanh{\big(w_c\big[a^{<t-1>};x^{<t>}\big]+b_c\big)}
    \end{equation}$$
    
5. Compute Memory Cell:

    $$\begin{equation}
    c^{<t>}=\Gamma_u \cdot \tilde{c}^{<t>} + \Gamma_f \cdot c^{<t-1>}
    \end{equation}$$
    
6. Output:

    $$\begin{equation}
    a^{<t>}=\Gamma_o \cdot \tanh{c^{<t>}}
    \end{equation}$$
    
**Peephole Connection**: as shown in the formulae, the gate values $ \Gamma \propto c^{\<t-1>}$, therefore, we can always include $c^{\<t-1>} $ into gate calculations to simplify the computing.

### <strong>Bidirectional RNN</strong>

<u>Problem</u>: Sometimes, our choices of previous words are dependent on the latter words. For example,

<center><strong>Teddy</strong> Roosevelt was a nice president.</center>
<center><strong>Teddy</strong> bears are now on sale!!!&emsp;&emsp;&emsp;</center>  
<br>
The word "Teddy" represents two completely different things, but without the context from the latter part, we cannot determine what the "Teddy" stands for. (This example is cited from Andrew Ng's Coursera Specialization)  

<u>Solution</u>: We make the RNN bidirectional:

<center><img src="../../images/DL/birnn.png" height="250"/></center>  
<br>
Each output is calculated as: $ \hat{y}^{\<t>}=g\Big(W_y\Big[\overrightarrow{a}^{\<t>};\overleftarrow{a}^{\<t>}\Big]+b_y\Big) $

### <strong>Deep RNN</strong>

Don't be fascinated by the name. It's just stacks of RNN layers:

<center><img src="../../images/DL/drnn.png" width="700"/></center>  
## Word Embeddings

Word embedding is a vectorized representation of a word. Because our PC cannot directly understand the meaning of words, we need to convert these words into numerical values first. So far, we have been using <a name="ohr"></a>**One-hot Encoding**:

$$\begin{equation}
x^{<1>}=\begin{bmatrix}
0 \\ \vdots \\ 1 \\ \vdots \\ 0
\end{bmatrix}\longleftarrow 425,\ 
x^{<2>}=\begin{bmatrix}
0 \\ \vdots \\ 1 \\ \vdots \\ 0
\end{bmatrix}\longleftarrow 3578,\ \cdots\cdots
\end{equation}$$

<u>Problem</u>: our RNN doesn't really learn anything about these words from one-hot representation.

### <strong>Featurized Representation</strong>

<u><strong>Intuition:</strong></u> Suppose we have an online shopping review: "Love this dress! Sexy and comfy!", we can represent this sentence as: 

<center><img src="../../images/DL/fr.png" width="600"/></center>  
<br>
We predefine a certain number of features (e.g. gender, royalty, food, size, cost, etc.). 

Then, we give each word (column categories) their relevance to each feature (row categories). As shown in the picture for example, "dress" is very closely related to the feature "gender", therefore given the value "1". Meanwhile, "love" is very closely related to the feature "positive", therefore given the value "0.99".

After we define all the featurized values for the words, we get a vectorized representation of each word:

$$\begin{equation}
\text{love}=e_ {1479}=\begin{bmatrix}
0.03 \\ 0.01 \\ 0.99 \\ 1.00 \\ \vdots
\end{bmatrix},\text{comfy}=e_ {987}=\begin{bmatrix}
0.01 \\ 0.56 \\ 0.98 \\ 0.00 \\ \vdots
\end{bmatrix},\cdots\cdots\end{equation}$$

This way, our RNN will get to know the rough meanings of these words.

For example, when it needs to generate the next word of this sentence: **"I want a glass of orange _____."**

Since it knows that **"orange"** is a **fruit** and that **"glass"** is closely related to **liquid**, there is a much higher possibility that our RNN will choose **"juice"** to fill in the blank.

<u><strong>Embedding matrix:</strong></u> To acquire the word embeddings such as $ \vec{e}_ {1479}$ and $\vec{e}_ {987} $ above, we can multiply our embedding matrix with the one-hot encoding:

$$\begin{equation}
E\times \vec{o}_ j=\vec{e}_ j
\end{equation}$$

where $ E$ is our featurized representation (i.e. embedding matrix) and $\vec{o}_ j $ is the one-hot encoding of the word (i.e. the index of the word).

In practice, this is too troublesome since the dimensions of our $ E$ tend to be huge (e.g. $(500,10000) $). Thus, we use specialized function to look up an embedding directly from the embedding matrix.

<u><strong>Analogies:</strong></u> One of the most useful properties of word embeddings is analogies. For example, **"man $ \rightarrow$ woman"="king $\rightarrow $ ?"**.

Suppose we have the following featurized representation:

|  | man | woman | king | queen |
|:-:|:-:|:-:|:-:|:-:|
| gender | -1 | 1 | -0.99 | 0.99 |
| royal | 0.01 | 0.02 | 0.97 | 0.96 |
| age | 0.01 | 0.01 | 0.78 | 0.77 |
| food | 0.03 | 0.04 | 0.04 | 0.02 |

In order to learn the analogy, our RNN will have the following thinking process:

$$\begin{align}
&\text{Goal: look for}\ w: \mathop{\arg\max}_ w{sim(e_w, e_{\text{king}}-(e_{\text{man}}-e_{\text{woman}}))} \\
&\because e_{\text{man}}-e_{\text{woman}}\approx\begin{bmatrix}-2 \\ 0 \\ 0 \\ 0\end{bmatrix}, e_{\text{king}}-e_{\text{queen}}\approx\begin{bmatrix}-2 \\ 0 \\ 0 \\ 0\end{bmatrix} \\
&\therefore e_{\text{man}}-e_{\text{woman}}\approx e_{\text{king}}-e_{\text{queen}} \\
&\text{Calculate cosine similarity: } sim(\vec{u},\vec{v})=\cos{\phi}=\frac{\vec{u}^T\vec{v}}{\|\vec{u}\|_ 2\|\vec{v}\|_ 2} \\
&\text{Confirm: }e_w\approx e_{queen}
\end{align}$$

### Learning 1: <strong>Word2Vec</strong>

<u>Problem</u>: We definitely do not want to write the embedding matrix by ourselves. Instead, we train a NN model to learn the word embeddings.

Suppose we have a sentence "Pewdiepie and MrBeast are two of the greatest youtubers in human history". Before Word2Vec, let's define context & target:

- **Context**: words around target word
    - last 4 words: "two of the greatest ______"
    - 4 words on both sides: "two of the greatest ______ in human history."
    - last 1 word: "greatest ______"
    - **skip-gram**: (any nearby word) "... MrBeast ... ______ ..."
- **Target**: the word we want our NN to generate
    - "youtubers"

<u>Algorithm</u>:

1. **Randomly** choose context & target words with **skip-gram**. (e.g. context "MrBeast" & target "youtubers") 
2. Learn **mapping** of "$ c\ (\text{"mrbeast"}[1234])\rightarrow t\ (\text{"youtubers"}[765]) $"
3. Use **softmax** to calculate the probability of appearance of target given context:

    $$\begin{equation}
    \hat{y}=P(t|c)=\frac{e^{\theta_t^Te_c}}{\sum_{j=1}^{n}{e^{\theta_j^Te_c}}}
    \end{equation}$$

4. Minimize the **loss** function:

    $$\begin{equation}
    \mathcal{L}(\hat{y},y)=-\sum_{i=1}^{n}{y_i\log{\hat{y}_ i}}
    \end{equation}$$

Notes:
* Computation of softmax is very slow: Hierarchical Softmax (i.e. Huffman Tree + LogReg) can solve this problem - with common words at the top and useless words at the bottom.
* $ c\ \&\ t $ should not be entirely random: words like "the/at/on/it/..." should not be chosen.

### Learning 2: <strong>Negative Sampling</strong>

<u>Problem</u>: Given a pair of words, predict whether it's a context-target pair.

For example, given the word "orange" as the context, we want our model to know that "orange & juice" is a context-target pair but "orange & king" is not.

<u>Algorithm</u>:

1. Pick a context-target pair $ (c,t) $ (the target should be near the context) from the text corpus as a **positive example**.
2. Pick random words $ \\{t_1,\cdots,t_k\\}$ from the dictionary and form word pairs $\\{(c,t_1),\cdots,(c,t_k)\\} $ as **negative examples** based on the following probability that the creator recommended:

    $$\begin{equation}
    P(w_i)=\frac{f(w_i)^{\frac{3}{4}}}{\sum_{j=1}^{n}{f(w_i)^{\frac{3}{4}}}}
    \end{equation}$$

    where $ w_i$ is the $i $th word in the dictionary.
    
3. Train a **binary classifier** based on the training examples from previous steps:

    $$\begin{equation}
    \hat{y}_ i=P(y=1|c,t_i)=\sigma(\theta_{t_i}^Te_c)
    \end{equation}$$
    
4. Repeat Step 1-3 till we form our final embedding matrix $ E $.  

Negative Sampling is relatively faster and less costly compared to Word2Vec, since it replaces softmax with binary classification.

### Learning 3: <strong>GloVe</strong> (Global Vectors)

<u>Problem</u>: Learn word embeddings based on how many times target $ i$ appears in context of word $j $.

<u>Algorithm</u>: 

1. Minimize

    $$\begin{equation}
    \sum_{i=1}^{n}{\sum_{j=1}^{n}{f\big(X_{ij}\big)\big(\theta_i^Te_j+b_i+b'_ j-\log{X_{ij}}\big)^2}}
    \end{equation}$$

    * $ X_{ij}$: #times $i$ appears in context of $j $
    * $ f\big(X_{ij}\big) $: weighing term
        * $ f\big(X_{ij}\big)=0$ if $X_{ij}=0 $
        * $ f\big(X_{ij}\big) $ high for uncommon words
        * $ f\big(X_{ij}\big) $ low for too-common words
    * $ b_i:t$, $b'_ j:c $

2. Compute the final embedding of word $ w $:

    $$\begin{equation}
    e_w^{\text{final}}=\frac{e_w+\theta_w}{2}
    \end{equation}$$

## Sequence Modeling

### <strong>Sentiment Classification</strong>

<u>Problem Setting</u>: (many-to-one) given text, predict sentiment.

<center><img src="../../images/DL/sent.png" width="550"/></center>  
<br>
<u>Model</u>:

<center><img src="../../images/DL/sentrnn.png" width="550"/></center>  

### <strong>Seq2Seq</strong>

<u>Problem Setting</u>: (many-to-many) given an entire sequence, generate a new sequence.

<u>Example 1: Machine Translation</u>:

<center><img src="../../images/DL/seq2seq.png" width="550"/></center>  
<br>
Machine Translation vs Language Model:
* Language Model: maximize $ P(y^{\<1>},\cdots,y^{\<T_y>}) $
* Machine Translation: maximize $ P(y^{\<1>},\cdots,y^{\<T_y>} \| \vec{x}) $

<u>Example 2: Image Captioning</u>

<center><img src="../../images/DL/seq2seqic.png" width="550"/></center>  

### <strong>Beam Search</strong>

<u>Problem</u>: So far, when we choose a word from softmax for each RNN block, we are doing **greedy search**, that we only look for **local optimum** instead of **global optimum**.

That is, we only choose the word with the highest $ P(y^{\<1>}\|\vec{x})$ and then the word with the highest $P(y^{\<2>}\|\vec{x}) $ and then ...

As we already know, local optimum does not necessarily represent global optimum. In the world of NLP, the word **"going"** always has a much higher probability to appear than the word **"visiting"**, but in certain situations when we need to use "visiting", the algorithm will still choose "going", therefore generating a weird sequence as a whole.

<u>Beam Search Algorithm</u>:

1. Define a beam size of $ B$ (usually $B\in\\{1\times10^n,3\times10^n\\},\ n\in\mathbb{Z}^+ $).
2. Look at the top $ B$ words with the highest $P$s for the first word. (i.e. look for $P(\vec{y}^{\<1>}\|\vec{x}) $)
3. Repeat till \<EOS>. Choose the sequence with the highest combined probability.

<u>Improvement</u>: The original Beam Search is very costly in computing, therefore it is necessary to refine it:

$$\begin{align}
&\because P(y^{<1>},\cdots,y^{<T_y>}|x)=\prod_{t=1}^{T_y}{P(y^{<t>}|x,y^{<1>},\cdots,y^{<t-1>})} \\
&\therefore \text{goal}=\mathop{\arg\max}_ y{\prod_{t=1}^{T_y}{P(y^{<t>}|x,y^{<1>},\cdots,y^{<t-1>})}} \\
&\Rightarrow \mathop{\arg\max}_ y{\sum_{t=1}^{T_y}{\log{P(y^{<t>}|x,y^{<1>},\cdots,y^{<t-1>})}}} \\
&\Rightarrow \mathop{\arg\max}_ y{\frac{1}{T_y^{\alpha}}\sum_{t=1}^{T_y}{\log{P(y^{<t>}|x,y^{<1>},\cdots,y^{<t-1>})}}}
\end{align}$$

* $ \prod\rightarrow\sum{\log} $: log scaling
* $ \frac{1}{T_y^{\alpha}}$: length normalization (when you add more negative values ($\log{(P<1)}<0 $), the sum becomes more negative)
* $ \alpha $: <strike>learning rate</strike> just a coefficient

<u>Error Analysis</u>: Suppose we want to analyze the following error:

* Human: Jimmy visits Africa in September. ($ y^\* $)
* Algorithm: Jimmy visited Africa last September. ($ \hat{y} $)

If $ P(y^\*\|x)>P(\hat{y}\|x)$, Beam search is at fault $\rightarrow$ increase $B $  
If $ P(y^\*\|x)\leq P(\hat{y}\|x)$, RNN is at fault $\rightarrow $ improve RNN (data augmentation, regularization, architecture, etc.)

### <strong>Bleu Score</strong>

<u>Problem</u>: For many sequence modeling problems (especially seq2seq), there is no fixed correct answer. For example, there are many different Chinese translated versions of the same fiction Sherlock Holmes, and they are all correct. In this case, how do we define "correctness" for machine translation? 

<u>Bilingual Evaluation Understudy</u>:

$$\begin{equation}
p_n=\frac{\sum_{\text{n-gram}\in\hat{y}}{\text{count}_ {clip}(\text{n-gram})}}{\sum_{\text{n-gram}\in\hat{y}}{\text{count}(\text{n-gram})}}
\end{equation}$$

* $ \text{n-gram}$: $n$ consecutive words (e.g. bigram: "I have a pen." $\rightarrow $ "I have", "have a", "a pen")
* $ \text{count}_ {clip}(\text{n-gram}) $: maximal #times an n-gram appears in one of the reference sequences
* $ \text{count}(\text{n-gram})$: #times an n-gram appears in $\hat{y} $

For example,

* input: "Le chat est sur le tapis."
* Reference 1: "The cat is on the mat."
* Reference 2: "There is a cat on the mat."
* MT output: "the cat the cat on the mat."

The unigrams here are: "the", "cat", "on", "mat". Then,

$$\begin{equation}
p_1=\frac{2+1+1+1}{3+2+1+1}=\frac{5}{7}
\end{equation}$$

The bigrams here are: "the cat", "cat the", "cat on", "on the", "the mat". Then,

$$\begin{equation}
p_2=\frac{1+0+1+1+1}{2+1+1+1+1}=\frac{2}{3}
\end{equation}$$

The final Bleu score will be calculated as:

$$\begin{equation}
\text{BLEU}=BP\times e^{\frac{1}{4}\sum_{n=1}^{4}{p_n}}
\end{equation}$$

* usually we take $ n=4 $ as the upper limit for n-grams.
* $ BP$: param to penalize short outputs ($\because $ short outputs tend to have high BLEU scores.)
* $ BP=1$ if $\text{len}(\hat{y})>\text{len}(\text{ref}) $
* $ BP=e^{\frac{1-\text{len}(\hat{y})}{\text{len}(\text{ref})}}$ if $\text{len}(\hat{y})\leq\text{len}(\text{ref}) $

### <strong>Attention Model</strong>

<u>Problem</u>: Our Seq2Seq model memorizes the entire sequence and then start to generate output sequence. However, a better approach to such problems like machine translation is actually to memorize part of the sequence, translate it, then memorize the next part of the sequence, translate it, and then keep going. Memorizing the entire fiction series of Sherlock Holmes and then translate it is just inefficient. 

<u>Model</u>:

<center><img src="../../images/DL/attention.png" width="400"/></center>  
<center>Attention Model = Encoding BRNN + Decoding RNN</center>

<u>Algorithm</u>:

1. Combine BRNN activations:

    $$\begin{equation}
    a^{<t'>}=\Big(\overleftarrow{a}^{<t'>},\overrightarrow{a}^{<t'>}\Big)
    \end{equation}$$
    
    where $ t' $ refers to the index of the encoding BRNN layer.

2. Calculate the amount of "attention" that $ y^{\<t>}$ should pay to $a^{\<t'>} $:

    $$\begin{equation}
    \alpha^{<t,t'>}=\frac{e^{(e^{<t,t'>})}}{\sum_{t'=1}^{T_x}{e^{(e^{<t,t'>})}}}
    \end{equation}$$

    where $ e^{\<t,t'>}=W_e^{\<t,t'>}[s^{\<t-1>};a^{\<t'>}] +b_e^{\<t,t'>}$ is a linear combination of both encoding activation $a^{\<t'>}$ and decoding activation $s^{\<t-1>}$. $t $ refers to the index of the decoding RNN layer.

3. Calculate the total attention at $ t $: 
    
    $$\begin{equation}
    c^{<t>}=\sum_{t'}{\alpha^{<t,t'>}a^{<t'>}}
    \end{equation}$$
    
4. Include the total attention into the input for output calculation:

    $$\begin{equation}
    \hat{y}^{<t>}=s^{<t>}=g\big(W_y[\hat{y}^{<t-1>};c^{<t>}]+b_y\big)
    \end{equation}$$