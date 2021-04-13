## Image Generation from Scene Graphs with Contextual Knowledge

### Motivation
Within the past decade, the computer vision community has seen astounding progress for several fundamental tasks such as image classification, object detection, instance segmentation and scene generation. However, a growing observation is that DL lacks the abstract, high-level understanding of images that humans have. This causes sub-optimal performance for vision tasks which require higher level semantic understanding of interactions between objects such as visual question answering, scene understanding and few-shot/zero-shot image classification. 

In order to guide these systems to learn a more abstract level of understanding of images, an increasing number of research has placed their attention on scene graph generation [[1]](#1), [[2]](#2), [[3]](#3). Given an input image, a scene graph aims to summarize information about the objects present and the relationships between these objects. This rich representation hence strengthens a visual system's ability to reason about the relational information present in the image, consequently improving the latent representation of the input image.

As illustrated in the image below, we see that works that use object detection networks for scene understanding will misunderstand the top row of images as semantically identical - both images contain a man beside a horse. With the generation of a scene graph as an intermediate step, visual systems will be able to generate more accurate descriptions of the interactions between objects - a man is feeding the horse with a bucket.

![Image of Scene Graph](https://alexfoodw.github.io/sg2im/images/scene-graph.jpg)

In this work, we propose to further combine low-level generated scene graphs with high-level contextual knowledge graphs and apply it on the task of image scene generation. We argue that annotated scene graphs contain noisy and unimportant information which may confuse the training of generative models, and hence have to be integrated with a more general or contextual level of understanding of objects and their key interactions in order to optimize training and accurately fabricate a scene. 

### Methodology
<img src="https://alexfoodw.github.io/sg2im/images/proposed_method.png" width="800">

As shown in the above image, we break down the generation process into 4 main steps, based on the combination of works from [[1]](#1) and [[4]](#4):
  1. Scene Graph Pruning, where we use the Commonsense KG to selectively remove noisy and unimportant relationships (and respective objects) from input scene graphs through a ranking algorithm inspired by [[1]](#1).
  2. Reasoning on scene graphs by Scene Graph Convolutions adapted from [[4]](#4).
  3. Scene layout generation performed by a Box Regression Network on final object embeddings.
  4. Pixel refinement by a Cascaded Refinement Network to fill up the final pixels given the scene layout.

For full details of the methodology, please refer to the report or the code.

### Experiments 
We train our model to generate 64x64 images on a restrained version of the Visual Genome [[5]](#5) dataset. In our experiments, we aim to show that our method is able to produce realistic images that respect the relations between objects, with faster training time and despite the restraining of total object categories. We train our model for a total of 30k iterations, with the model switching to evaluation mode after 3k iterations.

#### Qualitative Results
<img src="https://alexfoodw.github.io/sg2im/images/qual_res_0.png" width="800">
In this figure above, we list 12 sample generated images, where each respective input scene graph and the corresponding generated scene layout is shown as well. The input scene graphs are taken from our test set after training the model. We observe that although the images are not as sharp as input training images, the model manages to generate realistic looking objects which respect the relationships specified in each scene graph.

In particular, zooming into the bottom right image with a house on a hill (second set of images from bottom right of figure), we see that the model is impressively able to generate both large objects, like the sky and grass, and small objects like the building and hill.

#### Scene Graph Pruning Analysis
Let us now analyze the effect of our proposed Scene Graph Pruning algorithm, which removes unimportant relationships (and respective objects) from each training scene graph by referring to a weighted knowledge graph of all object and relationship categories. 

We propose to study the effectiveness of our pruning strategy by two metrics - Relations per Instance and Time per Iteration. The comparison of our pruning strategy versus a raw VG database used in our baseline model from [[4]](#4) is illustrated in the table below: 


Method | Relations / Instance | Time / Iteration (s)
------------ | -------------
Raw | 0.25 | 1.65
Refined (ours) | **0.11** | **1.52**


We firstly observe that, the lower the number of relations per object instance, the better we our model will be able to learn the interactions between object instances. Intuitively, this is true because the lower this ratio, the less confused our model will be since it will be able to learn about each object from a smaller and less noisy pool of focused relationships which define it. Hence, we observe that our pruning strategy is able attain a more than 50% decrease (0.25 to 0.11) in the number of relations per instance by keeping the top `k=10` relationships and pruning away the other unimportant relations with only a small number of object instances.

Next, we observe that by pruning away the mentioned unimportant relationships, we are able to train the model more efficiently. As seen in the table, we can attain an averaged (across 10 iterations) 0.13s decrease in time for every iteration. Since the training of this generative model is in the order of hundreds of thousands of iterations, this amounts to huge time savings for the training of a model which produces realistic outputs. For example, if we follow the recommended 1 million iterations training time in [[4]](#4), this would amount to 36 hours in time savings. 


#### Comparisons with Baseline
Finally, in order to comprehensively analyze the quality of generated images of our method, we compare our trained model with our baseline comparison model taken from [[4]](#4). The key difference between the two models is the proposed scene graph pruning strategy, and the significantly faster training time as discussed in the previous section. We feed an incrementally challenging scene graph input into each model, and compare the respective outputs in the figure below.

<img src="https://alexfoodw.github.io/sg2im/images/qual_res_1.png" width="800">

The analysis of this experiment is two-fold - our methods ability to handle increasingly complex inputs, and the comparison of between our method and our baseline model. Firstly, zooming in on the top row, we observe the our model is able to generate realistic images which respect each incrementally challenging scene graph. This can be seen from the images displaying relationships such as _sheep by sheep_ and _mountain behind tree_ even as the complexity of inputs increases. 

Now, when comparing the results generated by our method with our baseline model, we observe that our method is able to better respect the relationships between objects better. This can be seen in the scene graph input with _sheep by sheep_ (third set of images from the left), where our method clearly illustrates two sheep beside each other, while we only clearly see one sheep in the image generated by our baseline model. 

Further, although our method produces comparatively more blurry outputs for the 3 most challenging inputs (3 rightmost images), zooming in on the images we see that relationships between objects are better preserved by our method than the clearer outputs produced by our baseline. For instance, in relationships such as _tree behind grass_ for these 3 images, the trees are more clearly shown to be behind grass by our method, while it is not very obvious where are the tree objects in the images generated by the baseline model. 

Consequently, one possible explanation behind the comparatively blurry outputs could be due to our model prioritising the preservation of relationships between objects, which naturally causes output pixels of different objects to be less smoothly generated and combined together. 

Finally, we note that the baseline model spends a lot of time training on the full VG dataset (~3 days total training time), while our method removes a large portion of the dataset by focusing on our defined key objects (~11 hours total training time).

## References
<a id="1">[1]</a> 
Gu, J., Zhao, H., Lin, Z., Li, S., Cai, J., & Ling, M. (2019). Scene graph generation with external knowledge and image reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 1969-1978).

<a id="2">[2]</a> 
Zareian, A., Karaman, S., & Chang, S. F. (2020, August). Bridging knowledge graphs to generate scene graphs. In European Conference on Computer Vision (pp. 606-623). Springer, Cham.

<a id="3">[3]</a> 
Xu, D., Zhu, Y., Choy, C. B., & Fei-Fei, L. (2017). Scene graph generation by iterative message passing. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 5410-5419).

<a id="4">[4]</a> 
Johnson, J., Gupta, A., & Fei-Fei, L. (2018). Image generation from scene graphs. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1219-1228).

<a id="5">[5]</a> 
Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., ... & Fei-Fei, L. (2017). Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123(1), 32-73.
