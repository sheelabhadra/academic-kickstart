---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Virtual to Real Reinforcement Learning for Autonomous Driving"
subtitle: ""
summary: "Image translation networks to generate real-world images"
categories: [Paper Summary]
tags: [Reinforcement Learning, Autonomous Driving, Domain Adaptation]
date: 2019-11-03
math: true
diagram: true
markup: mmark
draft: false
featured: false
reading_time: true
share: false
profile: false
commentable: false
editable: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

[//]: # (Image References)

[image1]: image-translation-nw.png "Image Translation Network"

**Paper:** Download [here](https://arxiv.org/abs/1704.03952)  
**Video results:** Watch [here](https://www.youtube.com/watch?v=Bce2ZSlMuqY)  
**Authors:** Xinlei Pan, Yurong You, Ziyang Wang, Cewu Lu  
**Published in:** BMVC 2017

## Takeaway message
The paper introduces a realistic translation network that uses 2 conditional GANs to generate realistic images from virtual images. The intermediate images is the segmentation map which is the common ground between the real-world and virtual images. An RL agent was trained on the synthesized realistic images using A3C.

![alt text][image1]

## Motivations
RL algorithms for autonomous driving are trained on simulators. The driving policy trained on simulators doesn't translate well to the real-world due to the discrepancy between virtual images and their corresponding real-world images. But both the types of images have a common parsing structure (segmentation map) which can be exploited to generate realistic images in simulation.

## Proposed Solution
The approach consists of 2 image-to-image translation networks. The first network translates the virtual images to their segmentation maps and the second network translates segmented images into their realistic counterparts. Both the networks are GANs and have the same architecture. An A3C with 9 discrete actions is trained on the synthesized realistic images. The authors claim that this is a seminal work of adapting driving policy trained using simulations to the real-world. 

## Evaluation of the solution
Only the steering angle has been used for measuring error. TORCS was used for experiments. Cityscapes dataset was used to obtain the real-world images.
Comparison has been made with respect to 3 A3C agents trained using 3 different approaches.
- *Oracle:* Training and testing on the same environment.
- *Proposed Method:* Training on syntehsized realistic images from E-track. Testing on Cg-track2. The image translation networks were trained on images from both the training and testing tracks. 
- *Domain randomization:* Training on 10 different virtual environments. Testing on Cg-track2.

## Analysis of the problem, idea, and evaluation
**Pros**
- It could work for simple scenarios such as highway driving.
- It Works better than domain randomization method.

**Cons**
- The lane lines aren't preserved in this approach ( *NOTE: this may be a limitation of pix2pix* ).
- The synthesized real images are not always trust-worthy i.e. appearance road intersections even when they are not present in the ground truth data.
- Difficult for the approach to work for urban driving scenarios.
- The evaluation strategy predicts *only* steering angles and is hand-crafted since the actions are discrete.

## Contributions
In general, the approach can generate a decent representation of road, trees, buildings, sky, grass, and sidewalks. The network architecture of the GANs and the A3C DRL, hyperparameters, and training data details have been provided by the authors.

## Future directions
- Segmentation maps are not unique. Adding info regarding the color and texture of objects. 
- Addition of more complex scenarios such as cars, traffic lights, road-signs, and pedestrians.

## Questions
- What are the advantages of this approach over training an RL agent on the segmentation map? Personally I think that this would be easier to train.