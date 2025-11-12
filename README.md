# Gaze_Emotion_Modeling
Predicting change in emotion (post-pre scores across multiple emotive categories) given eye tracking information.
Independently selected research topic for EE 136 - Statistical Pattern Recognition, Tufts University final project.

**Authors:** Sally Kim, Laura Kaplan, Kevin Yu

---
## Table of Contents
- [Project Proposal](#project-proposal)
  - [1. Goals](#1-goals)
  - [2. Data](#2-data)
  - [3. Methods](#3-methods)
  - [4. Questions](#4-questions)
- [Repository Structure](#repository-structure)
- [Usage Instructions](#usage-instructions)
---

## Project Proposal
### 1. Goals
Our goal is to predict participants' change in emotion (post-pre scores across multiple emotive categories) given eye tracking information. To accomplish this, our model will be a multivariate linear regression model trained on tabular data. The raw data available for our task includes spatio-temporal gaze data during exposure to VR-video experiences. This is complemented by comprehensive pre-and post-exposure surveys documenting participant emotional states. The pre-and-post-exposure surveys can be examined as different measures (change in emotion with pre-survey as the baseline). Unfortunately, the raw gaze-data is time-series (not i.i.d.). To circumnavigate this complication, we will generate detailed aggregate summary statistics via value-based binning.

More concretely stated: We will categorize eye-velocities and blink-lengths, and use these categories as features to predict a vector output as a weighted emotion difference.

Our emotive difference labels will be produced by normalizing the post-exposure ratings by the pre-exposure ratings for each category.

**Input:** X = proportion of time spent in low, medium, high velocity eye movement and short, medium, high length blinks  
**Output:** y = emotion difference vector after viewing (weighted)

Our justification for probabilistic modeling is founded in a belief of some underlying relationship between gaze-data and emotional state. If such a relationship exists, then there is some likelihood by which we can predict one given the other. Eye movement and emotion data are both noisy and uncertain. The data we have access to is a limited sample (34 participants with 12-experiences each), and therefore will not perfectly represent the underlying relationships. Therefore, the correlations we do find will be probabilistic.

---

### 2. Data
In our raw dataset, we have 34 individuals, each looking at 12 experiences. These experiences differ in length, so there is no set amount of data for each experience. Typically, there are about 10,000 data points for each experience, but it can be as low as ~5000. For each experience, the number of timestamps are roughly the same for each participant.

In each experience, the raw data contain timestamps with the angles of both eyes, measured in horizontal and vertical angles of a sphere (in degrees), x and y respectively, with 0 ≤ x ≤ 360 and 0 ≤ y ≤ 180. There are also two binary indicators for if the individual is blinking in their left eye or right eye.

We will need to do preprocessing on the data to get the velocity between each timepoint for each eye (averaged). This can be done by just taking the angular distance and dividing by the difference in timestamp. We will also need to extract the length of each blink, which will take the assumption that no two blinks occur without at least one non-blink frame in between.

Our final input will be 408 (34 * 12) samples, each with 6 features (low, medium, high velocity and low, medium, high blink lengths). We will experiment with using different bins, potentially splitting into more than three categories for velocity or blink.

For our output variable, participants took surveys both before and after each experience. In particular, there is a set of questions that ask the participants to rate their emotions (joy, happiness, calmness, relaxation, anger, disgust, fear, anxiousness, and sadness) on a scale from 0 to 100. We plan to utilize the pre-exposure emotions as a baseline so that we can compare how each participant’s emotions changed with each experience. This means that for each input feature vector, we will have a corresponding output vector with length 9 corresponding to the change in the 9 emotions measured.

**Dataset:**  
https://www.kaggle.com/datasets/lumaatabbaa/vr-eyes-emotions-dataset-vreed  
L. Tabbaa et al., “VREED: Virtual Reality Emotion Recognition Dataset Using Eye Tracking & Physiological Measures,” Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., vol. 5, no. 4, p. 178:1-178:20, Dec. 2022, doi: 10.1145/3495002.

---

### 3. Methods

**Baseline:**  
We will use a MAP estimator, using evidence to select hyperparameters for the variance of the posterior and the prior (zero-mean Gaussian). Using the velocity and blink data, we will train a linear regression model using analytical methods of finding the MAP weight vector. The objective will be the scoring functions we have used before, where we measure the log probability of seeing the test data given our posterior (under a multivariate Gaussian posterior).

**Hypothesis 1:**  
We will use a PPE model instead of a MAP estimator, following similar methods to select the hyperparameters and using the same feature vectors to train our model. The objective will be the same scoring function as well. We believe this change is helpful because it can better characterize the points in our test set with different feature vectors as the points in our train set.

We hypothesize that compared to MAP estimation, PPE should improve our log probability score on our dataset, especially when we have not seen many velocity/blink metrics similar to the point we are testing on, because of the flexible variance we get from PPE. This is particularly important due to the small dataset size.

**Hypothesis 2:**  
We will also use various priors on both MAP and PPE models to see how picking one over another leads to different results. Since different types of distributions have different characteristics that may fit our data, we will explore utilizing several priors within our models. For example, we will use a Gaussian distribution with nonzero mean as one potential prior since it encapsulates the idea of the mode being the mean that appears within our data.

We hypothesize that a carefully selected non-Gaussian prior will perform better, since our data does not appear to be Gaussian in nature. As a minimal expectation, we believe the non-zero Gaussian will better model data since our preliminary data analysis shows that blink-length and gaze-velocity tend to be left-shifted from the median measure.

---

### 4. Questions
- Should we use a categorical method instead, where we take the change in emotion as a binary (increase or decrease)?
- We are planning on just taking an average of velocity for both eyes, but how could we incorporate data from both eyes separately?
- How can we intelligently pick our borders for each bin? Is there a way to determine how many bins or pick borders without a large grid search type routine?

---

## Repository Structure
*fill once we begin adding code*

## Usage Instructions
*how to interpret our preprocessing, reproduce our experiments, and locate our results*
