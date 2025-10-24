# Hands-On Perception: Event-based Data and Motion Compensation
## **Part 3**: Motion Compensation
In the previous approach, we saw how to encode raw event-based data into frames which are
compatible with CNNs and recurrent neural networks. However, one other use case would be with
the **Motion Compensation (MC)** algorithm. MC only requires raw events data, thus there is no
need to encode the events into images. We will also be using "eWiz" to run and visualize results.

### Sequence Selection
1. First, you need to clip a sequence from the "indoor_flying1" dataset between
<span style="color:red">10.0</span> s and <span style="color:red">10.2</span> s.
The ground truth flow data also needs to be **read** and **clipped** clipped accordingly.
Refer to the previous lab to obtain the corresponding ".hdf5" ground truth groups.

> **Note:** For clipping, you just have to modify the "clip" variable in the code. After
running the algorithm you can close the windows with the "ESC" key.

2. After completing the data reading step, **run** your algorithm and **denote** the obtained
loss (should be in your report).
3. While we can observe qualitative results, some quantitative metrics will need to be
implemented.

### **Accuracy Metrics:** Average Endpoint Error
Before delving deep into MC, let us go through some of the basic optical flow metrics we will
be using. The Average Endpoint Error (AEE), is defined as the distance between the endpoints
of the predicted $(u_{pred}, v_{pred})$ and ground truth $(u_{gt}, v_{gt})$ flow vectors,
it is averaged over the number of pixels in the image. The AEE equation is given below:
$$AEE = \sum_{x, y} {\lVert (\substack{u_{pred} \\ v_{pred}}) - (\substack{u_{gt} \\ v_{gt}}) \rVert}_2$$

Hence, the **goal** of this part of the lab, which involves using different sequences and
evaluating them with the aforementioned metric.
1. Implement the AEE inside the "get_average_endpoint_error" function. The number of pixels
in the image is already provided under the "num_pixels" variable.
2. Rerun the algorithm for the same sequence and **denote** the obtained AEE (should be
in your report).

### Questions
1. Compare the "Events Images" (before and after warping). How and why are these images
different? (Provide a thorough explanation along with the images you obtained.)
2. How does the IWE change during optimization? Explain.

### **Case Study**: Short Sequence
1. Now clip a shorter sequence from the "indoor_flying1" dataset between
<span style="color:red">10.0</span> s and <span style="color:red">10.015</span> s.
2. Run the algorithm and denote your obtained metrics.

### Questions
1. Did the algorithm converge? Why, or why not? (Include obtained metrics in your answer.)

### **Case Study**: Long Sequence
1. Now clip a longer sequence from the "indoor_flying1" dataset between
<span style="color:red">20.0</span> s and <span style="color:red">23.2</span> s.
2. Run the algorithm and denote your obtained metrics.

### Questions
1. Did the algorithm converge? Why, or why not? (Include obtained metrics in your answer.)

### Patch Initialization
When initializing the "pyramidal_optimizer" object, we introduce a "scale_range" argument.
Until now, the scale range was set from 1 to 5, meaning that the optimizer starts optimizing for $2^1$ patches across each image dimensions, up-samples, and repeats the optimization till
we get $2^5$ patches.

Again, clip a sequence from the "indoor_flying1" dataset between
<span style="color:red">10.0</span> s and <span style="color:red">10.2</span> s.
1. Set the scale range between 4 and 5.
2. Run your algorithm and denote your results.

### Questions
1. How did the algorithm perform compared to the previous run for the same sequence? Why?
(Include obtained metrics in your answer.)
2. What do you suggest to improve convergence?