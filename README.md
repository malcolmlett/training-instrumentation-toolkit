# Training Instrumentation Toolkit
TensorFlow addons for instrumenting NN model training loops in order to collect, analyse, and visualize detailed
metrics on gradients, weights, and layer activations.

Intended to improve development turnaround time by making it easier to access subtle indicators of training progress and
potential training problems. Aids in troubleshooting by providing insight into many aspects of model training in one go,
quickly highlighting if any of several common issues are occurring.

Example visualisation of metrics gathered during training:

![training overview plot](doc/training-overview-example.png)

Some of the kinds of problems that this tooling is targeted at include:
* vanishing and exploding gradients
* oscillating gradients
* neuron death
* and more generally for identifying the cause of slow training progress

This project came out of experiments looking into options for improving the "training observability" - ie: improving our
ability to observe how the model training algorithm is behaving. There's a gap between the techniques used in academic
literature to investigate these kinds of problems versus what's easy to do out of the box with the likes of TensorFlow
and PyTorch. I wanted to see what it would take to bridge that gap. I'm not seeking to replace existing tooling,
rather to suggest some data collection and visualisation approaches that I think would be good to add to existing
tooling for TensorFlow and PyTorch. In the meantime, this toolkit is easy to incorporate into your existing
training pipeline.

I have intentionally focused on basic plotting via matplotlib so that I could get away from any limitations inherent
within the existing plotting tools such as that provided by TensorBoard, Weights & Biases, etc.

Anyone is free to take what's here and use as a basis for their own work, though attribution would be greatly
appreciated.

## Importing

The following code can be used to import the package into a Jupiter notebook:

```python
import os
import sys
if not os.path.isdir('training-instrumentation-toolkit'):
  !git clone https://github.com/malcolmlett/training-instrumentation-toolkit.git
sys.path.append('training-instrumentation-toolkit')
import training_instrumentation as tinstr
import training_explainer as texpl
```

The toolkit is not currently published as a python package.

## Quickstart

The following runs a model training while capturing the metrics needed to generate the image above.

```python
import tensorflow as tf
import training_instrumentation as tinstr

def my_model():
    ....
    
def my_dataset():
    ...

variables = tinstr.VariableHistoryCallback(per_step=True)
gradients = tinstr.GradientHistoryCallback(per_step=True)
outputs = tinstr.LayerOutputHistoryCallback(per_step=True)
output_gradients = tinstr.LayerOutputGradientHistoryCallback(per_step=True)

model = my_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse', 'binary_crossentropy'])
dataset = my_dataset()
history = tinstr.fit(model, dataset.batch(32), epochs=10, callbacks=[
    variables, gradients, outputs, output_gradients, tinstr.HistoryStats(per_step=True)])

tinstr.plot_history_overview([history, variables, gradients, outputs, output_gradients])
```

## Docs

The following Jupiter notebooks demonstrate the core functionality of the _training-instrumentation-toolkit_,
including how you can easily use it as a basis for collecting your metrics. They include links to open the
notebooks directly within Google Colab so you can run them yourself.

1. [Instrumentation](doc/instrumenting.ipynb)
2. [Visualisations](doc/visualisations.ipynb)
3. [Train Explainer](doc/train-explainer.ipynb)

Detailed pydocs are available within the source code.

The above documentation notebooks are all found in the `doc` folder. If you navigate there you will also find other
experiments, including a series of notebooks that feed into blog posts.

**Blog Posts**

This toolkit is extensively used in a series of blog posts on improving instrumentation and visualisation of model dynamics during training:
* [Better ways to monitor NNs while training](https://medium.com/ai-advances/better-ways-to-monitor-nns-while-training-7c246867ca4f).
* [Neuron Death in ANNs](https://medium.com/ai-advances/neuron-death-in-anns-detecting-and-troubleshooting-4a7b5cc2f099)
* [Vanishing and Exploding Gradients](https://medium.com/ai-advances/neuron-death-in-anns-detecting-and-troubleshooting-4a7b5cc2f099)

## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation.
For citation information, please see the "Cite this repository" button on the right.
