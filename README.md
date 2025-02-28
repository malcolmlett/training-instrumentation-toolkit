# Training Instrumentation Toolkit
TensorFlow addons for instrumenting training loops in order to collect and analyse detailed metrics on gradients,
weights, and layer activations. Improves development turnaround time by granting access to more subtle indicators of problems than mere loss curves.
Aids in troubleshooting by providing insight into many aspects of model training in one go, quickly highlighting if any of the many common issues are occurring.

Example visualisation of metrics gathered during training:

![training overview plot](doc/training-overview-example.png)

## Importing

The toolkit is not currently published as a python package.

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

See `doc` folder for Jupiter notebooks with explanation of functionality.
