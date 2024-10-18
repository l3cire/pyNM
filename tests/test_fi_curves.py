from pyneural.neuron_models import HHNeuron
from pyneural.neuron_models import LIFNeuron
from pyneural import NeuralModel
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns

sns.set_theme(style="ticks",
              palette="Set1",
              font_scale=1,
              rc={
                  "axes.spines.right": False,
                  "axes.spines.top": False,
              },
              )

neuron_lif = LIFNeuron()
model = NeuralModel()
I = np.linspace(0, 30, 100)
f_lif = model.get_fi_curve(neuron_lif, I)

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
assert isinstance(ax, Axes)
ax.plot(I, f_lif, label = 'Leaky Integrate and Fire')
ax.set_title('Dynamics with $I_{ext} = 10$')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Membrane potential V (mV)')
ax.legend()
plt.show()

