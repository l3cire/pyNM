from src.Neuron import Neuron
from src.NeuralModel import NeuralModel
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

neuron_hh = Neuron(model='hh')
neuron_lif = Neuron(model='lif', params={'C_m':1})
model = NeuralModel()
I = np.linspace(0, 30, 100)
f_hh = model.get_fi_curve(neuron_hh, I)
f_lif = model.get_fi_curve(neuron_lif, I)

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
assert isinstance(ax, Axes)
ax.plot(I, f_hh, label = 'Hodgkin and Huxley')
ax.plot(I, f_lif, label = 'Leaky Integrate and Fire')
ax.set_title('Dynamics with $I_{ext} = 10$')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Membrane potential V (mV)')
ax.legend()
plt.show()

