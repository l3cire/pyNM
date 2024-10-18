from pyneural import NeuralModel
from pyneural.input_current import ConstInputCurrent, NoisyConstInputCurrent
from pyneural.neuron_models import LIFNeuron
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

neuron = LIFNeuron()
I_ext = NoisyConstInputCurrent(I=20, std=15)
stats = NeuralModel().simulate_neuron(neuron, 100000, 0.01, I_ext)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4))

assert isinstance(ax1, Axes)
ax1.plot([x.T for x in stats.step_data], [x.Vm for x in stats.step_data], color='b')
ax1.set_ylabel("Membrane Potential (mV)")

assert isinstance(ax2, Axes)
ax2.plot([stats.step_data[i].T for i in range(0, len(stats.step_data), 100)], [stats.step_data[i].I_ext for i in range(0, len(stats.step_data), 100)], color='r')
ax2.set_ylabel("External Current (µA/cm²)")

plt.show()

