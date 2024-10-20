from pyneural import NeuralModel
from pyneural.input_current import ConstInputCurrent
from pyneural.neuron_models import ConstCondNeuronGroup
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

neuron = ConstCondNeuronGroup(N_neurons = 1, params={'gL':0.1, 'gK':0.2, 'gNa':0})
I_ext = ConstInputCurrent(1, 75, 125, 10)
stats = NeuralModel().simulate_neurons(neuron, 20000, 0.01, I_ext)

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
assert isinstance(ax, Axes)
ax.plot([x.T for x in stats.step_data], [x.Vm[0] for x in stats.step_data])
ax.set_title('Dynamics with $I_{ext} = 10$')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Membrane potential V (mV)')
plt.show()

