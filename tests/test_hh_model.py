from pyneural import NeuralModel
from pyneural.input_current import ConstInputCurrent, NoisyConstInputCurrent
from pyneural.neuron_models import HHNeuronGroup
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

neuron = HHNeuronGroup(N_neurons=1)
I_ext = NoisyConstInputCurrent(N_neurons=1, I=5, std=10)
stats = NeuralModel().simulate_neurons(neuron, 100000, 0.01, I_ext)

plt.figure(figsize=(10, 8))

ax1 = plt.subplot(411)
assert isinstance(ax1, Axes)
ax1.plot([x.T for x in stats.step_data], [x.Vm for x in stats.step_data], color='b')
ax1.set_ylabel("Membrane Potential (mV)")

ax2 = plt.subplot(412)
assert isinstance(ax2, Axes)
ax2.plot([stats.step_data[i].T for i in range(0, len(stats.step_data), 100)], [stats.step_data[i].I_ext for i in range(0, len(stats.step_data), 100)], color='r')
ax2.set_ylabel("External Current (µA/cm²)")

ax3 = plt.subplot(413, sharex=ax1)
assert isinstance(ax3, Axes)
ax3.plot([x.T for x in stats.step_data], [x.gate_n for x in stats.step_data], label='n')
ax3.plot([x.T for x in stats.step_data], [x.gate_m for x in stats.step_data], label='m')
ax3.plot([x.T for x in stats.step_data], [x.gate_h for x in stats.step_data], label='h')
ax3.set_ylabel("Activation (frac)")
ax3.legend()

ax4 = plt.subplot(414, sharex=ax1)
assert isinstance(ax4, Axes)
ax4.plot([x.T for x in stats.step_data], [x.I_Na for x in stats.step_data], label='Na channels')
ax4.plot([x.T for x in stats.step_data], [x.I_K for x in stats.step_data], label='K channels')
ax4.plot([x.T for x in stats.step_data], [x.I_leak for x in stats.step_data], label='Leak channels')
ax4.set_ylabel("Current (µA/cm²)")
ax4.set_xlabel("Simulation Time (ms)")
ax4.legend()

plt.tight_layout()
plt.show()

