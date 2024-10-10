from src.Neuron import Neuron
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks",
              palette="Set1",
              font_scale=1,
              rc={
                  "axes.spines.right": False,
                  "axes.spines.top": False,
              },
              )

neuron = Neuron(model='hh')
I_ext = np.zeros(20000)
I_ext[7500:12500] = 10
stats = neuron.simulate(20000, 0.01, I_ext)


plt.figure(figsize=(10, 8))

ax1 = plt.subplot(411)
ax1.plot([x.T for x in stats.data], [x.Vm for x in stats.data], color='b')
ax1.set_ylabel("Membrane Potential (mV)")

ax2 = plt.subplot(412)
ax2.plot([x.T for x in stats.data], [x.I_ext for x in stats.data], color='r')
ax2.set_ylabel("External Current (µA/cm²)")

ax3 = plt.subplot(413, sharex=ax1)
ax3.plot([x.T for x in stats.data], [x.gate_n for x in stats.data], label='n')
ax3.plot([x.T for x in stats.data], [x.gate_m for x in stats.data], label='m')
ax3.plot([x.T for x in stats.data], [x.gate_h for x in stats.data], label='h')
ax3.set_ylabel("Activation (frac)")
ax3.legend()

ax4 = plt.subplot(414, sharex=ax1)
ax4.plot([x.T for x in stats.data], [x.I_Na for x in stats.data], label='Na channels')
ax4.plot([x.T for x in stats.data], [x.I_K for x in stats.data], label='K channels')
ax4.plot([x.T for x in stats.data], [x.I_leak for x in stats.data], label='Leak channels')
ax4.set_ylabel("Current (µA/cm²)")
ax4.set_xlabel("Simulation Time (ms)")
ax4.legend()

plt.tight_layout()
plt.show()
