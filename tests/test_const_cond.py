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

neuron = Neuron(model='const_g', gL=0.1, gK=0.2, gNa=0)
I_ext = np.zeros(20000)
I_ext[7000:13000] = 10
stats = neuron.simulate(20000, 0.01, I_ext)

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot([x.T for x in stats.data], [x.Vm for x in stats.data])
ax.set_title('Dynamics with $I_{ext} = 10$')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Membrane potential V (mV)')
plt.show()