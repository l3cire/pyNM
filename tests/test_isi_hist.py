from pyneural import NeuralModel
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns

from pyneural.input_current import NoisyConstInputCurrent

sns.set_theme(style="ticks",
              palette="Set1",
              font_scale=1,
              rc={
                  "axes.spines.right": False,
                  "axes.spines.top": False,
              },
              )

model = NeuralModel('hh')
I = np.linspace(10, 25, 5);
ngroup = model.create_model(5)
stats = model.simulate_neurons(ngroup, 1000000, 1, NoisyConstInputCurrent(5, I= I, std= 15))

fig,ax = plt.subplots(5, figsize = (10,8))
for i in range(5):
    ax[i].hist(stats.spike_intervals[i], bins=100)
    ax[i].set_xlabel('ISI (ms)')
    ax[i].set_title('$\mu_I=$'+str(I[i]))
    ax[i].set_ylabel('Count')

fig.tight_layout()
plt.show()



