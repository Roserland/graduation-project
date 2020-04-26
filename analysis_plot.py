import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./convergence.csv')[:100]
print(data)

loss = data[data['metric'] == 'loss']['value'].tolist()
fnr  = data[data['metric'] == 'fnr']['value'].tolist()
fpr  = data[data['metric'] == 'fpr']['value'].tolist()
err  = data[data['metric'] == 'error']['value'].tolist()

print(loss)

x_tick_loss = list(range(0, len(loss)))
x_tick_etc  = list(range(1, len(loss)+1, 2))
print(x_tick_loss)
print(x_tick_etc)

assert len(x_tick_etc) == len(fpr)

plt.plot(x_tick_loss, loss, label='loss')
plt.plot(x_tick_etc, err, label='error')
plt.plot(x_tick_etc, fpr, label='fpr')
plt.plot(x_tick_etc, fnr, label='fnr')
plt.legend()
plt.show()
