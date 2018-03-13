import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

csv = np.genfromtxt('/home/xn3cr0nx/Scrivania/Datasets/csv/JKHH/sa/cnt.csv', delimiter=',')

# fig = plt.figure()
# plt.plot(csv[0,:])

np_csv = np.array(csv)

# plt.figure(1)
# plt.subplot(111)
# plt.plot(np_csv[0,:])

# plt.subplot(112)
# plt.plot(np_csv[1,:])
# plt.show()

xlim, ylim = (0, 10000), (-1, 1)
fig = plt.figure(figsize=(18,7))
for column in range(csv.shape[1]):
	# fig.add_subplot(6, 6, column+1)
	ax = plt.subplot(6, 6, column+1)
	# plt.plot(csv[:,column], lw=0.2)
	plt.plot(csv[:,column], lw=0.2)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
plt.suptitle('Features')
plt.tight_layout(pad=0)
plt.show()