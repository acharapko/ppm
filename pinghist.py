import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#ping = np.loadtxt("local_ping.txt")
ping = np.loadtxt("remote_ping.txt")

mu = np.average(ping)
sigma = np.std(ping)
# the histogram of the data
n, bins, patches = plt.hist(ping, 40, normed=1, facecolor='green')

# add a 'best fit' line
#print n
y = mlab.normpdf(bins, mu, sigma)
l = plt.plot(bins, y, 'r-', linewidth=1)
#print y
plt.xlabel('Local Ping Latency (ms)')
plt.ylabel('Probability')
plt.title(r'$\mathrm{AWS\ Latency:}\ \mu='+str(mu)+',\ \sigma='+str(sigma)+'$')

plt.grid(True)

plt.show()
