import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt
data = np.random.normal(0.5, 0.2, 1000)

h = Hist(hist.axis.Regular(bins=10, start=0, stop=1, name="x"))
h.fill(data)
h.plot()
plt.show()

datax=np.random.normal(0.5, 0.2, 1000)
datay=np.random.normal(0.5, 0.2, 1000)
dataz=np.random.normal(0.5, 0.2, 1000)

hist3D = (
    Hist.new.Regular(100, 0, 10, circular=True, name="x")
    .Regular(10, 0.0, 10.0, name="y")
    .Variable([1, 2, 3, 4, 5, 5.5, 6], name="z")
    .Weight()
)

hist3D.fill(datax,datay,dataz)
hist_projected=hist3D.project("x")
hist_projected.plot()
plt.show()

print("code successfully implimented")
