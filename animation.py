import matplotlib.pyplot as plt
import matplotlib.animation

x = [1,1]
y = [1,2]

fig, (ax1,ax2) = plt.subplots(nrows=2, sharex=True, sharey=True)
line1, = ax1.plot(x)
line2, = ax2.plot(y)
ax1.set_xlim(-1,18)
ax1.set_ylim(-400,3000)


def update(i):
    x.append(x[-1]+x[-2])
    line1.set_data(range(len(x)), x)
    y.append(y[-1]+y[-2])
    line2.set_data(range(len(y)), y)

ani = matplotlib.animation.FuncAnimation(fig, update, frames=14, repeat=False)
plt.show()
