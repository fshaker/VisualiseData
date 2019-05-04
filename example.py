import datetime
import matplotlib.pyplot as plt
import pytz

x = [1,2,3]
y = [4,3,7]
rows = 6
cols = 4
s = 10
utc_now = pytz.utc.localize(datetime.datetime.utcnow())
pst_now = utc_now.astimezone(pytz.timezone("Canada/Eastern"))
fig = plt.figure(figsize=(10, 5))
#ax = fig.add_axes([0.1,0.2, 0.8, 0.6])
ax = fig.add_subplot(2,2,1)
ax.set_xticks([],[])
ax.set_yticks([],[])
ax.set_title("Title1", fontsize='8', color='r')
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
fig.suptitle("This is a really long string that I'd rather have wrapped so that it "
     "doesn't go outside of the figure, but if it's long enough it will go "
     "off the top or bottom!", fontsize='8', wrap=True)
# fig.suptitle("This is a subtitle %i on %s"%(s,str(pst_now.isoformat())), fontsize='8')
# ax1 = fig.add_axes([0.1,0.1,0.3,0.3])
# ax1.plot(x,y)
# ax1.set_title("This is a test plot", fontsize='small')
fig.show()
