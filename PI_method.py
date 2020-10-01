import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def crossplot(df, x,y,z, xmin = None, xmax = None, ymin = None, ymax = None):
    sc1 = plt.scatter(df[x],df[y], c=df[z],cmap='rainbow',s=3,alpha=0.75)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    plt.colorbar(sc1)
    plt.show()

def calc_c(df, c_range, a, b, cor,find_max = True, plot = False):
    corr_c = []
    for c in c_range:
        fun = df[a] - c * df[b]
        corr_matrix = np.corrcoef(fun,df[cor])
        corr_c.append(corr_matrix[0,1])
    if find_max:
        ind_max = np.argmax(corr_c)
    else:
        ind_max = np.argmin(corr_c)
    c_best = c_range[ind_max]
    if plot:
        plt.plot(c_range,corr_c)
        plt.xlabel('c')
        plt.ylabel('correlation')
        plt.vlines(c_best,min(corr_c),max(corr_c))
        plt.title("c_best: {}".format(c_best))
        plt.show()
    return c_best

filepath = "./del2_logexport.csv"
max_depth = 1500
max_li = 1300
max_fii = 3800

df = pd.read_csv(
    filepath_or_buffer = filepath,
    header = 0,
)

#print(df)
df = df.dropna()

df = df[df['MD'] > max_depth]

#print(df)

crossplot(df, "AI", "SI", "GR", 6500,11500,2500,6500)

#LI = AI - c * SI

c_range = np.arange(0.5,3,0.01)
c_best = calc_c(df,c_range,"AI", "SI", "GR", plot=True)
df["LI"] = df['AI'] - c_best * df['SI']

crossplot(df, "LI","GR","GR")

df = df[df["LI"] < max_li]

crossplot(df,"LI","GR","GR")

c_range = np.arange(-2,5,0.01)
c_best = calc_c(df,c_range,"AI","SI","Phie",find_max = False, plot = True)

df["FII"] = df['AI'] - c_best * df['SI']

crossplot(df,"FII","Phie","Phie",)

df = df[df["FII"] < max_fii]

crossplot(df,"FII","Phie","Phie",)

#SW

c_range = np.arange(-2,5,0.01)
c_best = calc_c(df,c_range,"AI","SI","SW",plot=True)

df["GI"] = df['AI'] - c_best * df['SI']


crossplot(df,"GI","SW","Phie")
sc2 = plt.scatter(df["GI"],df['SW'], c=df['Phie'],cmap='rainbow',s=3,alpha=0.75)
plt.xlabel('GI')
plt.ylabel('SW')
#plt.xlim([-1000,1500])
#plt.ylim([2500,6500])
plt.colorbar(sc2)
plt.show()

plt.hist(df["MD"], 100)
plt.show()
