import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
def regression_results(df,x_attr,y_attr,PLOT=False,normScale=False,ylim01=False):
    print "---------------------------------------------"
    print "Regression x={0};y={1}".format(x_attr,y_attr)
    regr = linear_model.LinearRegression()
    msk = np.random.rand(len(df)) < 0.8
    Xtrain = df[x_attr][msk].as_matrix()
    Xtest = df[x_attr][~msk].as_matrix()
    if normScale:
        Xtrain = StandardScaler().fit_transform(Xtrain)
        Xtest = StandardScaler().fit_transform(Xtest)
    
    Ytrain = df[y_attr][msk].as_matrix()
    Ytest = df[y_attr][~msk].as_matrix()
    regr.fit(Xtrain,Ytrain)
    # The coefficients
    print 'Coefficients: ', regr.coef_
    # The mean squared error
    print("Mean squared error: %.2f"
          % np.mean((regr.predict(Xtest) - Ytest) ** 2))
    print('R^2: %.2f' % regr.score(Xtest, Ytest))
    if PLOT:
        plt.figure()
        plt.plot(Xtest,Ytest,'o')
        plt.plot(Xtest,regr.predict(Xtest),'r-')
        if ylim01: plt.ylim(0,1)
        plt.title("MSE=%.2f"% np.mean((regr.predict(Xtest) - Ytest) ** 2)+'; R^2=%.2f' % regr.score(Xtest, Ytest))
        plt.xlabel(x_attr[0],fontsize=13)
        plt.ylabel(y_attr[0],fontsize=13)
def bucketize(data,Nbuckets):
    delta = (max(data)-min(data))/Nbuckets
    bucketized_data = []
    start=0
    end=delta
    for val in data:
        start=0
        end=delta
        #print "val: ",val
        for i in range(Nbuckets+3):
            #print start,end
            if val<=start and i==0:
                bucketized_data.append(0)
                break
            if val>=start and val<end:
                #print"added"
                bucketized_data.append(start)
                break
            else:
                start=end
                end+=delta
            if i==Nbuckets+2:
                bucketized_data.append(end)
    #print len(bucketized_data),len(data)
    assert len(bucketized_data)==len(data)
    return bucketized_data
    
def scatterplot(df,x_attr,y_attr,z_attr,z_data="",bucketize=True,cmap = plt.cm.rainbow,zlim01=False):
    fig=plt.figure()

    plt.xlim(0,1.03)
    plt.ylim(0,1.03)
    plt.xlabel(x_attr,fontsize=14)
    plt.ylabel(y_attr,fontsize=14)
    plt.title(z_attr,fontsize=15)
    if bucketize:
        bounds = np.sort(list(set(bucketized_data)))
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        plt.scatter(df[x_attr],df[y_attr],c=z_data,edgecolors='none',alpha=0.7,cmap=cmap,norm=norm)        
        ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
        matplotlib.colorbar.ColorbarBase(ax2,cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds)#, format='%1i')
    else:
        plt.scatter(df[x_attr],df[y_attr],c=df[z_attr],edgecolors='none',alpha=0.7,cmap=cmap)
        plt.colorbar()
        if zlim01: plt.clim(0,1)

def plot_attr_histo(attr):
    data = df[attr]
    a = plt.hist(data,bins=50,normed=True)
    avg = np.mean(data)
    var = np.var(data)
    plt.title("Normalized {0} distribution [N={1}; bins=50]".format(attr,len(data)))
    plt.suptitle("mu={0:.3f};std={1:.3f}".format(avg,var))
    pdf_x = np.linspace(np.min(data),np.max(data),100)
    pdf_y = 1.0/np.sqrt(2*np.pi*var)*np.exp(-0.5*(pdf_x-avg)**2/var)
    plt.plot(pdf_x,pdf_y,'--',color='red',linewidth=3)