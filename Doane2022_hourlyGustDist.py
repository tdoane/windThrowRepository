import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
import pdb
import plotly.express as px
import scipy.stats as sts
from scipy.integrate import cumtrapz
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "text.usetex":True,
    "font.family": "serif",
    "font.serif":["Times"],
    "font.size": 16})

grays = plt.get_cmap('hsv')

##Get Data
fPath = "MorganMonroeWind.csv"
wind = pd.read_csv(fPath, skiprows=2)
wind = wind.replace({-9999: None})
wS = wind['WS_1_1_1']
wDir = wind['WD_1_1_1']


##In this section, we fit hourly data to a mixture model involving Gumbel and Weibull distributions
wSpeed = wS.to_numpy(dtype = 'float')
wSpeed = wSpeed[np.logical_not(np.isnan(wSpeed))]
fit = sts.gumbel_r.fit(wSpeed) #Fit Gumbel to wind speed data
fitW = sts.weibull_min.fit(wSpeed) #Fit Weibull to wind speed data
wArray = np.arange(0., 15., 0.1)
a = fit[0] #Gumbel Parameters
b = fit[1]
print(fit)
print(fitW)
z = (wArray-a)/b #Convenient variable for the Gumbel
aW = fitW[0] #Weibull parameters
bW = fitW[1]
cW = fitW[2]

weibPDF = aW/cW*(wArray/cW)**(aW-1)*np.exp(-(wArray/cW)**aW)
weib = np.exp(-(wArray/cW)**aW)
gumPDF = 1/b*np.exp(-z - np.exp(-z))
gum = 1- np.exp(-np.exp(-z))

rank = 1 - np.arange(0, len(wSpeed))/(len(wSpeed)+1)
#figP, (axP, axG, axE) = plt.subplots(1, 3, figsize = (14.5, 4.))
figP = plt.figure(figsize = (13., 4.))
gs1 = gridspec.GridSpec(1,3)
axP = figP.add_subplot(gs1[0])
axP.semilogy(np.sort(wSpeed), rank, '-b', label = 'Data')
axP.semilogy(wArray, gum, '-r', label = 'Gumbel')
axP.semilogy(wArray, weib, 'c', label = 'Weibull')
axP.semilogy(wArray, 0.75*weib + 0.25*gum, '-k', label = 'Mixture Model')
axP.set_ylim([0.00001, 1.1])
axP.set_xlabel('Hourly Wind Speed [m s$^{-1}$]')
axP.set_ylabel('Exceedance Probability')
axP.legend(loc = 1, fontsize = 11)
#axP.text(-0.1, 1.1, 'a)')
axP2 = plt.axes([0,0,1,1])
ip = InsetPosition(axP, [0.15, 0.10, 0.45, 0.45])
axP2.set_yscale('log')
axP2.tick_params(axis='x', labelsize = 11)
axP2.tick_params(axis='y', labelsize=11)

axP2.set_axes_locator(ip)

#plt.figure()
axP2.hist(wSpeed, bins = 50, density = True)
axP2.plot(wArray, gumPDF, '-r')
axP2.plot(wArray, weibPDF, '-c')
axP2.plot(wArray, 0.75*weibPDF + 0.25*gumPDF, '-k')
axP2.patch.set_alpha(0.5)
axP2.set_ylim(0.000001, 0.4)


##Import gust data paired with hourly data. Gusts are the maximum gust that occur within that hour.
Gust = np.load('MMGust.npy')
Hourly = np.load('MMHourly.npy')
fig, ax = plt.subplots()
ax2 = ax.twinx()
collectMeanG = []
collectVarG = []

#evaluate the relationship between hourly averaged wind speed and gust speed. 
for i in np.arange(0, 10):
    ind = np.where(Hourly>i)
    hTemp = Hourly[ind]
    gTemp = Gust[ind]
    ind2 = np.where(hTemp<i+1.)
    hTemp = hTemp[ind2]
    gTemp = gTemp[ind2]
    gVar = np.var(gTemp)
    collectMeanG = np.append(collectMeanG, np.mean(gTemp))
    collectVarG = np.append(collectVarG, gVar)

    print(i, np.mean(gTemp), np.std(gTemp), len(gTemp))
    ax.plot(i+0.5, np.mean(gTemp), 'ok')
    ax2.plot(i, np.var(gTemp), '+r')
    
xTemp = np.arange(0, np.max(hTemp))
yTemp = 1.9*xTemp
     
plt.figure()
plt.plot(Hourly, Gust, 'ok', alpha= 0.1)
plt.xlabel('Hourly Averaged Wind Speed [m s$^{-1}$]')
plt.ylabel('Gust Wind Speed [m s$^{-1}$]')
plt.plot(xTemp, yTemp, '-', color = 'salmon')
plt.grid()

##-----------------------------------------------------------------##
##Construct distribution of squared gust speed (drag)
dD=5. 
drag = np.arange(2.5, 10000., dD) 
gust = np.sqrt(drag)

w = np.arange(0.25, 25., 0.1)

G, W = np.meshgrid(drag, w)
Z = (W-a)/b #convenient conversion for the Gumbel distribution
fact = G**(-1/2.)/2. #conversion factor to convert between squared gust and gust for in front of probability distribution
GgivenW = fact*1/np.sqrt(2*np.pi*2.9)*np.exp(-(np.sqrt(G)-1.96*W-0.41)**2/(2*2.9)) #parameters for this are determined by regressing collectMeanG against hourly wind speed and getting the average variance about that relationship with collectVarG.

temp = np.trapz(GgivenW, G, axis=1)
temp2 = np.tile(temp, (len(drag), 1)).T
GgivenW/=temp2 #This normalizes the distribution to 1 for when W is small or very large. In this case, the variance of g would stretch in to either negative values or beyond the domain. Both of these are non issues, so we normalize the distribution to 1 in these cases. 

weibWeight = 0.75
margW = (1-weibWeight)*1/b*np.exp(-Z - np.exp(-Z)) + weibWeight*aW/cW*(W/cW)**(aW-1)*np.exp(-(W/cW)**aW)
jointPDF = margW*GgivenW
margG = np.trapz(jointPDF, W, axis=0)
FG = np.cumsum(margG)*dD
FG/=np.max(FG) # The distribution sums to VERY close to unity, but because we raise this to a very large numner, it needs to be exacltly unity. 

nHours = 24*365
extPDF = nHours*FG**(nHours)*margG
extPDF/=np.trapz(extPDF, drag) #This also integrates very closely to unity, but is not quite unity. THis just ensures it, but does not change the shape.

##Plotting 
plt.figure()
plt.hist(Gust**2, bins = 100, density = True)
plt.plot(drag, margG)

axE = figP.add_subplot(gs1[2])
axE.plot(drag, extPDF, '-k')
axE.set_xlabel('Squared Gust Speed [m$^2$ s$^{-2}$]')
axE.set_ylabel('Probability Density [m$^{-2}$ s$^{2}$]')
axE.add_patch(Rectangle((225, 0), 41**2, 0.00015, color = 'orange'))
axE.text(500, 0.0, "derechos")
axE.set_xlim([0, 4000])
#axE.text(-0.1, 1.1, 'b)')

axG = figP.add_subplot(gs1[1])
axG.plot(Hourly, Gust**2, 'ok', alpha = 0.1, zorder =1, markersize = 4.)
axG.set_ylabel('Square of Gust Speed [m$^2$ s$^{-2}$]')
axG.set_xlim([0, 14])
axG.set_ylim([0, 1500])
axG.set_xlabel('Hourly Wind Speed [m s$^{-1}$]')
axG.contour(W, G, GgivenW, 50, colors = 'yellow', zorder = 2)
#axG.grid()
#axG.text(-0.1, 1.1, 'c)')
gs1.tight_layout(figP)

plt.show(block = False)
pdb.set_trace()


