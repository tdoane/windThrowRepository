import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import scipy.optimize as opt
from scipy.special import gamma
import pdb
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

##allow LaTeX syntaxt to be used and change the font size and family.
from matplotlib import rc
rc('text', usetex = True)
plt.rcParams.update({'font.size':14})
plt.rcParams.update({'font.family': 'serif'})

##Load probability density of topographic roughness
var = np.load('varVals.npy') #variance values
fVar = np.load('pdfZVar.npy') #probability density
cumKDE = cumtrapz(fVar, var) #cumulative probability density

eVar = np.trapz(var*fVar, var) #Mean topographic roughness (mu_r)
sigVar = np.trapz(var**2*fVar, var) - eVar**2 #variance of topographic roughness (sigma^2_r)

BeaufortFujita = np.array([13.9, 17.2, 20.8, 24.5, 28.5, 38., 49., 60., 74.]) ##Beafort-Fujita Scale in km/hr 
BeaufortFujita**=2

##Average pit-mound couplet parameters (from Doane et al., 2021)
A = 0.68#amplitude-squared
phi = 0.83 #aspect ratio
l = 1.5 # characteristic lengthscale
w = l/phi #characteristic width
D = 0.01 #Diffusivity, can play around with this, but one needs to iteratively adjust resistance parameters below

##Production rate probability parameters:
dP = 1.
p = np.arange(0., 250., dP)
dP = p[1]

##Extract information from wind speed data
col = 'salmon'

##f(x) and g(x) are used together to solve for the parameters of a Weibull distribution, given the mean and variance. We only define the two functions here. They are used later.
def f(x):
    temp = gamma(1 + 2/x)/((gamma(1 + 1/x))**2)-1-varPMeas/(muPMeas**2)
    return(temp)

def g(x):
    return(x*gamma(1 + 1/k) - muPMeas)

##CREATE JOINT DISTRIBUTION
##This function carries out the relationships between the marginal, conditional, and joint probability distributions that create the probability distribution of wind throw production rates. 
#fD is a marginal distribution and is the probabilty density of driving forces. 
#FR is the resistance function (strength of trees) and is the conditional distribution of the probability of wind throw given a driving force. 
#sqGust is the square of the gust speed (driving force and force that trees must withstand)
#num trees is the number of trees per hectare (we estimate 250) for southern Indiana.
def jointMake(fD, FR, sqGust, numTrees):
    joint = np.zeros((numTrees, len(sqGust)))#initialize the joint distribution to have the correct shape
    num = np.arange(0, numTrees, 1)#a numpy array of trees. 
    for i in range(len(sqGust)): #loop through squared gust speeds
        p = FR[i] #probability of tree toppling, given that gust
        temp = fD[i] #probability of that gust occuring
        binTemp = binom.pmf(num, numTrees-1, p) #binomial distribution where probability is set by p, and evaluate for numTrees
        joint[:,i] = temp*binTemp #fill in the joint distribution
    
    fP = np.trapz(joint, sqGust, axis=-1)#probability distributoin of wind throw production rates f_p(p)

    return(joint, fP)

##-------------------------------------------------------------------------------------------##
#Construct extreme value distribution of wind speeds

##Values from 'hourlyGustDist.py'
aW = 2.5909 #Shape parameter for hourly Weibull distribution
cW = 3.9541 #Scale parameter for hourly Weibull distribution

a = 2.83215 
b = 1.25711

dD=10.
drag = np.arange(.1, 74.**2, dD) #array of gust speeds
gust = np.sqrt(drag)

w0 = np.arange(0.25, 25, 0.5) #array of hourly wind speeds

#Here we construct joint and conditional distributions of hourly wind speed and gust speed GIVEN hourly wind speed
G, W = np.meshgrid(drag, w0) #construct two dimensional array of gustXhourly
Z = (W-a)/b #New variable for Gumbel distribution
fact = 1/2.*(G**(-1/2.)) #For converting the distribution of gust speeds to squared gust speeds
GgivenW = fact*1/np.sqrt(2*np.pi*2.9)*np.exp(-(np.sqrt(G)-1.96*W)**2/(2*2.9))#the distributon of gust GIVEN hourly wind speed

##These lines ensure that the probability distribution integrates to unity. For low wind speeds, the conditional distribution does not sum to unity because we cannot include negative terms. We renormalize for those cases here.
temp = np.trapz(GgivenW, G, axis=1)
temp2 = np.tile(temp, (len(drag), 1)).T
GgivenW/=temp2 

weibWeight = 0.75 #weight of hourly distribution that is Weibull
margW = (1-weibWeight)*1/b*np.exp(-Z - np.exp(-Z)) + weibWeight*aW/cW*(W/cW)**(aW-1)*np.exp(-(W/cW)**aW)#The weibull-gumbel model
jointPDF = margW*GgivenW#Joint pdf is the product of the marginal and conditional distributions
margG = np.trapz(jointPDF, W, axis=0) #marginal distribution of gust speeds is the integral over all hourly distributions

##Now we form the extreme value distribution of wind gusts
FG = np.cumsum(margG)*dD 
FG/=np.max(FG) # The distribution sums to VERY close to unity, but because we raise this to a very large numner, it needs to be exactly unity. 
nHours = 24*365 #Numer of hours in a year (samples)
fD = nHours*FG**(nHours)*margG #Construct the extreme value distribution
fD/=np.trapz(fD, drag) #The distribution integrates very close to unity, this ensures it

##-----------------------------------------------------------------------------------------------------##

##Distribution of Resistance Strengths
muR = np.arange(1700, 1850,10) #Initial guess for mean of the distribution of resistance function
sigmaR = np.arange(270, 370, 10) #Initial guess for standard deviation of 
errorSurf = np.ones((len(muR), len(sigmaR)))*np.nan #
minError= 1000 

SIGMA, MU = np.meshgrid(sigmaR, muR)##Make array of standard deviation and mean values

C = A**2*l**2*w**2*np.pi/32 #Geometric coefficient

muPMeas = eVar*4*D*(phi**2 + phi)/(A**2*l**2*np.pi)*10000 ##Equation 2 from Doane et al., 2022
varPMeas = sigVar*48*D*phi**4/(A**4*np.pi**2*l**2/2*(phi**2 + phi - 0.2))*10000**2 - muPMeas**2 #Equation 3 form Doane et al., 2022

sigPMeas = np.sqrt(varPMeas) 
varWeight = muPMeas/sigPMeas #A weight that weighs the relative importance of the mean and standard deviation. The error surface is defined as the difference between the sums of the weighted mean and standard deviations.

##Perform method of moments
k = opt.brentq(f, 0.1,2.) #solve for shape parameter of Weibull distribution for fp(p)
lam = opt.brentq(g, 0.0, 5.) #solve for scale parameter of Weibull distribution for fp(p)
print(k, lam)

fPWeib = k/(lam**k)*(p)**(k-1)*np.exp(-(p/lam)**k) #Probability distribution of wind throw production rate based on MoM
fPWeib[0] =1- np.sum(fPWeib[1:]) #fP(0) is inf because k<1. We assign the value of fP(0)


##Now we iterate through combinations of mean and standard deviation for the distribution of tree strengths. 

muBest = 10000 #Placeholder for best mean value for resistance functions
sigmaBest = 10000 #Placeholder for best standard deviation value for resistance functions
for i in np.arange(len(muR)): #Loop through mean values
    for j in np.arange(len(sigmaR)): #Loop through standard deviations
        mu =muR[i] 
        sigma = sigmaR[j]
        FR = 0.5*(1 + erf((drag - mu)/(np.sqrt(2)*sigma)))#Cumulative distribution function of tree strengths OR conditional probability of wind throw given a drag force
        joint, fP = jointMake(fD, FR, drag, 250) #Construct the joint and marginal (fp(p)) distribution
        print(np.sum(fP)) #print to check that fp(p) integrates to unity 
        muTemp = np.sum(p*fP) #Calculate mean production rate
        varTemp = (np.sum(p**2*fP)-muTemp**2) #calculate variance of production rate
        errorTemp = (1-varWeight)*(muTemp - muPMeas)**2 + (varWeight)*(np.sqrt(varTemp) - sigPMeas)**2 #Error between measured and modeled mean + standard deviation of production rates.
        errorSurf[i,j] = errorTemp #Add the value to the error surface
        
        if errorTemp<minError: #if errorTemp is the lowest so far, note the mean and standard deviatons
            muBest=mu
            sigmaBest=sigma
            minError=errorTemp
            
muArr=[]
sigArr=[]

##Plot the error surface as a function of the mean and standard deviations of reistance functions
plt.figure(1)
plt.pcolor(MU, SIGMA, np.log(errorSurf))
plt.xlabel('Mean Resistance Force')
plt.ylabel('Standard Deviation of Resistance Force')
plt.colorbar()
plt.plot(muBest, sigmaBest, '*', color = col)

## Plot overlapping distributions
col = 'cornflowerblue'
fig2, ax = plt.subplots()
ax.plot(drag, fD, '-k')
ax2 = ax.twinx()
ax2.plot(drag, FR, '-', color = col)
ax.set_ylabel('Probability Density [m$^{-2}$ sec$^{2}$]')
ax.set_xlabel('Squared Wind Speed [m$^2$ sec$^{-2}$]')
ax2.set_ylabel('Probability Relative Resistances Strengths' )

#plot Beaufort-Fujita scale along the x-axis
cmap2 = plt.get_cmap('Reds')
for i in np.arange(len(BeaufortFujita)-1):
    cols = cmap2(i/(len(BeaufortFujita)-1))
    ax2.add_patch(Rectangle((BeaufortFujita[i], 0.), BeaufortFujita[i+1]-BeaufortFujita[i], 0.05, facecolor = cols))

##Plot probability distribution of wind throw production rates by mechanical model and by overlap model
fig3, ax3 = plt.subplots()
FR = 0.5*(1 + erf((drag -muBest)/(np.sqrt(2)*sigmaBest)))
joint, fP = jointMake(fD, FR, drag, 250)
muTemp = np.sum(p*fP)
varTemp = (np.sum(p**2*fP)-muTemp**2)

print(muPMeas, muTemp)
print(varPMeas, varTemp)

ax3.plot(p, fP, '-', color = col) #plot mechanical model
ax3.plot(p, fPWeib, '-k') #plot weibull model (Method-of-Moments)
    
plt.grid()
ax3.set_ylabel('Probability Density [m$^2$ yr$^1$]')
ax3.set_xlabel('Tree Throw Production Rate [m$^{-2}$ yr$^{-1}$]')

plt.show(block = False)
pdb.set_trace()
