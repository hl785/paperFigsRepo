###############
### IMPORTS ###
###############

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX for text rendering
    "font.size": 12,      # Match LaTeX font size
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"
})
import matplotlib.patches as mpatches

###################################################################################################################################

###############
### HELPERS ###
###############
def toFigSize(pageFracWidth, aspectRatio):
    # A4 around 6.5 inches for \textwidth, check as some say 5.125 but looks worse.
    textwidth = 6.5
    figWidthInches = textwidth * pageFracWidth
    figHeightInches = figWidthInches / aspectRatio
    # Return in form x, y
    return figWidthInches, figHeightInches

def getDPI(fileType='.png'):
    return 'figure' if fileType == '.pdf' else 450

def getSplitNames():
    return ['Trotter', 'Strang', 'Yoshida', 'Learn5A', 'Learn8A', 'Learn8B']

def getSplitColours():
    return ['mediumvioletred', 'firebrick', 'peru', 'forestgreen', 'royalblue', 'lightskyblue']

def paramTransform(gamma):
    # Faster to not use jax?
    numParams = len(gamma)
    numBeta = numParams // 2
    numAlpha = numParams - numBeta

    alpha = gamma[:numAlpha]
    beta = gamma[numAlpha:]

    alphaSum = np.sum(alpha)
    betaSum = np.sum(beta)

    a = np.zeros(numParams + 2)
    b = np.zeros(numParams + 2)

    if numAlpha == numBeta + 1:
        a[:numAlpha] = alpha
        a[numAlpha] = 1 - 2 * alphaSum
        a[(numAlpha + 1) :] = np.flip(alpha, [0])

        b[:numBeta] = beta
        b[numBeta] = 0.5 - betaSum
        b[numBeta + 1] = 0.5 - betaSum
        b[(numBeta + 2) : numParams + 1] = np.flip(beta, [0])
    
    elif numAlpha == numBeta:
        a[:numAlpha] = alpha
        a[numAlpha] = 0.5 - alphaSum
        a[numAlpha + 1] = 0.5 - alphaSum
        a[(numAlpha + 2):] = np.flip(alpha, [0])

        b[:numBeta] = beta
        b[numAlpha] = 1 - 2 * betaSum
        b[(numBeta + 1):(numParams + 1)] = np.flip(beta, [0])

    else:
        assert False

    return a, b

def getLen5Order4Man():
    offset = 0.001
    zs = np.concatenate((np.linspace(-2.0, 0.0-offset, 4000),np.linspace(0.5+offset, 2.0, 3000)))
    inter1 = zs*(1.0-2*zs)
    inter2 = np.sqrt(3*np.power(zs, 4) - 3*np.power(zs, 3) + np.power(zs, 2) - 0.125*zs)
    inter3 = 3*zs*(8*np.power(zs, 3) - 8*np.power(zs, 2) + 1.0)
    ysPlus = (inter1 + inter2)/inter3
    ysMinus = (inter1 - inter2)/inter3
    inter4 = 4*np.power(zs, 2) - 4*zs + 1.0
    xsPlus = 1.0/6.0 - ysPlus*inter4
    xsMinus = 1.0/6.0 - ysMinus*inter4
    return xsPlus, xsMinus, ysPlus, ysMinus, zs

###################################################################################################################################

#################
### LOAD DATA ###
#################
def loadLossConvOrig():
    with np.load('lossConvOrig.npz') as data:
        unitNumStepsTrot   = data['unitNumStepsTrot']
        unitNumStepsStrang = data['unitNumStepsStrang']
        unitNumStepsYosh   = data['unitNumStepsYosh']

        unitNumStepsGammaLearn5A = data['unitNumStepsGammaLearn3A']
        unitNumStepsGammaLearn8A = data['unitNumStepsGammaLearn8A']
        unitNumStepsGammaLearn8B = data['unitNumStepsGammaLearn8B']

        lossTrotArr   = data['lossTrot'] 
        lossStrangArr = data['lossStrang'] 
        lossYoshArr   = data['lossYosh']

        lossLearn5AArr = data['lossLearn3A'] 
        lossLearn8AArr = data['lossLearn8A'] 
        lossLearn8BArr = data['lossLearn8B']

    # Number of Exponentials
    # TODO: Get from data
    timeRange = 10
    numExpsTrot   =   unitNumStepsTrot*timeRange*(2 - 0) + 0
    numExpsStrang = unitNumStepsStrang*timeRange*(4 - 2) + 1
    numExpsYosh   =   unitNumStepsYosh*timeRange*(8 - 2) + 1

    numExpsLearn5A = unitNumStepsGammaLearn5A*timeRange*(10 - 2) + 1
    numExpsLearn8A = unitNumStepsGammaLearn8A*timeRange*(16 - 2) + 1
    numExpsLearn8B = unitNumStepsGammaLearn8B*timeRange*(16 - 2) + 1

    # Calc avg loss
    lossTrot   = np.quantile(np.abs(lossTrotArr  ), 0.5, axis=0)
    lossStrang = np.quantile(np.abs(lossStrangArr), 0.5, axis=0)
    lossYosh   = np.quantile(np.abs(lossYoshArr  ), 0.5, axis=0)

    lossLearn5A = np.quantile(np.abs(lossLearn5AArr), 0.5, axis=0)
    lossLearn8A = np.quantile(np.abs(lossLearn8AArr), 0.5, axis=0)
    lossLearn8B = np.quantile(np.abs(lossLearn8BArr), 0.5, axis=0)

    # Calc std dev
    stdDev1 = 0.341
    stdDev2 = 0.477
    lowQTrot   = np.quantile(np.abs(lossTrotArr  ), 0.5 - stdDev1, axis=0)
    lowQStrang = np.quantile(np.abs(lossStrangArr), 0.5 - stdDev1, axis=0)
    lowQYosh   = np.quantile(np.abs(lossYoshArr  ), 0.5 - stdDev1, axis=0)

    lowQLearn5A = np.quantile(np.abs(lossLearn5AArr), 0.5 - stdDev1, axis=0)
    lowQLearn8A = np.quantile(np.abs(lossLearn8AArr), 0.5 - stdDev1, axis=0)
    lowQLearn8B = np.quantile(np.abs(lossLearn8BArr), 0.5 - stdDev1, axis=0)

    highQTrot   = np.quantile(np.abs(lossTrotArr  ), 0.5 + stdDev1, axis=0)
    highQStrang = np.quantile(np.abs(lossStrangArr), 0.5 + stdDev1, axis=0)
    highQYosh   = np.quantile(np.abs(lossYoshArr  ), 0.5 + stdDev1, axis=0)

    highQLearn5A = np.quantile(np.abs(lossLearn5AArr), 0.5 + stdDev1, axis=0)
    highQLearn8A = np.quantile(np.abs(lossLearn8AArr), 0.5 + stdDev1, axis=0)
    highQLearn8B = np.quantile(np.abs(lossLearn8BArr), 0.5 + stdDev1, axis=0)

    # Calc step sizes
    stepSizeTrot   =   unitNumStepsTrot**(-1.0)
    stepSizeStrang = unitNumStepsStrang**(-1.0)
    stepSizeYosh   =   unitNumStepsYosh**(-1.0)

    stepSizeLearn5A = unitNumStepsGammaLearn5A**(-1.0)
    stepSizeLearn8A = unitNumStepsGammaLearn8A**(-1.0)
    stepSizeLearn8B = unitNumStepsGammaLearn8B**(-1.0)
    
    return getSplitNames(), \
           [lossTrot, lossStrang, lossYosh, lossLearn5A, lossLearn8A, lossLearn8B], \
           [numExpsTrot, numExpsStrang, numExpsYosh, numExpsLearn5A, numExpsLearn8A, numExpsLearn8B], \
           [lowQTrot, lowQStrang, lowQYosh, lowQLearn5A, lowQLearn8A, lowQLearn8B], \
           [highQTrot, highQStrang, highQYosh, highQLearn5A, highQLearn8A, highQLearn8B], \
           [stepSizeTrot, stepSizeStrang, stepSizeYosh, stepSizeLearn5A, stepSizeLearn8A, stepSizeLearn8B]

def loadLossLandOrig():
    with np.load('lossLandOrig.npz') as data:
        gammaVals1 = data['possibleVals1']
        gammaVals2 = data['possibleVals2']
        gammaVals3 = data['possibleVals3']
        lossArr = data['allData']

    # Scale of points
    minLoss = np.min(lossArr)
    maxLoss = np.max(lossArr)

    xArr = np.zeros_like(lossArr)
    yArr = np.zeros_like(lossArr)
    zArr = np.zeros_like(lossArr)
    sArr = np.zeros_like(lossArr)
    for i in range(len(gammaVals1)):
        for j in range(len(gammaVals2)):
            for k in range(len(gammaVals3)):
                xArr[i,j,k] = gammaVals1[i]
                yArr[i,j,k] = gammaVals2[j]
                zArr[i,j,k] = gammaVals3[k]
                sArr[i,j,k] = 25*((1-(lossArr[i,j,k]-minLoss)/(maxLoss - minLoss))**5)

    return lossArr, xArr, yArr, zArr, sArr

###################################################################################################################################

################################
### FIGURE 1: paramTransform ###
################################
def plotParamTransform(pageFracWidth=0.85, aspectRatio=2.0, fileType='.png', name='paramTransform'):
    plt.close('all')
    fig, axs = plt.subplots(1, 2, figsize=(toFigSize(pageFracWidth, aspectRatio)), sharey=True)
    axs[0].set_ylabel('Parameter Value')
    
    gammas = [np.array([0.4, 0.075, 0.3]), np.array([0.2, 0.3])]
    for i, gamma in enumerate(gammas):
        ax = axs[i]
        alpha, beta = paramTransform(gamma)
        lenGamma = len(gamma)
        hatches = lenGamma*[r''] + 2*[r'\\'] + [r'xx'] + lenGamma*[r'//'] + [''] #['', '', '', '..', '..', '//..', '//', '//', '//', '//']
        color = ['#ccccff', '#ffcccc']*(lenGamma+1)
        numParams = 2*(lenGamma+2)
        knitParams = np.zeros(numParams)
        knitParams[::2] = alpha
        knitParams[1::2] = beta
        mids = np.linspace(0.0, 0.5, numParams + 1)
        mids = mids[:-1] + mids[1:]
        ax.bar(mids, knitParams, 1 / numParams, color = color, linewidth = 0.0, hatch = hatches)
        ax.set_xticks(mids)
        # TODO: This is hacky.
        ax.set_xticklabels(np.array([r'$\alpha_1$', r'$\beta_1$', r'$\alpha_2$', r'$\beta_2$', r'$\alpha_3$', r'$\beta_3$', r'$\alpha_4$', r'$\beta_4$', r'$\alpha_5$', r'$\beta_5$'])[:numParams])
        ax.yaxis.grid(which='major', color='#CCCCCC', linewidth=1.0)
    
    handle1 = mpatches.Patch(facecolor='white', hatch=r'\\', label='Consistency')
    handle2 = mpatches.Patch(facecolor='white', hatch=r'//', label='Symmetry')
    axs[1].legend(handles = [handle1]+[handle2], loc='best')
    
    plt.tight_layout()
    # plt.show()
    fig.savefig(name+fileType, bbox_inches='tight', transparent=True, dpi=getDPI(fileType))


###############################
### FIGURE 2: lossLandscape ###
###############################
def plotLossLandscape(pageFracWidth=0.95, aspectRatio=1.2, fileType='.png', name='lossLandscape'):
    lossArr, xArr, yArr, zArr, sArr = loadLossLandOrig()
    xsPlus, xsMinus, ysPlus, ysMinus, zs = getLen5Order4Man()
    box = np.array([[-0.5, 0.4], [-0.5, 0.4], [-0.5, 0.4]])
    inBoxPlus = (box[0,0] < xsPlus) & (xsPlus < box[0,1]) & (box[1,0] < ysPlus) & (ysPlus < box[1,1]) & (box[2,0] < zs) & (zs < box[2,1]) 
    inBoxMinus = (box[0,0] < xsMinus) & (xsMinus <box[0,1]) & (box[1,0] < ysMinus) & (ysMinus < box[1,1]) & (box[2,0] < zs) & (zs < box[2,1]) 

    # Set up fig
    plt.close('all')
    fig = plt.figure(figsize=(toFigSize(pageFracWidth, aspectRatio)))
    ax = plt.axes(projection='3d')

    img = ax.scatter3D(xArr, yArr, zArr, c=lossArr, cmap=plt.gray(), alpha=0.3, s = sArr, vmin= 0.0, vmax=2.0, linewidths=0)
    ax.scatter3D(xsPlus[inBoxPlus], ysPlus[inBoxPlus], zs[inBoxPlus], c='blue', label=r'$4^\mathrm{th}$ Order manifold')
    ax.scatter3D(xsMinus[inBoxMinus], ysMinus[inBoxMinus], zs[inBoxMinus], c='pink', label=r'_$4^\mathrm{th}$ Order manifold')
    ax.scatter3D(0.125, 0.25, 0.25, c='green', label=r'$4 \times$ Strang') # PolyStrang
    ax.scatter3D(0.36266572103945605, -0.10032032403589856, -0.1352975465549758, c='red', label='Learn5A') # Learn5A

    ax.set_xlabel(r'$\gamma_1$')
    ax.set_ylabel(r'$\gamma_2$')
    ax.set_zlabel(r'$\gamma_3$')
    cbar = fig.colorbar(img, location='right', shrink=0.5, pad=0.04)
    cbar.solids.set(alpha=1.0)
    ax.legend(loc='upper right')

    plt.tight_layout()
    # plt.show()
    fig.savefig(name+fileType, bbox_inches='tight', transparent=True, dpi=getDPI(fileType))

##############################
### FIGURE 3: loss2dPlanes ###
##############################
def plotLoss2dPlanes(pageFracWidth=0.8, aspectRatio=0.5, fileType='.png', name='loss2dPlanes'):
    print('TODO!')

###############################
### FIGURE 4: splitVisLearn ###
###############################
def plotSplitVisLearn(pageFracWidth=0.8, aspectRatio=2.0, fileType='.png', name='splitVisLearn'):
    print('TODO!')

##########################
### FIGURE 5: lossConv ###
##########################
def plotLossConv(pageFracWidth=0.85, aspectRatio=2.0, fileType='.png', name='lossConv'):
    print('TODO!')

############################
### FIGURE 6: lossRelAdv ###
############################
def plotLossRelAdv(pageFracWidth=0.85, aspectRatio=1.7, fileType='.png', name='lossRelAdv'):
    _, losses, numExps, _, _, _ = loadLossConvOrig()
    lossTrot, lossStrang, lossYosh, lossLearn5A, lossLearn8A, lossLearn8B = losses
    numExpsTrot, numExpsStrang, numExpsYosh, numExpsLearn5A, numExpsLearn8A, numExpsLearn8B = numExps

    classicalNumExps = np.concatenate((   numExpsTrot,  numExpsStrang,    numExpsYosh), axis=0)
    learnedNumExps   = np.concatenate((numExpsLearn5A, numExpsLearn8A, numExpsLearn8B), axis=0)

    numExpsCommon = np.geomspace(max(min(classicalNumExps), min(learnedNumExps)),
                                 min(max(classicalNumExps), max(learnedNumExps)), 100000)
    
    lossTrotCommon   = np.exp(np.interp(np.log(numExpsCommon), np.log(numExpsTrot)  , np.log(lossTrot)  ))
    lossStrangCommon = np.exp(np.interp(np.log(numExpsCommon), np.log(numExpsStrang), np.log(lossStrang)))
    lossYoshCommon   = np.exp(np.interp(np.log(numExpsCommon), np.log(numExpsYosh)  , np.log(lossYosh)  ))

    lossLearn5ACommon = np.exp(np.interp(np.log(numExpsCommon), np.log(numExpsLearn5A), np.log(lossLearn5A)))
    lossLearn8ACommon = np.exp(np.interp(np.log(numExpsCommon), np.log(numExpsLearn8A), np.log(lossLearn8A)))
    lossLearn8BCommon = np.exp(np.interp(np.log(numExpsCommon), np.log(numExpsLearn8B), np.log(lossLearn8B)))

    ratioAll = np.fmin(np.fmin(lossTrotCommon, lossStrangCommon), lossYoshCommon) / np.fmin(np.fmin(lossLearn5ACommon, lossLearn8ACommon), lossLearn8BCommon)
    ratioYosh5A = lossYoshCommon / lossLearn5ACommon
    ratioYosh8A = lossYoshCommon / lossLearn8ACommon
    ratioYosh8B = lossYoshCommon / lossLearn8BCommon

    allMin = np.argmin(ratioAll)
    yosh5AMin = np.argmin(ratioYosh5A)
    yosh8AMin = np.argmin(ratioYosh8A)
    yosh8BMin = np.argmin(ratioYosh8B)

    # maxInd = np.argmax(ratioAll)
    # print(f'The maximum relative advantage {ratioAll[maxInd]:.3f} is found at numExps = {numExpsCommon[maxInd]:.0f}')
    # print(f'Loss of Trot at maximum relative advantage is {lossTrotCommon[maxInd]:.8f}')
    # print(f'Loss of Strang at maximum relative advantage is {lossStrangCommon[maxInd]:.8f}')
    # print(f'Loss of Yosh at maximum relative advantage is {lossYoshCommon[maxInd]:.8f}')
    # print(f'Loss of Learn5A at maximum relative advantage is {lossLearn5ACommon[maxInd]:.8f}')
    # print(f'Loss of Learn8A at maximum relative advantage is {lossLearn8ACommon[maxInd]:.8f}')
    # print(f'Loss of Learn8B at maximum relative advantage is {lossLearn8BCommon[maxInd]:.8f}')

    # print(f'Relative accuracy of Trot vs Yosh at maximum relative advantage is {lossYoshCommon[maxInd]/lossTrotCommon[maxInd]:.3f}')
    # print(f'Relative accuracy of Strang vs Yosh at maximum relative advantage is {lossYoshCommon[maxInd]/lossStrangCommon[maxInd]:.3f}')
    # print(f'Relative accuracy of Yosh vs Yosh at maximum relative advantage is {lossYoshCommon[maxInd]/lossYoshCommon[maxInd]:.3f}')
    # print(f'Relative accuracy of Learn5A vs Yosh at maximum relative advantage is {lossYoshCommon[maxInd]/lossLearn5ACommon[maxInd]:.3f}')
    # print(f'Relative accuracy of Learn8A vs Yosh at maximum relative advantage is {lossYoshCommon[maxInd]/lossLearn8ACommon[maxInd]:.3f}')
    # print(f'Relative accuracy of Learn8B vs Yosh at maximum relative advantage is {lossYoshCommon[maxInd]/lossLearn8BCommon[maxInd]:.3f}')

    # print(f'Relative speed of Trot vs Yosh at maximum relative advantage is {np.sqrt(lossYoshCommon[maxInd]/lossTrotCommon[maxInd]):.3f}')
    # print(f'Relative speed of Strang vs Yosh at maximum relative advantage is {np.sqrt(lossYoshCommon[maxInd]/lossStrangCommon[maxInd]):.3f}')
    # print(f'Relative speed of Yosh vs Yosh at maximum relative advantage is {np.sqrt(lossYoshCommon[maxInd]/lossYoshCommon[maxInd]):.3f}')
    # print(f'Relative speed of Learn5A vs Yosh at maximum relative advantage is {np.sqrt(lossYoshCommon[maxInd]/lossLearn5ACommon[maxInd]):.3f}')
    # print(f'Relative speed of Learn8A vs Yosh at maximum relative advantage is {np.sqrt(lossYoshCommon[maxInd]/lossLearn8ACommon[maxInd]):.3f}')
    # print(f'Relative speed of Learn8B vs Yosh at maximum relative advantage is {np.sqrt(lossYoshCommon[maxInd]/lossLearn8BCommon[maxInd]):.3f}')
    
    # Set up fig
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(toFigSize(pageFracWidth, aspectRatio)))
    ax.set_xlabel(r'number of exponentials')
    ax.set_ylabel(r'relative advantage of learned')

    minMin = np.min([allMin, yosh5AMin, yosh8AMin, yosh8BMin])
    ax.loglog(numExpsCommon[:minMin+1], (minMin+1)*[1.0], 'black', linestyle='-', alpha=1.0, label=r'Equal performance')
    ax.text(1.8*10**5, 5.0/4.0, 'learned better', ha="center", va="bottom")
    ax.text(1.8*10**5, 4.0/5.0, 'classical better', ha="center", va="top")

    ax.loglog(numExpsCommon[:allMin+1], ratioAll[:allMin+1], 'grey', linestyle='-', marker='', alpha=1.0, label='Best Classical vs Best Learned')
    ax.loglog(numExpsCommon[:yosh5AMin+1], ratioYosh5A[:yosh5AMin+1], 'forestgreen' , linestyle='-', marker='', alpha=1.0, label='Yoshida vs Learn5A')
    ax.loglog(numExpsCommon[:yosh8AMin+1], ratioYosh8A[:yosh8AMin+1], 'royalblue'   , linestyle='-', marker='', alpha=1.0, label='Yoshida vs Learn8A')
    ax.loglog(numExpsCommon[:yosh8BMin+1], ratioYosh8B[:yosh8BMin+1], 'lightskyblue', linestyle='-', marker='', alpha=1.0, label='Yoshida vs Learn8B')

    ax.legend(loc='best')
    ax.grid(which='major', color='#CCCCCC', linewidth=1.0)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
    ax.set_xscale('log')

    plt.tight_layout()
    # plt.show()
    fig.savefig(name+fileType, bbox_inches='tight', transparent=True, dpi=getDPI(fileType))

#############################
### FIGURE 7: lossConvGen ###
#############################
def plotLossConvGen(pageFracWidth=0.85, aspectRatio=2.0, fileType='.png', name='lossConvGen'):
    print('TODO!')

#############################
### FIGURE 8: lossLandGen ###
#############################
def plotLossLandGen(pageFracWidth=0.66, aspectRatio=1.0, fileType='.png', name='lossLandGen'):
    print('TODO!')

##############################
### FIGURE 9: bestFitCoefs ###
##############################
def plotBestFitCoefs(pageFracWidth=0.85, aspectRatio=2.0, fileType='.png', name='bestFitCoefs'):
    print('TODO!')

##################################
### FIGURE 10: loss2dPlanesGen ###
##################################
def plotLoss2dPlanesGen(pageFracWidth=0.8, aspectRatio=0.5, fileType='.png', name='loss2dPlanesGen'):
    print('TODO!')

###############################
### FIGURE 11: lossConvProj ###
###############################
def plotLossConvProj(pageFracWidth=0.85, aspectRatio=2.0, fileType='.png', name='lossConvProj'):
    print('TODO!')

##################################
### FIGURE 12: sampleInitConds ###
##################################
def plotSampleInitConds(pageFracWidth=0.85, aspectRatio=2.0, fileType='.png', name='sampleInitConds'):
    print('TODO!')

#############################
### FIGURE 13: paramOptim ###
#############################
def plotParamOptim(pageFracWidth=0.85, aspectRatio=2.0, fileType='.png', name='paramOptim'):
    print('TODO!')

############################
### FIGURE 14: allOptims ###
############################
def plotAllOptims(pageFracWidth=0.85, aspectRatio=2.0, fileType='.png', name='allOptims'):
    print('TODO!')

###################################################################################################################################

##################
### CALL PLOTS ###
##################
plotParamTransform( 0.85, 2.0, '.png', 'paramTransform' )
plotLossLandscape(  0.95, 1.2, '.png', 'lossLandscape'  )
plotLoss2dPlanes(   0.80, 0.5, '.png', 'loss2dPlanes'   )
plotSplitVisLearn(  0.80, 2.0, '.png', 'splitVisLearn'  )
plotLossConv(       0.85, 2.0, '.png', 'lossConv'       )
plotLossRelAdv(     0.85, 1.7, '.png', 'lossRelAdv'     )
plotLossConvGen(    0.85, 2.0, '.png', 'lossConvGen'    )
plotLossLandGen(    0.66, 1.0, '.png', 'lossLandGen'    )
plotBestFitCoefs(   0.85, 2.0, '.png', 'bestFitCoefs'   )
plotLoss2dPlanesGen(0.80, 0.5, '.png', 'loss2dPlanesGen')
plotLossConvProj(   0.85, 2.0, '.png', 'lossConvProj'   )
plotSampleInitConds(0.85, 2.0, '.png', 'sampleInitConds')
plotParamOptim(     0.85, 2.0, '.png', 'paramOptim'     )
plotAllOptims(      0.85, 2.0, '.png', 'allOptims'      )