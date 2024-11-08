###############
### IMPORTS ###
###############

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX for text rendering
    "font.size": 10,      # Match LaTeX font size
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"
})
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

###################################################################################################################################

###############
### HELPERS ###
###############
def toFigSize(pageFracWidth, aspectRatio):
    # A4 around 6.5 inches for \textwidth, check as some say 5.125 but looks worse.
    textwidth = 7.5
    figWidthInches = textwidth * pageFracWidth
    figHeightInches = figWidthInches / aspectRatio
    # Return in form x, y
    return figWidthInches, figHeightInches

def getDPI(fileType='.png'):
    return 'figure' if fileType == '.pdf' else 450

def getSplitNames(incStrang4X = False):
    if incStrang4X:
        return ['Trotter', 'Strang', 'Yoshida', r'$4 \times$ Strang', 'Learn5A', 'Learn8A', 'Learn8B']
    return ['Trotter', 'Strang', 'Yoshida', 'Learn5A', 'Learn8A', 'Learn8B']

def getSplitColours(incStrang4X = False):
    if incStrang4X:
        return ['mediumvioletred', 'firebrick', 'peru', 'sandybrown', 'forestgreen', 'royalblue', 'lightskyblue']
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

def coordTransform(cent, p1, p2, x, y, z):
    # vShift = v - alpha
    xShift = x - cent[0]
    yShift = y - cent[1]
    zShift = z - cent[2]

    # lambdai = np.dot(pi, vShift)
    coordCand1 = p1[0] * xShift + p1[1] * yShift + p1[2] * zShift
    coordCand2 = p2[0] * xShift + p2[1] * yShift + p2[2] * zShift

    # lambda = np.linalg.inv(np.array([[1, np.dot(p1, p2)],[np.dot(p1, p2), 1]])) * lambda
    scaleA = np.dot(p1, p2)
    coord1 = coordCand1 - scaleA*coordCand2
    coord2 = coordCand2 - scaleA*coordCand1

    scaleB = 1.0/(1.0-scaleA*scaleA)
    coord1 = scaleB * coord1
    coord2 = scaleB * coord2

    return coord1, coord2

###################################################################################################################################

#################
### LOAD DATA ###
#################
def loadLossConvOrig():
    with np.load('data/lossConvOrig.npz') as data:
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

def loadLossConvOptions(path, timeRange=10):
    with np.load(path) as data:
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
    
    return [lossTrot, lossStrang, lossYosh, lossLearn5A, lossLearn8A, lossLearn8B], \
           [numExpsTrot, numExpsStrang, numExpsYosh, numExpsLearn5A, numExpsLearn8A, numExpsLearn8B]

def loadLossLand(path):
    with np.load(path) as data:
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

def loadLand2dPlanes(path, name='losses'):
    allPSS = []
    with np.load(path + '/polyStrangSlwDyn.npz') as data:
        xs = data['xs']
        ys = data['ys']
        zs = data['zs']
        loss = data[name]
        cent = data['cent']
        perp1 = data['perp1']
        perp2 = data['perp2']
        allPSS = [xs, ys, zs, loss, cent, perp1, perp2]
        
    allPSM = []
    with np.load(path + '/polyStrangMedDyn.npz') as data:
        xs = data['xs']
        ys = data['ys']
        zs = data['zs']
        loss = data[name]
        cent = data['cent']
        perp1 = data['perp1']
        perp2 = data['perp2']
        allPSM = [xs, ys, zs, loss, cent, perp1, perp2]

    allPSF = []
    with np.load(path + '/polyStrangFstDyn.npz') as data:
        xs = data['xs']
        ys = data['ys']
        zs = data['zs']
        loss = data[name]
        cent = data['cent']
        perp1 = data['perp1']
        perp2 = data['perp2']
        allPSF = [xs, ys, zs, loss, cent, perp1, perp2]

    allLS = []
    with np.load(path + '/learnedSlwDyn.npz') as data:
        xs = data['xs']
        ys = data['ys']
        zs = data['zs']
        loss = data[name]
        cent = data['cent']
        perp1 = data['perp1']
        perp2 = data['perp2']
        allLS = [xs, ys, zs, loss, cent, perp1, perp2]

    allLM = []
    with np.load(path + '/learnedMedDyn.npz') as data:
        xs = data['xs']
        ys = data['ys']
        zs = data['zs']
        loss = data[name]
        cent = data['cent']
        perp1 = data['perp1']
        perp2 = data['perp2']
        allLM = [xs, ys, zs, loss, cent, perp1, perp2]

    allLF = []
    with np.load(path + '/learnedFstDyn.npz') as data:
        xs = data['xs']
        ys = data['ys']
        zs = data['zs']
        loss = data[name]
        cent = data['cent']
        perp1 = data['perp1']
        perp2 = data['perp2']
        allLF = [xs, ys, zs, loss, cent, perp1, perp2]
    
    return allPSS, allPSM, allPSF, allLS, allLM, allLF

def loadIvpRefs(path):
    with np.load(path) as data:
        pot = data['pot']
        xGrid = data['xGrid']
        uInitsVal = data['uInitsVal']
        uFinalRefVal = data['uFinalRefVal']

    return pot, xGrid, uInitsVal, uFinalRefVal

def loadParamLossTrain():
    with np.load('data/paramLossTrain.npz') as data:
        learn5AVal = data['learn5aVal']
        learn8AVal = data['learn8aVal']
        learn8BVal = data['learn8bVal']

        learn5ALoss = data['learn5aLoss']
        learn8ALoss = data['learn8aLoss']
        learn8BLoss = data['learn8bLoss']

        learn5AP1 = data['learn5aP1']
        learn8AP1 = data['learn8aP1']
        learn8BP1 = data['learn8bP1']

        learn5AP2 = data['learn5aP2']
        learn8AP2 = data['learn8aP2']
        learn8BP2 = data['learn8bP2']

        learn5AP3 = data['learn5aP3']
        learn8AP3 = data['learn8aP3']
        learn8BP3 = data['learn8bP3']

        learn8AP4 = data['learn8aP4']
        learn8BP4 = data['learn8bP4']

        learn8AP5 = data['learn8aP5']
        learn8BP5 = data['learn8bP5']

        learn8AP6 = data['learn8aP6']
        learn8BP6 = data['learn8bP6']

    return [[learn5AVal, learn5ALoss, learn5AP1, learn5AP2, learn5AP3, [], [], [], True], 
            [learn8AVal, learn8ALoss, learn8AP1, learn8AP2, learn8AP3, learn8AP4, learn8AP5, learn8AP6, False],
            [learn8BVal, learn8BLoss, learn8BP1, learn8BP2, learn8BP3, learn8BP4, learn8BP5, learn8BP6, False]]
    
def loadParamLossOptims():
    with np.load('data/paramLossOptims.npz') as data:
        adaGL0p01M0p05Loss = data['adaGL0p01M0p05Loss']
        adaGL0p01M0p10Loss = data['adaGL0p01M0p10Loss']
        adamL0p01M0p05Loss = data['adamL0p01M0p05Loss']
        adamL0p01M0p10Loss = data['adamL0p01M0p10Loss']
        lionL0p01M0p05Loss = data['lionL0p01M0p05Loss']
        lionL0p01M0p10Loss = data['lionL0p01M0p10Loss']
        lemaL0p20M0p05Loss = data['lemaL0p20M0p05Loss']
        lemaL0p20M0p10Loss = data['lemaL0p20M0p10Loss']
    
        adaGL0p01M0p05X = data['adaGL0p01M0p05X']
        adaGL0p01M0p10X = data['adaGL0p01M0p10X']
        adamL0p01M0p05X = data['adamL0p01M0p05X']
        adamL0p01M0p10X = data['adamL0p01M0p10X']
        lionL0p01M0p05X = data['lionL0p01M0p05X']
        lionL0p01M0p10X = data['lionL0p01M0p10X']
        lemaL0p20M0p05X = data['lemaL0p20M0p05X']
        lemaL0p20M0p10X = data['lemaL0p20M0p10X']
    
        adaGL0p01M0p05Y = data['adaGL0p01M0p05Y']
        adaGL0p01M0p10Y = data['adaGL0p01M0p10Y']
        adamL0p01M0p05Y = data['adamL0p01M0p05Y']
        adamL0p01M0p10Y = data['adamL0p01M0p10Y']
        lionL0p01M0p05Y = data['lionL0p01M0p05Y']
        lionL0p01M0p10Y = data['lionL0p01M0p10Y']
        lemaL0p20M0p05Y = data['lemaL0p20M0p05Y']
        lemaL0p20M0p10Y = data['lemaL0p20M0p10Y']
    
        adaGL0p01M0p05Z = data['adaGL0p01M0p05Z']
        adaGL0p01M0p10Z = data['adaGL0p01M0p10Z']
        adamL0p01M0p05Z = data['adamL0p01M0p05Z']
        adamL0p01M0p10Z = data['adamL0p01M0p10Z']
        lionL0p01M0p05Z = data['lionL0p01M0p05Z']
        lionL0p01M0p10Z = data['lionL0p01M0p10Z']
        lemaL0p20M0p05Z = data['lemaL0p20M0p05Z']
        lemaL0p20M0p10Z = data['lemaL0p20M0p10Z']

    return [[adaGL0p01M0p05Loss, adaGL0p01M0p05X, adaGL0p01M0p05Y, adaGL0p01M0p05Z, 'mediumvioletred', 'AdaGrad 0.05' ], 
            [adaGL0p01M0p10Loss, adaGL0p01M0p10X, adaGL0p01M0p10Y, adaGL0p01M0p10Z, 'firebrick'      , 'AdaGrad 0.1'  ], 
            [adamL0p01M0p05Loss, adamL0p01M0p05X, adamL0p01M0p05Y, adamL0p01M0p05Z, 'peru'           , 'Adam 0.05'    ], 
            [adamL0p01M0p10Loss, adamL0p01M0p10X, adamL0p01M0p10Y, adamL0p01M0p10Z, 'darkorange'     , 'Adam 0.1'     ], 
            [lionL0p01M0p05Loss, lionL0p01M0p05X, lionL0p01M0p05Y, lionL0p01M0p05Z, 'yellowgreen'    , 'Lion 0.05'    ], 
            [lionL0p01M0p10Loss, lionL0p01M0p10X, lionL0p01M0p10Y, lionL0p01M0p10Z, 'forestgreen'    , 'Lion 0.1'     ], 
            [lemaL0p20M0p05Loss, lemaL0p20M0p05X, lemaL0p20M0p05Y, lemaL0p20M0p05Z, 'royalblue'      , 'Lev-Marq 0.05'], 
            [lemaL0p20M0p10Loss, lemaL0p20M0p10X, lemaL0p20M0p10Y, lemaL0p20M0p10Z, 'lightskyblue'   , 'Lev-Marq 0.1' ]]

###################################################################################################################################

################################
### FIGURE 1: paramTransform ###
################################
def plotParamTransform(pageFracWidth=0.85, aspectRatio=2.0, fileType='.png', name='paramTransform'):
    plt.close('all')
    fig, axs = plt.subplots(1, 2, figsize=(toFigSize(pageFracWidth, aspectRatio)), sharey=True, layout='constrained')
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
    axs[1].legend(handles = [handle1]+[handle2], loc='lower right', framealpha=1.0)
    
    fig.savefig(name+fileType, bbox_inches='tight', transparent=True, dpi=getDPI(fileType))


###############################
### FIGURE 2: lossLandscape ###
###############################
def plotLossLandscape(pageFracWidth=0.95, aspectRatio=1.2, fileType='.png', name='lossLandscape'):
    lossArr, xArr, yArr, zArr, sArr = loadLossLand('data/lossLandOrig.npz')
    xsPlus, xsMinus, ysPlus, ysMinus, zs = getLen5Order4Man()
    box = np.array([[-0.5, 0.4], [-0.5, 0.4], [-0.5, 0.4]])
    inBoxPlus = (box[0,0] < xsPlus) & (xsPlus < box[0,1]) & (box[1,0] < ysPlus) & (ysPlus < box[1,1]) & (box[2,0] < zs) & (zs < box[2,1]) 
    inBoxMinus = (box[0,0] < xsMinus) & (xsMinus <box[0,1]) & (box[1,0] < ysMinus) & (ysMinus < box[1,1]) & (box[2,0] < zs) & (zs < box[2,1]) 

    # Set up fig
    plt.close('all')
    fig = plt.figure(figsize=(toFigSize(pageFracWidth, aspectRatio)), layout='constrained')
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

    fig.savefig(name+fileType, bbox_inches='tight', transparent=True, dpi=getDPI(fileType))

##############################
### FIGURE 3: loss2dPlanes ###
##############################
def plotLoss2dPlanes(pageFracWidth=0.8, aspectRatio=1.0, fileType='.png', name='loss2dPlanes'):
    allPSS, allPSM, allPSF, allLS, allLM, allLF = loadLand2dPlanes('data/loss2dPlanesData')

    def addSlice(x, y, loss, ax):
        cPlot = ax.contourf(x, y, loss, cmap = cmapComp, norm = normalizerComp, alpha=0.2, antialiased=True)
        cPlot = ax.contour(x, y, loss, alpha=1.0, colors='grey')
        ax.clabel(cPlot, inline=True)

    def pipeLine(allData, ax):
        xs, ys, zs, loss, cent, perp1, perp2 = allData
        lambda1, lambda2 = coordTransform(cent, perp1, perp2, xs, ys, zs)
        addSlice(lambda1, lambda2, loss, ax)

    plt.close('all')
    fig, axs = plt.subplots(2, 3, figsize=(toFigSize(pageFracWidth, aspectRatio)), layout='constrained', sharex=True, sharey=True)

    minVal = np.min(np.array([
             np.min(allPSS[3]),
             np.min(allPSM[3]),
             np.min(allPSF[3]),
             np.min(allLS[3]),
             np.min(allLM[3]),
             np.min(allLF[3])
    ]))

    maxVal = np.max(np.array([
             np.max(allPSS[3]),
             np.max(allPSM[3]),
             np.max(allPSF[3]),
             np.max(allLS[3]),
             np.max(allLM[3]),
             np.max(allLF[3])
    ]))

    cmapComp='bwr'
    normalizerComp=matplotlib.colors.Normalize(minVal, maxVal)
    imComp=matplotlib.cm.ScalarMappable(norm=normalizerComp, cmap = cmapComp)

    pipeLine(allPSS, axs[0,0])
    pipeLine(allPSM, axs[0,1])
    pipeLine(allPSF, axs[0,2])

    pipeLine(allLS, axs[1,0])
    pipeLine(allLM, axs[1,1])
    pipeLine(allLF, axs[1,2])

    speeds = [r'Slow Dyn $\perp e_1$', r'Med Dyn $\perp e_2$', r'Fast Dyn $\perp e_3$']
    xLabels = [r'$e_2$', r'$e_1$', r'$e_1$']
    yLabels = [r'$e_3$', r'$e_3$', r'$e_2$']
    for i in range(3):
        axs[0,i].set_title(speeds[i])
        axs[0,i].set_xlabel(xLabels[i])
        axs[0,i].set_ylabel(yLabels[i])
        axs[1,i].set_xlabel(xLabels[i])
        axs[1,i].set_ylabel(yLabels[i])

    minima = [r'$\gamma = [0.125, 0.25, 0.25]$', r'$\gamma = [0.3314, -0.07304, -0.1821]$']
    for i in range(2):
        tempAx = axs[i,2].twinx()
        tempAx.set_ylabel(minima[i])
        tempAx.set_yticks([])


    cbar = fig.colorbar(imComp, ax=axs)
    cbar.solids.set(alpha=0.2)
    
    fig.savefig(name+fileType, bbox_inches='tight', transparent=True, dpi=getDPI(fileType))

###############################
### FIGURE 4: splitVisLearn ###
###############################
def plotSplitVisLearn(pageFracWidth=0.8, aspectRatio=2.0, fileType='.png', name='splitVisLearn'):
    
    def pipeLine(ax, alpha, beta, colour = 'grey', label = '', traceWidth = -1.0):
        assert(len(alpha) == len(beta))
        xs = np.array([0.0])
        ys = np.array([0.0])

        for i in range(len(alpha)):
            newX = xs[-1] + alpha[i]
            oldY = ys[-1]
            newY = oldY + beta[i]
            xs = np.append(xs, np.array([newX, newX]))
            ys = np.append(ys, np.array([oldY, newY]))

        ax.plot(xs, ys, color=colour, label=label)

        if traceWidth > 0.0:
          ax.plot(xs, ys, color=colour, alpha=0.2, lw=traceWidth)

        return xs, ys

    # Set up fig
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(toFigSize(pageFracWidth, aspectRatio)), sharey=True, layout='constrained')
    ax.set_xlabel(r'$u_1(t)$')
    ax.set_ylabel(r'$u_2(t)$')

    gammas = [[], # Strang
              [0.6756035959798289, 1.3512071919596578], # Yoshida
              [0.125, 0.25, 0.25], # 4X Strang
              [0.36266572103945605, -0.10032032403589856, -0.1352975465549758], # Learn5A
              [0.2134815929093979, -0.05820764895590353, 0.41253264535444745, -0.13523357389399546, 0.4443203153968813, -0.02509257759188356], # Learn8A
              [0.11775349336573762, 0.38763572759753917, 0.36597039590662095, 0.2921906385941626, 0.05641644544703187, -0.021241661286415584]] # Learn8B

    alphas = [np.array([1.0])]  # Force add Trotter as not sym
    betas = [np.array([1.0])]    # Force add Trotter as not sym

    for gamma in gammas:
        alpha, beta = paramTransform(np.array(gamma))
        alphas.append(alpha)
        betas.append(beta)

    splitColours = getSplitColours(True)
    for i, splitName in enumerate(getSplitNames(True)):
        pipeLine(ax, alphas[i], betas[i], splitColours[i], splitName, 10.0)
    
    ax.plot(np.array([0.0, 1.0]), np.array([0.0, 1.0]), color='grey', label='Exact Sol', zorder=0)

    ax.legend(loc='center', ncols=1, bbox_to_anchor=(-0.23, 0.5))
    ax.grid(which='major', color='#CCCCCC', linewidth=1.0)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)

    fig.savefig(name+fileType, bbox_inches='tight', transparent=True, dpi=getDPI(fileType))

##########################
### FIGURE 5: lossConv ###
##########################
def plotLossConv(pageFracWidth=0.85, aspectRatio=2.0, fileType='.png', name='lossConv'):
    splitNames, losses, numExps, lowQs, highQs, _ = loadLossConvOrig()
    lossTrot, lossStrang, lossYosh, lossLearn5A, lossLearn8A, lossLearn8B = losses

    trotMin = np.argmin(lossTrot)
    strangMin = np.argmin(lossStrang)
    yoshMin = np.argmin(lossYosh)
    learn5AMin = np.argmin(lossLearn5A)
    learn8AMin = np.argmin(lossLearn8A)
    learn8BMin = np.argmin(lossLearn8B)
    minInds = [trotMin, strangMin, yoshMin, learn5AMin, learn8AMin, learn8BMin]

    # Set up fig
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(toFigSize(pageFracWidth, aspectRatio)), layout='constrained')
    ax.set_xlabel(r'number of exponentials')
    ax.set_ylabel(r'func $L_2$ norm')

    for i, splitCol in enumerate(getSplitColours()):
        plotTo = minInds[i]+1
        filtExps = numExps[i][:plotTo]
        ax.loglog(filtExps, losses[i][:plotTo], splitCol, linestyle='-', marker='', alpha=1.0, label=splitNames[i])
        ax.fill_between(filtExps, lowQs[i][:plotTo], highQs[i][:plotTo], color=splitCol, alpha=0.2, linewidth=0)
    
    # Hach in Learn5AProj
    with np.load('data/lossConvProj.npz') as data:
        unitNumStepsLearn5AProj = data['unitNumStepsGammaLearn3AProj']
        lossesLearn5AProj = data['lossLearn3AProj']
    numExpsLearn5AProj = unitNumStepsLearn5AProj*10*(10 - 2) + 1
    lossLearn5AProj = np.quantile(np.abs(lossesLearn5AProj), 0.5, axis=0)
    learn5AProjMin = np.argmin(lossLearn5AProj)
    plotTo = learn5AProjMin+1
    stdDev1 = 0.341
    lowQLearn5AProj = np.quantile(np.abs(lossesLearn5AProj), 0.5 - stdDev1, axis=0)
    highQLearn5AProj = np.quantile(np.abs(lossesLearn5AProj), 0.5 + stdDev1, axis=0)
    ax.loglog(numExpsLearn5AProj[:plotTo], lossLearn5AProj[:plotTo], 'lime', linestyle='-', marker='', alpha=1.0, label='Learn5AProj', zorder=0)
    ax.fill_between(numExpsLearn5AProj[:plotTo], lowQLearn5AProj[:plotTo], highQLearn5AProj[:plotTo], color='lime', alpha=0.2, linewidth=0, zorder=0)

    ax.loglog(numExps[0][:10], 10*[2], 'black', linestyle='-', alpha=0.5, label=r'Unitarity')
    orderLineWidth = np.array([200, 1000])
    ax.loglog(orderLineWidth, (1.00e-3/(orderLineWidth[0]**-1.0)) * np.power(orderLineWidth, -1.0), 'black', linestyle=':', alpha=0.5, label=r'Order 1, 2, 4')
    ax.loglog(orderLineWidth, (3.33e-4/(orderLineWidth[0]**-2.0)) * np.power(orderLineWidth, -2.0), 'black', linestyle=':', alpha=0.5)
    ax.loglog(orderLineWidth, (1.00e-4/(orderLineWidth[0]**-4.0)) * np.power(orderLineWidth, -4.0), 'black', linestyle=':', alpha=0.5)

    ax.legend(loc='best', ncols=3)
    ax.grid(which='major', color='#CCCCCC', linewidth=1.0)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
    ax.set_xscale('log')

    fig.savefig(name+fileType, bbox_inches='tight', transparent=True, dpi=getDPI(fileType))

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

    # Set up fig
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(toFigSize(pageFracWidth, aspectRatio)), layout='constrained')
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

    fig.savefig(name+fileType, bbox_inches='tight', transparent=True, dpi=getDPI(fileType))

#############################
### FIGURE 7: lossConvGen ###
#############################
def plotLossConvGen(pageFracWidth=0.85, aspectRatio=2.0, fileType='.png', name='lossConvGen'):
    splitNames, origlosses, origNumExps, _, _, _ = loadLossConvOrig()
    newPotlosses, newPotNumExps = loadLossConvOptions('data/lossConvNewPot.npz')
    newAlllosses, newAllNumExps = loadLossConvOptions('data/lossConvAllNew.npz', 30)
    losses = [origlosses, newPotlosses, newAlllosses]
    numExps = [origNumExps, newPotNumExps, newAllNumExps]

    # Set up fig
    plt.close('all')
    fig, axs = plt.subplots(1, 2, figsize=(toFigSize(pageFracWidth, aspectRatio)), sharey=True, layout='constrained')
    axs[0].set_xlabel(r'number of exponentials')
    axs[1].set_xlabel(r'number of exponentials')
    axs[0].set_ylabel(r'func $L_2$ norm')
    axsList = [0, 0, 1]
    linSty = ['-', '--', '-']

    for i, loss in enumerate(losses):
        lossTrot, lossStrang, lossYosh, lossLearn5A, lossLearn8A, lossLearn8B = loss

        trotMin = np.argmin(lossTrot)
        strangMin = np.argmin(lossStrang)
        yoshMin = np.argmin(lossYosh)
        learn5AMin = np.argmin(lossLearn5A)
        learn8AMin = np.argmin(lossLearn8A)
        learn8BMin = np.argmin(lossLearn8B)
        minInds = [trotMin, strangMin, yoshMin, learn5AMin, learn8AMin, learn8BMin]

        for j, splitCol in enumerate(getSplitColours()):
            plotTo = minInds[j]+1
            axs[axsList[i]].loglog(numExps[i][j][:plotTo], loss[j][:plotTo], splitCol, linestyle=linSty[i], marker='', alpha=1.0)
    
    axs[0].loglog(numExps[0][0][:10], 10*[2], 'black', linestyle='-', alpha=0.5, label=r'Unitarity')
    orderLineWidth = np.array([200, 1000])
    axs[0].loglog(orderLineWidth, (1.00e-3/(orderLineWidth[0]**-1.0)) * np.power(orderLineWidth, -1.0), 'black', linestyle=':', alpha=0.5, label=r'Order 1, 2, 4')
    axs[0].loglog(orderLineWidth, (3.33e-4/(orderLineWidth[0]**-2.0)) * np.power(orderLineWidth, -2.0), 'black', linestyle=':', alpha=0.5)
    axs[0].loglog(orderLineWidth, (1.00e-4/(orderLineWidth[0]**-4.0)) * np.power(orderLineWidth, -4.0), 'black', linestyle=':', alpha=0.5)
    
    orderLineWidth = 4*np.array([200, 1000])
    axs[1].loglog(numExps[2][0][:10], 10*[2], 'black', linestyle='-', alpha=0.5)
    axs[1].loglog(orderLineWidth, (1.00e-3/(orderLineWidth[0]**-1.0)) * np.power(orderLineWidth, -1.0), 'black', linestyle=':', alpha=0.5)
    axs[1].loglog(orderLineWidth, (3.33e-4/(orderLineWidth[0]**-2.0)) * np.power(orderLineWidth, -2.0), 'black', linestyle=':', alpha=0.5)
    axs[1].loglog(orderLineWidth, (1.00e-4/(orderLineWidth[0]**-4.0)) * np.power(orderLineWidth, -4.0), 'black', linestyle=':', alpha=0.5)

    # Auxiliary settings
    handles, _ = axs[0].get_legend_handles_labels()
    handle1 = Line2D([0], [0], label='Training set up', marker='', color='black', linestyle='-', alpha=0.5)
    handle2 = Line2D([0], [0], label='Different potential', marker='', color='black', linestyle='--', alpha=0.5)
    handles.extend([handle1, handle2])
    axs[0].legend(loc='best', handles=handles)

    handles, _ = axs[1].get_legend_handles_labels()
    for i, splitCol in enumerate(getSplitColours()):
        handle = mpatches.Patch(color=splitCol, label=splitNames[i])
        handles.extend([handle])
    axs[1].legend(loc='best', handles=handles, ncols=2)

    axs[0].grid(which='major', color='#CCCCCC', linewidth=1.0)
    axs[0].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
    axs[0].set_xscale('log')

    axs[1].grid(which='major', color='#CCCCCC', linewidth=1.0)
    axs[1].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
    axs[1].set_xscale('log')

    fig.savefig(name+fileType, bbox_inches='tight', transparent=True, dpi=getDPI(fileType))

#############################
### FIGURE 8: lossLandGen ###
#############################
def plotLossLandGen(pageFracWidth=0.66, aspectRatio=1.0, fileType='.png', name='lossLandGen'):
    lossArr, xArr, yArr, zArr, sArr = loadLossLand('data/lossLandGen.npz')
    xsPlus, xsMinus, ysPlus, ysMinus, zs = getLen5Order4Man()
    box = np.array([[-0.5, 0.4], [-0.5, 0.4], [-0.5, 0.4]])
    inBoxPlus = (box[0,0] < xsPlus) & (xsPlus < box[0,1]) & (box[1,0] < ysPlus) & (ysPlus < box[1,1]) & (box[2,0] < zs) & (zs < box[2,1]) 
    inBoxMinus = (box[0,0] < xsMinus) & (xsMinus <box[0,1]) & (box[1,0] < ysMinus) & (ysMinus < box[1,1]) & (box[2,0] < zs) & (zs < box[2,1])
    
    # Set up fig
    plt.close('all')
    fig = plt.figure(figsize=(toFigSize(pageFracWidth, aspectRatio)), layout='constrained')
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

    fig.savefig(name+fileType, bbox_inches='tight', transparent=True, dpi=getDPI(fileType))

##############################
### FIGURE 9: bestFitCoefs ###
##############################
def plotBestFitCoefs(pageFracWidth=0.85, aspectRatio=2.0, fileType='.png', name='bestFitCoefs'):
    splitNames, losses, _, _, _, stepSizes = loadLossConvOrig()
    splitColours = getSplitColours()
    
    # Set up fig
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(toFigSize(pageFracWidth, aspectRatio)), layout='constrained')
    ax.set_xlabel(r'step size, $h$')
    ax.set_ylabel(r'$L_2$ error')

    learnedCoefs = [np.array([1.012819, 0.000100, 36.723808]), 
                    np.array([0.161597, 0.186682, 17.520861]), 
                    np.array([0.555560, 0.000150, 14.332067])]

    for i, coefs in enumerate(learnedCoefs):
        stepSize = stepSizes[3+i]
        ax.loglog(stepSize, losses[3+i], splitColours[3+i], linestyle='', marker='+', alpha=1.0, label=splitNames[3+i])
        lossPred = coefs[0]**2 * stepSize**2 + coefs[1]**2 * stepSize**4 + coefs[2]**2 * stepSize**6
        labelStr = r'$y = {:.3f}h^2 + {:.3f}h^4 + {:.1f}h^6$'.format(coefs[0]**2, coefs[1]**2, coefs[2]**2)
        ax.loglog(stepSize, lossPred, splitColours[3+i], linestyle='--', marker='', alpha=1.0, label=labelStr)

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(loc='best', ncols=2, handles=handles[0::2]+handles[1::2])
    ax.grid(which='major', color='#CCCCCC', linewidth=1.0)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
    ax.set_xscale('log')

    fig.savefig(name+fileType, bbox_inches='tight', transparent=True, dpi=getDPI(fileType))

##################################
### FIGURE 10: loss2dPlanesGen ###
##################################
def plotLoss2dPlanesGen(pageFracWidth=0.8, aspectRatio=1.0, fileType='.png', name='loss2dPlanesGen'):
    allPSS, allPSM, allPSF, allLS, allLM, allLF = loadLand2dPlanes('data/loss2dPlanesGenData', 'ordCondErr')

    def addSlice(x, y, loss, ax):
        cPlot = ax.contourf(x, y, loss, cmap = cmapComp, norm = normalizerComp, alpha=0.2, antialiased=True)
        cPlot = ax.contour(x, y, loss, alpha=1.0, colors='grey')
        ax.clabel(cPlot, inline=True)

    def pipeLine(allData, ax):
        xs, ys, zs, loss, cent, perp1, perp2 = allData
        lambda1, lambda2 = coordTransform(cent, perp1, perp2, xs, ys, zs)
        addSlice(lambda1, lambda2, loss, ax)

    plt.close('all')
    fig, axs = plt.subplots(2, 3, figsize=(toFigSize(pageFracWidth, aspectRatio)), layout='constrained', sharex=True, sharey=True)

    minVal = np.min(np.array([
             np.min(allPSS[3]),
             np.min(allPSM[3]),
             np.min(allPSF[3]),
             np.min(allLS[3]),
             np.min(allLM[3]),
             np.min(allLF[3])
    ]))

    maxVal = np.max(np.array([
             np.max(allPSS[3]),
             np.max(allPSM[3]),
             np.max(allPSF[3]),
             np.max(allLS[3]),
             np.max(allLM[3]),
             np.max(allLF[3])
    ]))

    cmapComp='bwr'
    normalizerComp=matplotlib.colors.Normalize(minVal, maxVal)
    imComp=matplotlib.cm.ScalarMappable(norm=normalizerComp, cmap = cmapComp)

    pipeLine(allPSS, axs[0,0])
    pipeLine(allPSM, axs[0,1])
    pipeLine(allPSF, axs[0,2])

    pipeLine(allLS, axs[1,0])
    pipeLine(allLM, axs[1,1])
    pipeLine(allLF, axs[1,2])

    speeds = [r'Slow Dyn $\perp e_1$', r'Med Dyn $\perp e_2$', r'Fast Dyn $\perp e_3$']
    xLabels = [r'$e_2$', r'$e_1$', r'$e_1$']
    yLabels = [r'$e_3$', r'$e_3$', r'$e_2$']
    for i in range(3):
        axs[0,i].set_title(speeds[i])
        axs[0,i].set_xlabel(xLabels[i])
        axs[0,i].set_ylabel(yLabels[i])
        axs[1,i].set_xlabel(xLabels[i])
        axs[1,i].set_ylabel(yLabels[i])

    minima = [r'$\gamma = [0.125, 0.25, 0.25]$', r'$\gamma = [0.3314, -0.07304, -0.1821]$']
    for i in range(2):
        tempAx = axs[i,2].twinx()
        tempAx.set_ylabel(minima[i])
        tempAx.set_yticks([])

    cbar = fig.colorbar(imComp, ax=axs)
    cbar.solids.set(alpha=0.2)
    
    fig.savefig(name+fileType, bbox_inches='tight', transparent=True, dpi=getDPI(fileType))

##################################
### FIGURE 11: sampleInitConds ###
##################################
def plotSampleInitConds(pageFracWidth=0.85, aspectRatio=2.0, fileType='.png', name='sampleInitConds'):
    randFigs = [9, 83, 122, 156, 170, 174]
    pot, xGrid, uInitsVal, _ = loadIvpRefs('data/ivpRefsOrig.npz')

    def pipeLine(ax, xGrid, uInit, pot, showAx2 = False):
        xGrid = np.array(xGrid)
        uInit = np.array(uInit)

        ax.plot(xGrid, np.abs(uInit), 'k', label='Mod')
        ax.plot(xGrid, np.real(uInit), 'b', label='Real')
        ax.plot(xGrid, np.imag(uInit), 'r', label='Imag')
        ax.grid(which='major', color='#CCCCCC', linewidth=1.0)
        ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)

        ax2 = ax.twinx()
        ax2.set_yscale('log')
        if showAx2:
            ax2.set_ylabel('Potential')
        else:
            ax2.set_yticklabels([])

        ax2.plot(xGrid, pot - min(pot) + 1, color='yellow', alpha=0.5, label='Potential')
        ax2.fill_between(xGrid, pot - min(pot) + 1, 0, color='yellow', alpha=0.1)

    # Set up fig
    plt.close('all')
    fig, axs = plt.subplots(2, 3, figsize=(toFigSize(pageFracWidth, aspectRatio)), layout='constrained', sharex=True, sharey=True)
    axs[1,0].set_xlabel('Spatial Domain')
    axs[1,1].set_xlabel('Spatial Domain')
    axs[1,2].set_xlabel('Spatial Domain')
    axs[0,0].set_ylabel('Magnitude of Sol')
    axs[1,0].set_ylabel('Magnitude of Sol')

    pipeLine(axs[0,0], xGrid, uInitsVal[0][randFigs[0]], pot)
    pipeLine(axs[0,1], xGrid, uInitsVal[0][randFigs[1]], pot)
    pipeLine(axs[0,2], xGrid, uInitsVal[0][randFigs[2]], pot, True)
    pipeLine(axs[1,0], xGrid, uInitsVal[0][randFigs[3]], pot)
    pipeLine(axs[1,1], xGrid, uInitsVal[0][randFigs[4]], pot)
    pipeLine(axs[1,2], xGrid, uInitsVal[0][randFigs[5]], pot, True)
    handles, _ = axs[1,1].get_legend_handles_labels()
    handle = Line2D([0], [0], color='yellow', alpha=0.5, label='Potential')
    handles.extend([handle])

    fig.legend(loc='center', ncols = 4, handles=handles, bbox_to_anchor=(0.5, -0.05))

    fig.savefig(name+fileType, bbox_inches='tight', transparent=True, dpi=getDPI(fileType))

#############################
### FIGURE 12: paramOptim ###
#############################
def plotParamOptim(pageFracWidth=0.85, aspectRatio=2.0, fileType='.png', name='paramOptim'):
    allLearns = loadParamLossTrain()
    splitNames = getSplitNames()
    splitColours = getSplitColours()

    def pipeLine(axs, val, loss, p1, p2, p3, p4, p5, p6, color, label, short = False):
        handle = axs[0].plot(val, color=color, linestyle='-', label=label)
        axs[0].plot(loss, color=color, linestyle='--', alpha=0.2)
        
        axs[1].plot(p1, color=color, linestyle='-', alpha=0.75)
        axs[1].plot(p2, color=color, linestyle='-', alpha=0.75)
        axs[1].plot(p3, color=color, linestyle='-', alpha=0.75)

        if short:
            return handle
    
        axs[1].plot(p4, color=color, linestyle='-', alpha=0.75)
        axs[1].plot(p5, color=color, linestyle='-', alpha=0.75)
        axs[1].plot(p6, color=color, linestyle='-', alpha=0.75)

        return handle

    # Set up fig
    plt.close('all')
    fig, axs = plt.subplots(1, 2, figsize=(toFigSize(pageFracWidth, aspectRatio)), layout='constrained')
    handles, _ = axs[0].get_legend_handles_labels()
    # handles = []

    for i, allLearn in enumerate(allLearns):
        handle = pipeLine(axs, allLearn[0], allLearn[1], allLearn[2], allLearn[3], allLearn[4], 
                          allLearn[5], allLearn[6], allLearn[7], splitColours[i+3], splitNames[i+3], allLearn[8])
        handles.extend(handle)
        # handles.append(handle)
        
    axs[0].set_xlabel(r'Training iteration')
    axs[0].grid(which='major', color='#CCCCCC', linewidth=1.0)
    axs[0].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
        
    axs[1].set_xlabel(r'Training iteration')
    axs[1].grid(which='major', color='#CCCCCC', linewidth=1.0)
    axs[1].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)

    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_ylabel(r'Loss')
    axs[1].set_xscale('log')
    axs[1].set_ylabel(r'$\gamma_k$')
    axs[0].legend(handles=handles, loc='best')

    fig.savefig(name+fileType, bbox_inches='tight', transparent=True, dpi=getDPI(fileType))

############################
### FIGURE 13: allOptims ###
############################
def plotAllOptims(pageFracWidth=0.85, aspectRatio=2.0, fileType='.png', name='allOptims'):
    datas = loadParamLossOptims()
    
    def pipeLine(axs, data):
        loss, x, y, z, color, label = data
        axs[0].plot(loss, color=color, linestyle='-')
        axs[1].plot(x, color=color, linestyle='-', label=label)
        axs[2].plot(y, color=color, linestyle='-')
        axs[3].plot(z, color=color, linestyle='-')
    
    # Set up fig
    plt.close('all')
    fig, axs = plt.subplots(1, 4, figsize=(toFigSize(pageFracWidth, aspectRatio)), layout='constrained')
    
    for data in datas:
        pipeLine(axs, data)

    for ax in axs.reshape(-1):
        ax.set_xlabel(r'Train iter')
        ax.grid(which='major', color='#CCCCCC', linewidth=1.0)
        ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)

    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_ylabel(r'Loss')
    axs[1].set_ylabel(r'$\gamma_1$')
    axs[2].set_ylabel(r'$\gamma_2$')
    axs[3].set_ylabel(r'$\gamma_3$')
    fig.legend(loc='center', ncols=4, bbox_to_anchor=(0.5, -0.15))

    fig.savefig(name+fileType, bbox_inches='tight', transparent=True, dpi=getDPI(fileType))

###################################################################################################################################

##################
### CALL PLOTS ###
##################
plotParamTransform( 0.85, 3.7, '.png', 'paramTransform' )
plotLossLandscape(  0.85, 1.2, '.png', 'lossLandscape'  )
plotLoss2dPlanes(   0.85, 1.7, '.png', 'loss2dPlanes'   )
plotSplitVisLearn(  0.85, 2.8, '.png', 'splitVisLearn'  )
plotLossConv(       0.85, 2.0, '.png', 'lossConv'       )
plotLossRelAdv(     0.85, 2.4, '.png', 'lossRelAdv'     )
plotLossConvGen(    0.85, 2.0, '.png', 'lossConvGen'    )
plotLossLandGen(    0.85, 1.2, '.png', 'lossLandGen'    )
plotBestFitCoefs(   0.85, 2.4, '.png', 'bestFitCoefs'   )
plotLoss2dPlanesGen(0.85, 1.7, '.png', 'loss2dPlanesGen')
plotSampleInitConds(0.85, 2.0, '.png', 'sampleInitConds')
plotParamOptim(     0.85, 3.2, '.png', 'paramOptim'     )
plotAllOptims(      0.85, 4.0, '.png', 'allOptims'      )