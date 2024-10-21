###############
### IMPORTS ###
###############

import numpy as np
import matplotlib
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

def loadLand2dPlanesOrig():
    allPSS = []
    with np.load('loss2dPlanesData/polyStrangSlwDyn.npz') as data:
        xs = data['xs']
        ys = data['ys']
        zs = data['zs']
        loss = data['losses']
        cent = data['cent']
        perp1 = data['perp1']
        perp2 = data['perp2']
        allPSS = [xs, ys, zs, loss, cent, perp1, perp2]
        
    allPSM = []
    with np.load('loss2dPlanesData/polyStrangMedDyn.npz') as data:
        xs = data['xs']
        ys = data['ys']
        zs = data['zs']
        loss = data['losses']
        cent = data['cent']
        perp1 = data['perp1']
        perp2 = data['perp2']
        allPSM = [xs, ys, zs, loss, cent, perp1, perp2]

    allPSF = []
    with np.load('loss2dPlanesData/polyStrangFstDyn.npz') as data:
        xs = data['xs']
        ys = data['ys']
        zs = data['zs']
        loss = data['losses']
        cent = data['cent']
        perp1 = data['perp1']
        perp2 = data['perp2']
        allPSF = [xs, ys, zs, loss, cent, perp1, perp2]

    allLS = []
    with np.load('loss2dPlanesData/learnedSlwDyn.npz') as data:
        xs = data['xs']
        ys = data['ys']
        zs = data['zs']
        loss = data['losses']
        cent = data['cent']
        perp1 = data['perp1']
        perp2 = data['perp2']
        allLS = [xs, ys, zs, loss, cent, perp1, perp2]

    allLM = []
    with np.load('loss2dPlanesData/learnedMedDyn.npz') as data:
        xs = data['xs']
        ys = data['ys']
        zs = data['zs']
        loss = data['losses']
        cent = data['cent']
        perp1 = data['perp1']
        perp2 = data['perp2']
        allLM = [xs, ys, zs, loss, cent, perp1, perp2]

    allLF = []
    with np.load('loss2dPlanesData/learnedFstDyn.npz') as data:
        xs = data['xs']
        ys = data['ys']
        zs = data['zs']
        loss = data['losses']
        cent = data['cent']
        perp1 = data['perp1']
        perp2 = data['perp2']
        allLF = [xs, ys, zs, loss, cent, perp1, perp2]
    
    return allPSS, allPSM, allPSF, allLS, allLM, allLF

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
def plotLoss2dPlanes(pageFracWidth=0.8, aspectRatio=1.0, fileType='.png', name='loss2dPlanes'):
    allPSS, allPSM, allPSF, allLS, allLM, allLF = loadLand2dPlanesOrig()

    def addSlice(x, y, loss, ax):
        cPlot = ax.contourf(x, y, loss, cmap = cmapComp, norm = normalizerComp, alpha=0.2, antialiased=True)
        cPlot = ax.contour(x, y, loss, alpha=1.0, colors='grey')
        ax.clabel(cPlot, inline=True)

    def pipeLine(allData, ax):
        xs, ys, zs, loss, cent, perp1, perp2 = allData
        lambda1, lambda2 = coordTransform(cent, perp1, perp2, xs, ys, zs)
        addSlice(lambda1, lambda2, loss, ax)

    plt.close('all')
    fig, axs = plt.subplots(3, 2, figsize=(toFigSize(pageFracWidth, aspectRatio)), layout="constrained", sharex=True, sharey=True)

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
    pipeLine(allPSM, axs[1,0])
    pipeLine(allPSF, axs[2,0])

    pipeLine(allLS, axs[0,1])
    pipeLine(allLM, axs[1,1])
    pipeLine(allLF, axs[2,1])

    cols = [r'$[0.125, 0.25, 0.25]$', r'$[0.3314, -0.07304, -0.1821]$']
    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    xLabels = [r'$e_2$', r'$e_1$', r'$e_1$']
    yLabels = [r'$e_3$', r'$e_3$', r'$e_2$']
    speeds = [r'Slow Dyn', r'Med Dyn', r'Fast Dyn']
    for i in range(len(axs)):
        axs[i,0].set_xlabel(xLabels[i])
        axs[i,0].set_ylabel(yLabels[i])
        axs[i,1].set_xlabel(xLabels[i])
        axs[i,1].set_ylabel(yLabels[i])
        tempAx = axs[i,1].twinx()
        # tempAx.set_ylabel(speeds[i], size='large')
        tempAx.set_ylabel(speeds[i])
        tempAx.set_yticks([])

    cbar = fig.colorbar(imComp, ax=axs)
    cbar.solids.set(alpha=0.2)
    
    # plt.tight_layout()
    # plt.show()
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
    fig, ax = plt.subplots(1, 1, figsize=(toFigSize(pageFracWidth, aspectRatio)), sharey=True)
    ax.set_xlabel(r'Cumulative $\alpha$')
    ax.set_ylabel(r'Cumulative $\beta$')

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

    ax.legend(loc='best')
    ax.grid(which='major', color='#CCCCCC', linewidth=1.0)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)

    plt.tight_layout()
    # plt.show()
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
    fig, ax = plt.subplots(1, 1, figsize=(toFigSize(pageFracWidth, aspectRatio)))
    ax.set_xlabel(r'number of exponentials')
    ax.set_ylabel(r'func $L_2$ norm')

    for i, splitCol in enumerate(getSplitColours()):
        plotTo = minInds[i]+1
        filtExps = numExps[i][:plotTo]
        ax.loglog(filtExps, losses[i][:plotTo], splitCol, linestyle='-', marker='', alpha=1.0, label=splitNames[i])
        ax.fill_between(filtExps, lowQs[i][:plotTo], highQs[i][:plotTo], color=splitCol, alpha=0.1)

    ax.loglog(numExps[0][:10], 10*[2], 'black', linestyle='-', alpha=0.5, label=r'Unitarity bound')
    ax.loglog(numExps[0][-4:], 2*(losses[0][-1]/(numExps[0][-1]**-1.0)) * np.power(numExps[0], -1.0)[-4:], 'black', linestyle=':', alpha=0.5, label=r'Order 1, 2, 4 lines')
    ax.loglog(numExps[1][-4:], 2*(losses[1][-1]/(numExps[1][-1]**-2.0)) * np.power(numExps[1], -2.0)[-4:], 'black', linestyle=':', alpha=0.5)
    ax.loglog(numExps[2][-4:], 2*(losses[2][-1]/(numExps[2][-1]**-4.0)) * np.power(numExps[2], -4.0)[-4:], 'black', linestyle=':', alpha=0.5)

    ax.legend(loc='best')
    ax.grid(which='major', color='#CCCCCC', linewidth=1.0)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
    ax.set_xscale('log')

    plt.tight_layout()
    # plt.show()
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
def plotLoss2dPlanesGen(pageFracWidth=0.8, aspectRatio=1.0, fileType='.png', name='loss2dPlanesGen'):
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
plotLoss2dPlanes(   0.80, 1.0, '.png', 'loss2dPlanes'   )
plotSplitVisLearn(  0.80, 2.0, '.png', 'splitVisLearn'  )
plotLossConv(       0.85, 2.0, '.png', 'lossConv'       )
plotLossRelAdv(     0.85, 1.7, '.png', 'lossRelAdv'     )
plotLossConvGen(    0.85, 2.0, '.png', 'lossConvGen'    )
plotLossLandGen(    0.66, 1.0, '.png', 'lossLandGen'    )
plotBestFitCoefs(   0.85, 2.0, '.png', 'bestFitCoefs'   )
plotLoss2dPlanesGen(0.80, 1.0, '.png', 'loss2dPlanesGen')
plotLossConvProj(   0.85, 2.0, '.png', 'lossConvProj'   )
plotSampleInitConds(0.85, 2.0, '.png', 'sampleInitConds')
plotParamOptim(     0.85, 2.0, '.png', 'paramOptim'     )
plotAllOptims(      0.85, 2.0, '.png', 'allOptims'      )