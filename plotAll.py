###############
### IMPORTS ###
###############

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX for text rendering
        "font.size": 10,  # Match LaTeX font size
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)
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


def getDPI(fileType=".png"):
    return "figure" if fileType == ".pdf" else 450


def getInds(incStrang4X, incLearn5AProj):
    if incStrang4X and incLearn5AProj:
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if incStrang4X:
        return [0, 1, 2, 3, 4, 5, 6, 8, 9]
    if incLearn5AProj:
        return [0, 1, 2, 4, 5, 6, 7, 8, 9]
    return [0, 1, 2, 4, 5, 6, 8, 9]


def getSplitNames(incStrang4X=False, incLearn5AProj=False):
    inds = getInds(incStrang4X, incLearn5AProj)
    names = [
        "Trotter",
        "Strang",
        "Yoshida",
        r"$4 \times$ Strang",
        "Learn5A",
        "Learn8A",
        "Learn8B",
        "Learn5AProj",
        "Blanes4",
        "Blanes7",
    ]
    return [names[i] for i in inds]


def getSplitColours(incStrang4X=False, incLearn5AProj=False):
    inds = getInds(incStrang4X, incLearn5AProj)
    colours = [
        "mediumvioletred",
        "firebrick",
        "peru",
        "sandybrown",
        "forestgreen",
        "royalblue",
        "lightskyblue",
        "lime",
        "black",
        "deeppink",
    ]
    return [colours[i] for i in inds]


def getSplitRelLen(incStrang4X=False, incLearn5AProj=False):
    def fnGen(mult, add):
        return lambda val: val * mult + add

    inds = getInds(incStrang4X, incLearn5AProj)
    fns = [
        fnGen(2 - 0, 0),
        fnGen(4 - 2, 1),
        fnGen(8 - 2, 1),
        fnGen(10 - 2, 1),
        fnGen(10 - 2, 1),
        fnGen(16 - 2, 1),
        fnGen(16 - 2, 1),
        fnGen(10 - 2, 1),
        fnGen(8 - 2, 1),
        fnGen(14 - 2, 1),
    ]
    return [fns[i] for i in inds]


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
        a[(numAlpha + 2) :] = np.flip(alpha, [0])

        b[:numBeta] = beta
        b[numAlpha] = 1 - 2 * betaSum
        b[(numBeta + 1) : (numParams + 1)] = np.flip(beta, [0])

    else:
        assert False

    return a, b


def getSymSplitGamma(incStrang4X=False, incLearn5AProj=False):
    inds = getInds(incStrang4X, incLearn5AProj)
    gammas = [
        None,  # Trotter
        [],  # Strang
        [0.6756035959798289, 1.3512071919596578],  # Yoshida
        [0.125, 0.25, 0.25],  # 4X Strang
        [0.36266572103945605, -0.10032032403589856, -0.1352975465549758],  # Learn5A
        [
            0.2134815929093979,
            -0.05820764895590353,
            0.41253264535444745,
            -0.13523357389399546,
            0.4443203153968813,
            -0.02509257759188356,
        ],  # Learn8A
        [
            0.11775349336573762,
            0.38763572759753917,
            0.36597039590662095,
            0.2921906385941626,
            0.05641644544703187,
            -0.021241661286415584,
        ],  # Learn8B
        [0.346035366644, -0.112267102113, -0.132],  # Learn5AProj
        [0.11888010966548, 0.29619504261126],  # Blanes4
        [
            0.0829844064174052,
            0.396309801498368,
            -0.0390563049223486,
            0.245298957184271,
            0.604872665711080,
        ],
    ]  # Blanes7

    return [gammas[i] for i in inds[1:]]


def getSplitAlphaBeta(incStrang4X=False, incLearn5AProj=False):
    gammas = getSymSplitGamma(incStrang4X, incLearn5AProj)
    alphas = [np.array([1.0])]  # Force add Trotter as not sym
    betas = [np.array([1.0])]  # Force add Trotter as not sym

    for gamma in gammas:
        alpha, beta = paramTransform(np.array(gamma))
        alphas.append(alpha)
        betas.append(beta)

    return alphas, betas


def getLen5Order4Man():
    offset = 0.001
    zs = np.concatenate(
        (np.linspace(-2.0, 0.0 - offset, 4000), np.linspace(0.5 + offset, 2.0, 3000))
    )
    inter1 = zs * (1.0 - 2 * zs)
    inter2 = np.sqrt(
        3 * np.power(zs, 4) - 3 * np.power(zs, 3) + np.power(zs, 2) - 0.125 * zs
    )
    inter3 = 3 * zs * (8 * np.power(zs, 3) - 8 * np.power(zs, 2) + 1.0)
    ysPlus = (inter1 + inter2) / inter3
    ysMinus = (inter1 - inter2) / inter3
    inter4 = 4 * np.power(zs, 2) - 4 * zs + 1.0
    xsPlus = 1.0 / 6.0 - ysPlus * inter4
    xsMinus = 1.0 / 6.0 - ysMinus * inter4
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
    coord1 = coordCand1 - scaleA * coordCand2
    coord2 = coordCand2 - scaleA * coordCand1

    scaleB = 1.0 / (1.0 - scaleA * scaleA)
    coord1 = scaleB * coord1
    coord2 = scaleB * coord2

    return coord1, coord2


###################################################################################################################################


#################
### LOAD DATA ###
#################
def loadLossConv(path, incLearn5AProj=False, timeRange=10):
    with np.load(path) as data:
        unitNumStepsTrot = data["unitNumStepsTrot"]
        unitNumStepsStrang = data["unitNumStepsStrang"]
        unitNumStepsYosh = data["unitNumStepsYosh"]

        unitNumStepsLearn5A = data["unitNumStepsLearn5A"]
        unitNumStepsLearn8A = data["unitNumStepsLearn8A"]
        unitNumStepsLearn8B = data["unitNumStepsLearn8B"]

        unitNumStepsLearn5AProj = data["unitNumStepsLearn5AProj"]
        unitNumStepsBlanes4 = data["unitNumStepsBlanes4"]
        unitNumStepsBlanes7 = data["unitNumStepsBlanes7"]

        lossTrotArr = data["lossTrot"]
        lossStrangArr = data["lossStrang"]
        lossYoshArr = data["lossYosh"]

        lossLearn5AArr = data["lossLearn5A"]
        lossLearn8AArr = data["lossLearn8A"]
        lossLearn8BArr = data["lossLearn8B"]

        lossLearn5AProjArr = data["lossLearn5AProj"]
        lossBlanes4Arr = data["lossBlanes4"]
        lossBlanes7Arr = data["lossBlanes7"]

    lossArrList = [
        lossTrotArr,
        lossStrangArr,
        lossYoshArr,
        None,
        lossLearn5AArr,
        lossLearn8AArr,
        lossLearn8BArr,
        lossLearn5AProjArr,
        lossBlanes4Arr,
        lossBlanes7Arr,
    ]
    unitNumStepsList = [
        unitNumStepsTrot,
        unitNumStepsStrang,
        unitNumStepsYosh,
        None,
        unitNumStepsLearn5A,
        unitNumStepsLearn8A,
        unitNumStepsLearn8B,
        unitNumStepsLearn5AProj,
        unitNumStepsBlanes4,
        unitNumStepsBlanes7,
    ]

    inds = getInds(False, incLearn5AProj)
    lossArrList = [lossArrList[i] for i in inds]
    unitNumStepsList = [unitNumStepsList[i] for i in inds]

    # Number of Exponentials
    numExpsList = [
        fn(unitNumSteps * timeRange)
        for (fn, unitNumSteps) in zip(
            getSplitRelLen(False, incLearn5AProj), unitNumStepsList
        )
    ]

    # Calc avg loss
    lossList = [np.quantile(np.abs(lossArr), 0.5, axis=0) for lossArr in lossArrList]

    # Calc std dev
    stdDev1 = 0.341
    stdDev2 = 0.477
    lowQList = [
        np.quantile(np.abs(lossArr), 0.5 - stdDev1, axis=0) for lossArr in lossArrList
    ]
    highQList = [
        np.quantile(np.abs(lossArr), 0.5 + stdDev1, axis=0) for lossArr in lossArrList
    ]

    # Calc step sizes
    stepSizeList = [unitNumSteps ** (-1.0) for unitNumSteps in unitNumStepsList]

    return (
        getSplitNames(False, incLearn5AProj),
        lossList,
        numExpsList,
        lowQList,
        highQList,
        stepSizeList,
    )


def loadLossLand(path):
    with np.load(path) as data:
        gammaVals1 = data["possibleVals1"]
        gammaVals2 = data["possibleVals2"]
        gammaVals3 = data["possibleVals3"]
        lossArr = data["allData"]

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
                xArr[i, j, k] = gammaVals1[i]
                yArr[i, j, k] = gammaVals2[j]
                zArr[i, j, k] = gammaVals3[k]
                sArr[i, j, k] = 25 * (
                    (1 - (lossArr[i, j, k] - minLoss) / (maxLoss - minLoss)) ** 5
                )

    return lossArr, xArr, yArr, zArr, sArr


def loadLand2dPlanes(path, name="losses"):
    def loadLand2dPlane(path, filename, name):
        with np.load(path + filename) as data:
            return [
                data["xs"],
                data["ys"],
                data["zs"],
                data[name],
                data["cent"],
                data["perp1"],
                data["perp2"],
            ]

    allPSS = loadLand2dPlane(path, "/polyStrangSlwDyn.npz", name)
    allPSM = loadLand2dPlane(path, "/polyStrangMedDyn.npz", name)
    allPSF = loadLand2dPlane(path, "/polyStrangFstDyn.npz", name)

    allLS = loadLand2dPlane(path, "/learnedSlwDyn.npz", name)
    allLM = loadLand2dPlane(path, "/learnedMedDyn.npz", name)
    allLF = loadLand2dPlane(path, "/learnedFstDyn.npz", name)

    return allPSS, allPSM, allPSF, allLS, allLM, allLF


def loadIvpRefs(path):
    with np.load(path) as data:
        pot = data["pot"]
        xGrid = data["xGrid"]
        uInitsVal = data["uInitsVal"]
        uFinalRefVal = data["uFinalRefVal"]

    return pot, xGrid, uInitsVal, uFinalRefVal


def loadParamLossTrain():
    with np.load("data/paramLossTrain.npz") as data:
        learn5AVal = data["learn5aVal"]
        learn8AVal = data["learn8aVal"]
        learn8BVal = data["learn8bVal"]

        learn5ALoss = data["learn5aLoss"]
        learn8ALoss = data["learn8aLoss"]
        learn8BLoss = data["learn8bLoss"]

        learn5AP1 = data["learn5aP1"]
        learn8AP1 = data["learn8aP1"]
        learn8BP1 = data["learn8bP1"]

        learn5AP2 = data["learn5aP2"]
        learn8AP2 = data["learn8aP2"]
        learn8BP2 = data["learn8bP2"]

        learn5AP3 = data["learn5aP3"]
        learn8AP3 = data["learn8aP3"]
        learn8BP3 = data["learn8bP3"]

        learn8AP4 = data["learn8aP4"]
        learn8BP4 = data["learn8bP4"]

        learn8AP5 = data["learn8aP5"]
        learn8BP5 = data["learn8bP5"]

        learn8AP6 = data["learn8aP6"]
        learn8BP6 = data["learn8bP6"]

    return [
        [learn5AVal, learn5ALoss, learn5AP1, learn5AP2, learn5AP3, [], [], [], True],
        [
            learn8AVal,
            learn8ALoss,
            learn8AP1,
            learn8AP2,
            learn8AP3,
            learn8AP4,
            learn8AP5,
            learn8AP6,
            False,
        ],
        [
            learn8BVal,
            learn8BLoss,
            learn8BP1,
            learn8BP2,
            learn8BP3,
            learn8BP4,
            learn8BP5,
            learn8BP6,
            False,
        ],
    ]


def loadParamLossOptims():
    with np.load("data/paramLossOptims.npz") as data:
        adaGL0p01M0p05Loss = data["adaGL0p01M0p05Loss"]
        adaGL0p01M0p10Loss = data["adaGL0p01M0p10Loss"]
        adamL0p01M0p05Loss = data["adamL0p01M0p05Loss"]
        adamL0p01M0p10Loss = data["adamL0p01M0p10Loss"]
        lionL0p01M0p05Loss = data["lionL0p01M0p05Loss"]
        lionL0p01M0p10Loss = data["lionL0p01M0p10Loss"]
        lemaL0p20M0p05Loss = data["lemaL0p20M0p05Loss"]
        lemaL0p20M0p10Loss = data["lemaL0p20M0p10Loss"]

        adaGL0p01M0p05X = data["adaGL0p01M0p05X"]
        adaGL0p01M0p10X = data["adaGL0p01M0p10X"]
        adamL0p01M0p05X = data["adamL0p01M0p05X"]
        adamL0p01M0p10X = data["adamL0p01M0p10X"]
        lionL0p01M0p05X = data["lionL0p01M0p05X"]
        lionL0p01M0p10X = data["lionL0p01M0p10X"]
        lemaL0p20M0p05X = data["lemaL0p20M0p05X"]
        lemaL0p20M0p10X = data["lemaL0p20M0p10X"]

        adaGL0p01M0p05Y = data["adaGL0p01M0p05Y"]
        adaGL0p01M0p10Y = data["adaGL0p01M0p10Y"]
        adamL0p01M0p05Y = data["adamL0p01M0p05Y"]
        adamL0p01M0p10Y = data["adamL0p01M0p10Y"]
        lionL0p01M0p05Y = data["lionL0p01M0p05Y"]
        lionL0p01M0p10Y = data["lionL0p01M0p10Y"]
        lemaL0p20M0p05Y = data["lemaL0p20M0p05Y"]
        lemaL0p20M0p10Y = data["lemaL0p20M0p10Y"]

        adaGL0p01M0p05Z = data["adaGL0p01M0p05Z"]
        adaGL0p01M0p10Z = data["adaGL0p01M0p10Z"]
        adamL0p01M0p05Z = data["adamL0p01M0p05Z"]
        adamL0p01M0p10Z = data["adamL0p01M0p10Z"]
        lionL0p01M0p05Z = data["lionL0p01M0p05Z"]
        lionL0p01M0p10Z = data["lionL0p01M0p10Z"]
        lemaL0p20M0p05Z = data["lemaL0p20M0p05Z"]
        lemaL0p20M0p10Z = data["lemaL0p20M0p10Z"]

    return [
        [
            adaGL0p01M0p05Loss,
            adaGL0p01M0p05X,
            adaGL0p01M0p05Y,
            adaGL0p01M0p05Z,
            "mediumvioletred",
            "AdaGrad 0.05",
        ],
        [
            adaGL0p01M0p10Loss,
            adaGL0p01M0p10X,
            adaGL0p01M0p10Y,
            adaGL0p01M0p10Z,
            "firebrick",
            "AdaGrad 0.1",
        ],
        [
            adamL0p01M0p05Loss,
            adamL0p01M0p05X,
            adamL0p01M0p05Y,
            adamL0p01M0p05Z,
            "peru",
            "Adam 0.05",
        ],
        [
            adamL0p01M0p10Loss,
            adamL0p01M0p10X,
            adamL0p01M0p10Y,
            adamL0p01M0p10Z,
            "darkorange",
            "Adam 0.1",
        ],
        [
            lionL0p01M0p05Loss,
            lionL0p01M0p05X,
            lionL0p01M0p05Y,
            lionL0p01M0p05Z,
            "yellowgreen",
            "Lion 0.05",
        ],
        [
            lionL0p01M0p10Loss,
            lionL0p01M0p10X,
            lionL0p01M0p10Y,
            lionL0p01M0p10Z,
            "forestgreen",
            "Lion 0.1",
        ],
        [
            lemaL0p20M0p05Loss,
            lemaL0p20M0p05X,
            lemaL0p20M0p05Y,
            lemaL0p20M0p05Z,
            "royalblue",
            "Lev-Marq 0.05",
        ],
        [
            lemaL0p20M0p10Loss,
            lemaL0p20M0p10X,
            lemaL0p20M0p10Y,
            lemaL0p20M0p10Z,
            "lightskyblue",
            "Lev-Marq 0.1",
        ],
    ]

def loadTrainDataSizeCorr():
    with np.load('data/trainDataSizeCorr.npz') as data:
        trainingSetSizes = data['trainingSetSizes']
        trainL2Loss = data['trainL2Loss']
        valL2Loss = data['valL2Loss']
        params = data['params']

    return trainingSetSizes, trainL2Loss, valL2Loss, params

###################################################################################################################################


################################
### FIGURE 1: paramTransform ###
################################
def plotParamTransform(
    pageFracWidth=0.85, aspectRatio=2.0, fileType=".png", name="paramTransform"
):
    plt.close("all")
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(toFigSize(pageFracWidth, aspectRatio)),
        sharey=True,
        layout="constrained",
    )
    axs[0].set_ylabel("Parameter Value")

    gammas = [np.array([0.4, 0.075, 0.3]), np.array([0.2, 0.3])]
    for i, gamma in enumerate(gammas):
        ax = axs[i]
        alpha, beta = paramTransform(gamma)
        lenGamma = len(gamma)
        hatches = (
            lenGamma * [r""] + 2 * [r"\\"] + [r"xx"] + lenGamma * [r"//"] + [""]
        )  # ['', '', '', '..', '..', '//..', '//', '//', '//', '//']
        color = ["#ccccff", "#ffcccc"] * (lenGamma + 1)
        numParams = 2 * (lenGamma + 2)
        knitParams = np.zeros(numParams)
        knitParams[::2] = alpha
        knitParams[1::2] = beta
        mids = np.linspace(0.0, 0.5, numParams + 1)
        mids = mids[:-1] + mids[1:]
        ax.bar(
            mids, knitParams, 1 / numParams, color=color, linewidth=0.0, hatch=hatches
        )
        ax.set_xticks(mids)
        # TODO: This is hacky.
        ax.set_xticklabels(
            np.array(
                [
                    r"$\alpha_1$",
                    r"$\beta_1$",
                    r"$\alpha_2$",
                    r"$\beta_2$",
                    r"$\alpha_3$",
                    r"$\beta_3$",
                    r"$\alpha_4$",
                    r"$\beta_4$",
                    r"$\alpha_5$",
                    r"$\beta_5$",
                ]
            )[:numParams]
        )
        ax.yaxis.grid(which="major", color="#CCCCCC", linewidth=1.0)

    handle1 = mpatches.Patch(facecolor="white", hatch=r"\\", label="Consistency")
    handle2 = mpatches.Patch(facecolor="white", hatch=r"//", label="Symmetry")
    axs[1].legend(handles=[handle1] + [handle2], loc="lower right", framealpha=1.0)

    fig.savefig(
        name + fileType, bbox_inches="tight", transparent=True, dpi=getDPI(fileType)
    )


###############################
### FIGURE 2: lossLandscape ###
###############################
def plotLossLandscape(
    pageFracWidth=0.95, aspectRatio=1.2, fileType=".png", name="lossLandscape"
):
    lossArr, xArr, yArr, zArr, sArr = loadLossLand("data/lossLandOrig.npz")
    xsPlus, xsMinus, ysPlus, ysMinus, zs = getLen5Order4Man()
    box = np.array([[-0.5, 0.4], [-0.5, 0.4], [-0.5, 0.4]])
    inBoxPlus = (
        (box[0, 0] < xsPlus)
        & (xsPlus < box[0, 1])
        & (box[1, 0] < ysPlus)
        & (ysPlus < box[1, 1])
        & (box[2, 0] < zs)
        & (zs < box[2, 1])
    )
    inBoxMinus = (
        (box[0, 0] < xsMinus)
        & (xsMinus < box[0, 1])
        & (box[1, 0] < ysMinus)
        & (ysMinus < box[1, 1])
        & (box[2, 0] < zs)
        & (zs < box[2, 1])
    )

    # Set up fig
    plt.close("all")
    fig = plt.figure(
        figsize=(toFigSize(pageFracWidth, aspectRatio)), layout="constrained"
    )
    ax = plt.axes(projection="3d")

    img = ax.scatter3D(
        xArr,
        yArr,
        zArr,
        c=lossArr,
        cmap=plt.gray(),
        alpha=0.3,
        s=sArr,
        vmin=0.0,
        vmax=2.0,
        linewidths=0,
    )
    ax.scatter3D(
        xsPlus[inBoxPlus],
        ysPlus[inBoxPlus],
        zs[inBoxPlus],
        c="blue",
        label=r"$4^\mathrm{th}$ Order manifold",
    )
    ax.scatter3D(
        xsMinus[inBoxMinus],
        ysMinus[inBoxMinus],
        zs[inBoxMinus],
        c="pink",
        label=r"_$4^\mathrm{th}$ Order manifold",
    )
    # TODO: Get from getSplitNames(), getSplitColours(), getSymSplitGamma()
    ax.scatter3D(0.125, 0.25, 0.25, c="green", label=r"$4 \times$ Strang")  # PolyStrang
    ax.scatter3D(
        0.36266572103945605,
        -0.10032032403589856,
        -0.1352975465549758,
        c="red",
        label="Learn5A",
    )  # Learn5A

    ax.set_xlabel(r"$\gamma_1$")
    ax.set_ylabel(r"$\gamma_2$")
    ax.set_zlabel(r"$\gamma_3$")
    cbar = fig.colorbar(img, location="right", shrink=0.5, pad=0.04)
    cbar.solids.set(alpha=1.0)
    ax.legend(loc="upper right")

    fig.savefig(
        name + fileType, bbox_inches="tight", transparent=True, dpi=getDPI(fileType)
    )


##############################
### FIGURE 3: loss2dPlanes ###
##############################
def plotLoss2dPlanes(
    pageFracWidth=0.8, aspectRatio=1.0, fileType=".png", name="loss2dPlanes"
):
    allPSS, allPSM, allPSF, allLS, allLM, allLF = loadLand2dPlanes(
        "data/loss2dPlanesData"
    )

    def addSlice(x, y, loss, ax):
        cPlot = ax.contourf(
            x, y, loss, cmap=cmapComp, norm=normalizerComp, alpha=0.2, antialiased=True
        )
        cPlot = ax.contour(x, y, loss, alpha=1.0, colors="grey")
        ax.clabel(cPlot, inline=True)

    def pipeLine(allData, ax):
        xs, ys, zs, loss, cent, perp1, perp2 = allData
        lambda1, lambda2 = coordTransform(cent, perp1, perp2, xs, ys, zs)
        addSlice(lambda1, lambda2, loss, ax)

    plt.close("all")
    fig, axs = plt.subplots(
        2,
        3,
        figsize=(toFigSize(pageFracWidth, aspectRatio)),
        layout="constrained",
        sharex=True,
        sharey=True,
    )

    minVal = np.min(
        np.array(
            [
                np.min(allPSS[3]),
                np.min(allPSM[3]),
                np.min(allPSF[3]),
                np.min(allLS[3]),
                np.min(allLM[3]),
                np.min(allLF[3]),
            ]
        )
    )

    maxVal = np.max(
        np.array(
            [
                np.max(allPSS[3]),
                np.max(allPSM[3]),
                np.max(allPSF[3]),
                np.max(allLS[3]),
                np.max(allLM[3]),
                np.max(allLF[3]),
            ]
        )
    )

    cmapComp = "bwr"
    normalizerComp = matplotlib.colors.Normalize(minVal, maxVal)
    imComp = matplotlib.cm.ScalarMappable(norm=normalizerComp, cmap=cmapComp)

    pipeLine(allPSS, axs[0, 0])
    pipeLine(allPSM, axs[0, 1])
    pipeLine(allPSF, axs[0, 2])

    pipeLine(allLS, axs[1, 0])
    pipeLine(allLM, axs[1, 1])
    pipeLine(allLF, axs[1, 2])

    speeds = [r"Slow Dyn $\perp e_1$", r"Med Dyn $\perp e_2$", r"Fast Dyn $\perp e_3$"]
    xLabels = [r"$e_2$", r"$e_1$", r"$e_1$"]
    yLabels = [r"$e_3$", r"$e_3$", r"$e_2$"]
    for i in range(3):
        axs[0, i].set_title(speeds[i])
        axs[0, i].set_xlabel(xLabels[i])
        axs[0, i].set_ylabel(yLabels[i])
        axs[1, i].set_xlabel(xLabels[i])
        axs[1, i].set_ylabel(yLabels[i])

    minima = [
        r"$\gamma = [0.125, 0.25, 0.25]$",
        r"$\gamma = [0.3314, -0.07304, -0.1821]$",
    ]
    for i in range(2):
        tempAx = axs[i, 2].twinx()
        tempAx.set_ylabel(minima[i])
        tempAx.set_yticks([])

    cbar = fig.colorbar(imComp, ax=axs)
    cbar.solids.set(alpha=0.2)

    fig.savefig(
        name + fileType, bbox_inches="tight", transparent=True, dpi=getDPI(fileType)
    )


###############################
### FIGURE 4: splitVisLearn ###
###############################
def plotSplitVisLearn(
    pageFracWidth=0.8, aspectRatio=2.0, fileType=".png", name="splitVisLearn"
):
    def pipeLine(ax, alpha, beta, colour="grey", label="", traceWidth=-1.0):
        assert len(alpha) == len(beta)
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
    plt.close("all")
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(toFigSize(pageFracWidth, aspectRatio)),
        sharey=True,
        layout="constrained",
    )
    ax.set_xlabel(r"$u_1(t)$")
    ax.set_ylabel(r"$u_2(t)$")

    alphas, betas = getSplitAlphaBeta(True, True)
    splitColours = getSplitColours(True, True)
    for i, splitName in enumerate(getSplitNames(True, True)):
        pipeLine(ax, alphas[i], betas[i], splitColours[i], splitName)

    ax.legend(loc="center", ncols=1, bbox_to_anchor=(-0.23, 0.5))
    ax.grid(which="major", color="#CCCCCC", linewidth=1.0)
    ax.grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=0.7)

    fig.savefig(
        name + fileType, bbox_inches="tight", transparent=True, dpi=getDPI(fileType)
    )


##########################
### FIGURE 5: lossConv ###
##########################
def plotLossConv(pageFracWidth=0.95, aspectRatio=2.0, fileType=".png", name="lossConv"):
    splitNames, losses, numExps, lowQs, highQs, _ = loadLossConv(
        "data/lossConvOrig.npz", True
    )
    minInds = [
        np.argmin(loss) + 1 for loss in losses
    ]  # TODO: Check why the +1 does not overflow

    # Set up fig
    plt.close("all")
    fig, axs = plt.subplots(
        1, 2, figsize=(toFigSize(pageFracWidth, aspectRatio)), layout="constrained"
    )
    for j, ax in enumerate(axs):
        ax.set_xlabel(r"number of exponentials")
        ax.set_ylabel(r"func $L_2$ norm")
        for i, splitCol in enumerate(getSplitColours(False, True)):
            label = splitNames[i] if j == 0 else None
            plotTo = minInds[i]
            if splitNames[i] == "Learn5AProj":
                plotTo -= 1
            filtExps = numExps[i][:plotTo]
            ax.loglog(
                filtExps,
                losses[i][:plotTo],
                splitCol,
                linestyle="-",
                marker="",
                alpha=1.0,
                label=label,
            )
            ax.fill_between(
                filtExps,
                lowQs[i][:plotTo],
                highQs[i][:plotTo],
                color=splitCol,
                alpha=0.2,
                linewidth=0,
            )
        ax.loglog(numExps[0][:10], 10 * [2], "black", linestyle="-", alpha=0.5)
        orderLineWidth = np.array([200, 1000])
        if j == 1:
            ax.loglog(
                orderLineWidth,
                (1.00e-3 / (orderLineWidth[0] ** -1.0))
                * np.power(orderLineWidth, -1.0),
                "black",
                linestyle=":",
                alpha=0.5,
                label=r"Order 1, 2, 4",
            )
            ax.loglog(
                orderLineWidth,
                (3.33e-4 / (orderLineWidth[0] ** -2.0))
                * np.power(orderLineWidth, -2.0),
                "black",
                linestyle=":",
                alpha=0.5,
            )
            ax.loglog(
                orderLineWidth,
                (1.00e-4 / (orderLineWidth[0] ** -4.0))
                * np.power(orderLineWidth, -4.0),
                "black",
                linestyle=":",
                alpha=0.5,
            )

        ax.legend(loc="best", ncols=1, fontsize=9)
        ax.grid(which="major", color="#CCCCCC", linewidth=1.0)
        ax.grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=0.7)
        ax.set_xscale("log")
        if j == 0:
            ax.set_xlim(100, 2.1e3)
            ax.set_ylim(1e-4, 3)
            ax.set_xticks([100, 200, 400, 1000, 2000])
            ax.set_xticklabels([100, 200, 400, 1000, 2000])
        if j == 1:
            ax.set_ylabel(None)

    axs[1].annotate("Unitarity bound", xy=(2e3, 1.5))

    fig.savefig(
        name + fileType, bbox_inches="tight", transparent=True, dpi=getDPI(fileType)
    )


############################
### FIGURE 6: lossRelAdv ###
############################
def plotLossRelAdv(
    pageFracWidth=0.85, aspectRatio=1.7, fileType=".png", name="lossRelAdv"
):
    _, losses, numExps, _, _, _ = loadLossConv("data/lossConvOrig.npz", True)
    (
        lossTrot,
        lossStrang,
        lossYosh,
        lossLearn5A,
        lossLearn8A,
        lossLearn8B,
        lossLearn5AProj,
        lossBlanes4,
        lossBlanes7,
    ) = losses
    (
        numExpsTrot,
        numExpsStrang,
        numExpsYosh,
        numExpsLearn5A,
        numExpsLearn8A,
        numExpsLearn8B,
        numExpsLearn5AProj,
        numExpsBlanes4,
        numExpsBlanes7,
    ) = numExps

    classicalNumExps = np.concatenate(
        (numExpsTrot, numExpsStrang, numExpsYosh, numExpsBlanes4, numExpsBlanes7),
        axis=0,
    )
    learnedNumExps = np.concatenate(
        (numExpsLearn5A, numExpsLearn8A, numExpsLearn8B, numExpsLearn5AProj), axis=0
    )

    numExpsCommon = np.geomspace(
        max(min(classicalNumExps), min(learnedNumExps)),
        min(max(classicalNumExps), max(learnedNumExps)),
        100000,
    )

    lossTrotCommon = np.exp(
        np.interp(np.log(numExpsCommon), np.log(numExpsTrot), np.log(lossTrot))
    )
    lossStrangCommon = np.exp(
        np.interp(np.log(numExpsCommon), np.log(numExpsStrang), np.log(lossStrang))
    )
    lossYoshCommon = np.exp(
        np.interp(np.log(numExpsCommon), np.log(numExpsYosh), np.log(lossYosh))
    )
    lossBlanes4Common = np.exp(
        np.interp(np.log(numExpsCommon), np.log(numExpsBlanes4), np.log(lossBlanes4))
    )
    lossBlanes7Common = np.exp(
        np.interp(np.log(numExpsCommon), np.log(numExpsBlanes7), np.log(lossBlanes7))
    )

    lossLearn5ACommon = np.exp(
        np.interp(np.log(numExpsCommon), np.log(numExpsLearn5A), np.log(lossLearn5A))
    )
    lossLearn8ACommon = np.exp(
        np.interp(np.log(numExpsCommon), np.log(numExpsLearn8A), np.log(lossLearn8A))
    )
    lossLearn8BCommon = np.exp(
        np.interp(np.log(numExpsCommon), np.log(numExpsLearn8B), np.log(lossLearn8B))
    )
    lossLearn5AProjCommon = np.exp(
        np.interp(
            np.log(numExpsCommon), np.log(numExpsLearn5AProj), np.log(lossLearn5AProj)
        )
    )

    ratioAll = np.fmin(
        np.fmin(
            np.fmin(np.fmin(lossTrotCommon, lossStrangCommon), lossYoshCommon),
            lossBlanes4Common,
        ),
        lossBlanes7Common,
    ) / np.fmin(
        np.fmin(np.fmin(lossLearn5ACommon, lossLearn8ACommon), lossLearn8BCommon),
        lossLearn5AProjCommon,
    )
    ratioBlanes7L5A = lossBlanes7Common / lossLearn5ACommon
    ratioBlanes7L8A = lossBlanes7Common / lossLearn8ACommon
    ratioBlanes7L8B = lossBlanes7Common / lossLearn8BCommon
    ratioBlanes7L5P = lossBlanes7Common / lossLearn5AProjCommon

    # allMin = np.argmin(ratioAll)+1
    # blanes7L5AMin = np.argmin(ratioBlanes7L5A)+1
    # blanes7L8AMin = np.argmin(ratioBlanes7L8A)+1
    # blanes7L8BMin = np.argmin(ratioBlanes7L8B)+1
    # blanes7L5PMin = np.argmin(ratioBlanes7L5P)+1

    # Can set to all data
    allMin = len(ratioAll)
    blanes7L5AMin = len(ratioBlanes7L5A)
    blanes7L8AMin = len(ratioBlanes7L8A)
    blanes7L8BMin = len(ratioBlanes7L8B)
    blanes7L5PMin = len(ratioBlanes7L5P)

    # Set up fig
    plt.close("all")
    fig, ax = plt.subplots(
        1, 1, figsize=(toFigSize(pageFracWidth, aspectRatio)), layout="constrained"
    )
    ax.set_xlabel(r"number of exponentials")
    ax.set_ylabel(r"relative advantage of learned")

    minMin = np.min(
        [allMin, blanes7L5AMin, blanes7L8AMin, blanes7L8BMin, blanes7L5PMin]
    )
    ax.plot(
        numExpsCommon[:minMin],
        minMin * [1.0],
        "black",
        linestyle="-",
        alpha=1.0,
        label=r"Equal performance",
    )
    ax.text(1.8e4, 1.05, "learned better", ha="center", va="bottom")
    ax.text(1.8e4, 0.95, "classical better", ha="center", va="top")

    ax.plot(
        numExpsCommon[:blanes7L5AMin],
        ratioBlanes7L5A[:blanes7L5AMin],
        "forestgreen",
        linestyle="-",
        marker="",
        alpha=1.0,
        label="Blanes7 vs Learn5A",
    )
    ax.plot(
        numExpsCommon[:blanes7L8AMin],
        ratioBlanes7L8A[:blanes7L8AMin],
        "royalblue",
        linestyle="-",
        marker="",
        alpha=1.0,
        label="Blanes7 vs Learn8A",
    )
    ax.plot(
        numExpsCommon[:blanes7L8BMin],
        ratioBlanes7L8B[:blanes7L8BMin],
        "lightskyblue",
        linestyle="-",
        marker="",
        alpha=1.0,
        label="Blanes7 vs Learn8B",
    )
    ax.plot(
        numExpsCommon[:blanes7L5PMin],
        ratioBlanes7L5P[:blanes7L5PMin],
        "lime",
        linestyle="-",
        marker="",
        alpha=1.0,
        label="Blanes7 vs Learn5AProj",
    )
    ax.plot(
        numExpsCommon[:allMin],
        ratioAll[:allMin],
        "black",
        linestyle=":",
        linewidth=2,
        marker="",
        alpha=1.0,
        label="Best Classical vs Best Learned",
    )

    ax.legend(loc="upper right", ncols=2, fontsize=9)
    ax.grid(which="major", color="#CCCCCC", linewidth=1.0)
    ax.grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=0.7)
    ax.set_xscale("log")
    ax.set_ylim(0, 2)
    ax.set_xlim(None, 4e5)
    ax.set_yscale("linear")

    fig.savefig(
        name + fileType, bbox_inches="tight", transparent=True, dpi=getDPI(fileType)
    )

    #### Generate table
    # number of exponentials
    optExp = np.argmax(ratioAll)
    lossTable = {
        "Trotter": lossTrotCommon,
        "Strang": lossStrangCommon,
        "Yoshida": lossYoshCommon,
        "Blanes4": lossBlanes4Common,
        "Blanes7": lossBlanes7Common,
        "Learn5A": lossLearn5ACommon,
        "Learn8A": lossLearn8ACommon,
        "Learn8B": lossLearn8BCommon,
        "Learn5AProj": lossLearn5AProjCommon,
    }
    lossBlanes7 = lossTable["Blanes7"][optExp]

    with open(name + "table.tex", encoding="utf-8", mode="w") as f:
        print(r"\begin{tabular}{ |c|c|c|c| }", file=f)
        print(r"\hline", file=f)
        print(
            r"\multirow{2}{*}{\textbf{Splitting}} & \textbf{$L_2$-error for} & \textbf{Rel accuracy} & \textbf{Rel speed} \\ ",
            file=f,
        )
        print(
            r"& \textbf{$",
            int(round(numExpsCommon[optExp])),
            r"$ subflows} & \textbf{vs Blanes7}  & \textbf{vs Balnes7} \\",
            file=f,
        )
        print(r"\hline \hline", file=f)
        for method, loss in lossTable.items():
            sameErrorExp = np.argmin((loss - lossBlanes7) ** 2)
            speedup = numExpsCommon[optExp] / numExpsCommon[sameErrorExp]
            print(
                f"{method:16s} & ${loss[optExp]:8.3f}$ & ${lossBlanes7/loss[optExp]:8.2f}$ & ${speedup:8.2f}$",
                end="",
                file=f,
            )
            print(r"\\\hline", file=f)
        print(r"\end{tabular}", file=f)


#############################
### FIGURE 7: lossConvGen ###
#############################
def plotLossConvGen(
    pageFracWidth=0.85, aspectRatio=2.0, fileType=".png", name="lossConvGen"
):
    splitNames, origlosses, origNumExps, _, _, _ = loadLossConv(
        "data/lossConvOrig.npz", True
    )
    _, newPotlosses, newPotNumExps, _, _, _ = loadLossConv(
        "data/lossConvNewPot.npz", True
    )
    _, newAlllosses, newAllNumExps, _, _, _ = loadLossConv(
        "data/lossConvAllNew.npz", True, 30
    )
    losses = [origlosses, newPotlosses, newAlllosses]
    numExps = [origNumExps, newPotNumExps, newAllNumExps]

    # Set up fig
    plt.close("all")
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(toFigSize(pageFracWidth, aspectRatio)),
        sharey=True,
        layout="constrained",
    )
    axs[0].set_xlabel(r"number of exponentials")
    axs[1].set_xlabel(r"number of exponentials")
    axs[0].set_ylabel(r"func $L_2$ norm")
    axsList = [0, 0, 1]
    linSty = ["-", "--", "-"]

    for i, loss in enumerate(losses):
        minInds = [np.argmin(lossSub) + 1 for lossSub in loss]
        # minInds = [len(lossSub) for lossSub in loss]
        minInds[6] -= 1
        for j, splitCol in enumerate(getSplitColours(False, True)):
            plotTo = minInds[j]
            axs[axsList[i]].loglog(
                numExps[i][j][:plotTo],
                loss[j][:plotTo],
                splitCol,
                linestyle=linSty[i],
                marker="",
                alpha=1.0,
            )

    axs[0].loglog(
        numExps[0][0][:10],
        10 * [2],
        "black",
        linestyle="-",
        alpha=0.5,
        label=r"Unitarity",
    )
    orderLineWidth = np.array([200, 1000])
    axs[0].loglog(
        orderLineWidth,
        (1.00e-3 / (orderLineWidth[0] ** -1.0)) * np.power(orderLineWidth, -1.0),
        "black",
        linestyle=":",
        alpha=0.5,
        label=r"Order 1, 2, 4",
    )
    axs[0].loglog(
        orderLineWidth,
        (3.33e-4 / (orderLineWidth[0] ** -2.0)) * np.power(orderLineWidth, -2.0),
        "black",
        linestyle=":",
        alpha=0.5,
    )
    axs[0].loglog(
        orderLineWidth,
        (1.00e-4 / (orderLineWidth[0] ** -4.0)) * np.power(orderLineWidth, -4.0),
        "black",
        linestyle=":",
        alpha=0.5,
    )

    orderLineWidth = 4 * np.array([1e4, 5e4])
    axs[1].loglog(numExps[2][0][:10], 10 * [2], "black", linestyle="-", alpha=0.5)
    axs[1].loglog(
        orderLineWidth,
        (1.00e-7 / (orderLineWidth[0] ** -1.0)) * np.power(orderLineWidth, -1.0),
        "black",
        linestyle=":",
        alpha=0.5,
    )
    axs[1].loglog(
        orderLineWidth,
        (3.33e-8 / (orderLineWidth[0] ** -2.0)) * np.power(orderLineWidth, -2.0),
        "black",
        linestyle=":",
        alpha=0.5,
    )
    axs[1].loglog(
        orderLineWidth,
        (1.00e-8 / (orderLineWidth[0] ** -4.0)) * np.power(orderLineWidth, -4.0),
        "black",
        linestyle=":",
        alpha=0.5,
    )

    # Auxiliary settings
    handles, _ = axs[0].get_legend_handles_labels()
    handle1 = Line2D(
        [0],
        [0],
        label="Training set up",
        marker="",
        color="black",
        linestyle="-",
        alpha=0.5,
    )
    handle2 = Line2D(
        [0],
        [0],
        label="Different potential",
        marker="",
        color="black",
        linestyle="--",
        alpha=0.5,
    )
    handles.extend([handle1, handle2])
    axs[0].legend(loc="best", handles=handles)

    handles, _ = axs[1].get_legend_handles_labels()
    for i, splitCol in enumerate(getSplitColours(False, True)):
        handle = mpatches.Patch(color=splitCol, label=splitNames[i])
        handles.extend([handle])
    axs[1].legend(loc="best", handles=handles, ncols=1)

    axs[0].grid(which="major", color="#CCCCCC", linewidth=1.0)
    axs[0].grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=0.7)
    axs[0].set_xscale("log")

    axs[1].grid(which="major", color="#CCCCCC", linewidth=1.0)
    axs[1].grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=0.7)
    axs[1].set_xscale("log")

    fig.savefig(
        name + fileType, bbox_inches="tight", transparent=True, dpi=getDPI(fileType)
    )


#############################
### FIGURE 8: lossLandGen ###
#############################
def plotLossLandGen(
    pageFracWidth=0.66, aspectRatio=1.0, fileType=".png", name="lossLandGen"
):
    lossArr, xArr, yArr, zArr, sArr = loadLossLand("data/lossLandGen.npz")
    xsPlus, xsMinus, ysPlus, ysMinus, zs = getLen5Order4Man()
    box = np.array([[-0.5, 0.4], [-0.5, 0.4], [-0.5, 0.4]])
    inBoxPlus = (
        (box[0, 0] < xsPlus)
        & (xsPlus < box[0, 1])
        & (box[1, 0] < ysPlus)
        & (ysPlus < box[1, 1])
        & (box[2, 0] < zs)
        & (zs < box[2, 1])
    )
    inBoxMinus = (
        (box[0, 0] < xsMinus)
        & (xsMinus < box[0, 1])
        & (box[1, 0] < ysMinus)
        & (ysMinus < box[1, 1])
        & (box[2, 0] < zs)
        & (zs < box[2, 1])
    )

    # Set up fig
    plt.close("all")
    fig = plt.figure(
        figsize=(toFigSize(pageFracWidth, aspectRatio)), layout="constrained"
    )
    ax = plt.axes(projection="3d")

    img = ax.scatter3D(
        xArr,
        yArr,
        zArr,
        c=lossArr,
        cmap=plt.gray(),
        alpha=0.3,
        s=sArr,
        vmin=0.0,
        vmax=2.0,
        linewidths=0,
    )
    ax.scatter3D(
        xsPlus[inBoxPlus],
        ysPlus[inBoxPlus],
        zs[inBoxPlus],
        c="blue",
        label=r"$4^\mathrm{th}$ Order manifold",
    )
    ax.scatter3D(
        xsMinus[inBoxMinus],
        ysMinus[inBoxMinus],
        zs[inBoxMinus],
        c="pink",
        label=r"_$4^\mathrm{th}$ Order manifold",
    )
    # TODO: Get from getSplitNames(), getSplitColours(), getSymSplitGamma()
    ax.scatter3D(0.125, 0.25, 0.25, c="green", label=r"$4 \times$ Strang")  # PolyStrang
    ax.scatter3D(
        0.36266572103945605,
        -0.10032032403589856,
        -0.1352975465549758,
        c="red",
        label="Learn5A",
    )  # Learn5A

    ax.set_xlabel(r"$\gamma_1$")
    ax.set_ylabel(r"$\gamma_2$")
    ax.set_zlabel(r"$\gamma_3$")
    # cbar = fig.colorbar(img, location='right', shrink=0.5, pad=0.04)
    # cbar.solids.set(alpha=1.0)
    # ax.legend(loc='upper right')

    fig.savefig(
        name + fileType, bbox_inches="tight", transparent=True, dpi=getDPI(fileType)
    )


##############################
### FIGURE 9: bestFitCoefs ###
##############################
def plotBestFitCoefs(
    pageFracWidth=0.85, aspectRatio=2.0, fileType=".png", name="bestFitCoefs"
):
    splitNames, losses, _, _, _, stepSizes = loadLossConv("data/lossConvOrig.npz")
    splitColours = getSplitColours()

    # Set up fig
    plt.close("all")
    fig, ax = plt.subplots(
        1, 1, figsize=(toFigSize(pageFracWidth, aspectRatio)), layout="constrained"
    )
    ax.set_xlabel(r"step size, $h$")
    ax.set_ylabel(r"$L_2$ error")

    learnedCoefs = [
        np.array([1.012819, 0.000100, 36.723808]),
        np.array([0.161597, 0.186682, 17.520861]),
        np.array([0.555560, 0.000150, 14.332067]),
    ]

    for i, coefs in enumerate(learnedCoefs):
        stepSize = stepSizes[3 + i]
        ax.loglog(
            stepSize,
            losses[3 + i],
            splitColours[3 + i],
            linestyle="",
            marker="+",
            alpha=1.0,
            label=splitNames[3 + i],
        )
        lossPred = (
            coefs[0] ** 2 * stepSize**2
            + coefs[1] ** 2 * stepSize**4
            + coefs[2] ** 2 * stepSize**6
        )
        labelStr = r"$y = {:.3f}h^2 + {:.3f}h^4 + {:.1f}h^6$".format(
            coefs[0] ** 2, coefs[1] ** 2, coefs[2] ** 2
        )
        ax.loglog(
            stepSize,
            lossPred,
            splitColours[3 + i],
            linestyle="--",
            marker="",
            alpha=1.0,
            label=labelStr,
        )

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(loc="best", ncols=2, handles=handles[0::2] + handles[1::2])
    ax.grid(which="major", color="#CCCCCC", linewidth=1.0)
    ax.grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=0.7)
    ax.set_xscale("log")

    fig.savefig(
        name + fileType, bbox_inches="tight", transparent=True, dpi=getDPI(fileType)
    )


##################################
### FIGURE 10: loss2dPlanesGen ###
##################################
def plotLoss2dPlanesGen(
    pageFracWidth=0.8, aspectRatio=1.0, fileType=".png", name="loss2dPlanesGen"
):
    allPSS, allPSM, allPSF, allLS, allLM, allLF = loadLand2dPlanes(
        "data/loss2dPlanesGenData", "ordCondErr"
    )

    def addSlice(x, y, loss, ax):
        cPlot = ax.contourf(
            x, y, loss, cmap=cmapComp, norm=normalizerComp, alpha=0.2, antialiased=True
        )
        cPlot = ax.contour(x, y, loss, alpha=1.0, colors="grey")
        ax.clabel(cPlot, inline=True)

    def pipeLine(allData, ax):
        xs, ys, zs, loss, cent, perp1, perp2 = allData
        lambda1, lambda2 = coordTransform(cent, perp1, perp2, xs, ys, zs)
        addSlice(lambda1, lambda2, loss, ax)

    plt.close("all")
    fig, axs = plt.subplots(
        2,
        3,
        figsize=(toFigSize(pageFracWidth, aspectRatio)),
        layout="constrained",
        sharex=True,
        sharey=True,
    )

    minVal = np.min(
        np.array(
            [
                np.min(allPSS[3]),
                np.min(allPSM[3]),
                np.min(allPSF[3]),
                np.min(allLS[3]),
                np.min(allLM[3]),
                np.min(allLF[3]),
            ]
        )
    )

    maxVal = np.max(
        np.array(
            [
                np.max(allPSS[3]),
                np.max(allPSM[3]),
                np.max(allPSF[3]),
                np.max(allLS[3]),
                np.max(allLM[3]),
                np.max(allLF[3]),
            ]
        )
    )

    cmapComp = "bwr"
    normalizerComp = matplotlib.colors.Normalize(minVal, maxVal)
    imComp = matplotlib.cm.ScalarMappable(norm=normalizerComp, cmap=cmapComp)

    pipeLine(allPSS, axs[0, 0])
    pipeLine(allPSM, axs[0, 1])
    pipeLine(allPSF, axs[0, 2])

    pipeLine(allLS, axs[1, 0])
    pipeLine(allLM, axs[1, 1])
    pipeLine(allLF, axs[1, 2])

    speeds = [r"Slow Dyn $\perp e_1$", r"Med Dyn $\perp e_2$", r"Fast Dyn $\perp e_3$"]
    xLabels = [r"$e_2$", r"$e_1$", r"$e_1$"]
    yLabels = [r"$e_3$", r"$e_3$", r"$e_2$"]
    for i in range(3):
        axs[0, i].set_title(speeds[i])
        axs[0, i].set_xlabel(xLabels[i])
        axs[0, i].set_ylabel(yLabels[i])
        axs[1, i].set_xlabel(xLabels[i])
        axs[1, i].set_ylabel(yLabels[i])

    minima = [r"$\gamma = [0.125, 0.25, 0.25]$", r"$\gamma = [0.331, -0.073, -0.182]$"]
    for i in range(2):
        tempAx = axs[i, 2].twinx()
        tempAx.set_ylabel(minima[i])
        tempAx.set_yticks([])

    cbar = fig.colorbar(imComp, ax=axs)
    cbar.solids.set(alpha=0.2)

    fig.savefig(
        name + fileType, bbox_inches="tight", transparent=True, dpi=getDPI(fileType)
    )


##################################
### FIGURE 11: sampleInitConds ###
##################################
def plotSampleInitConds(
    pageFracWidth=0.85, aspectRatio=2.0, fileType=".png", name="sampleInitConds"
):
    randFigs = [9, 83, 122, 156, 170, 174]
    pot, xGrid, uInitsVal, _ = loadIvpRefs("data/ivpRefsOrig.npz")

    def pipeLine(ax, xGrid, uInit, pot, showAx2=False):
        xGrid = np.array(xGrid)
        uInit = np.array(uInit)

        ax.plot(xGrid, np.abs(uInit), "k", label="Mod")
        ax.plot(xGrid, np.real(uInit), "b", label="Real")
        ax.plot(xGrid, np.imag(uInit), "r", label="Imag")
        ax.grid(which="major", color="#CCCCCC", linewidth=1.0)
        ax.grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=0.7)

        ax2 = ax.twinx()
        ax2.set_yscale("log")
        if showAx2:
            ax2.set_ylabel("Potential")
        else:
            ax2.set_yticklabels([])

        ax2.plot(
            xGrid, pot - min(pot) + 1, color="yellow", alpha=0.5, label="Potential"
        )
        ax2.fill_between(xGrid, pot - min(pot) + 1, 0, color="yellow", alpha=0.1)

    # Set up fig
    plt.close("all")
    fig, axs = plt.subplots(
        2,
        3,
        figsize=(toFigSize(pageFracWidth, aspectRatio)),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    axs[1, 0].set_xlabel("Spatial Domain")
    axs[1, 1].set_xlabel("Spatial Domain")
    axs[1, 2].set_xlabel("Spatial Domain")
    axs[0, 0].set_ylabel("Magnitude of Sol")
    axs[1, 0].set_ylabel("Magnitude of Sol")

    pipeLine(axs[0, 0], xGrid, uInitsVal[0][randFigs[0]], pot)
    pipeLine(axs[0, 1], xGrid, uInitsVal[0][randFigs[1]], pot)
    pipeLine(axs[0, 2], xGrid, uInitsVal[0][randFigs[2]], pot, True)
    pipeLine(axs[1, 0], xGrid, uInitsVal[0][randFigs[3]], pot)
    pipeLine(axs[1, 1], xGrid, uInitsVal[0][randFigs[4]], pot)
    pipeLine(axs[1, 2], xGrid, uInitsVal[0][randFigs[5]], pot, True)
    handles, _ = axs[1, 1].get_legend_handles_labels()
    handle = Line2D([0], [0], color="yellow", alpha=0.5, label="Potential")
    handles.extend([handle])

    fig.legend(loc="center", ncols=4, handles=handles, bbox_to_anchor=(0.5, -0.05))

    fig.savefig(
        name + fileType, bbox_inches="tight", transparent=True, dpi=getDPI(fileType)
    )


#############################
### FIGURE 12: paramOptim ###
#############################
def plotParamOptim(
    pageFracWidth=0.85, aspectRatio=2.0, fileType=".png", name="paramOptim"
):
    allLearns = loadParamLossTrain()
    splitNames = getSplitNames()
    splitColours = getSplitColours()

    def pipeLine(axs, val, loss, p1, p2, p3, p4, p5, p6, color, label, short=False):
        handle = axs[0].plot(val, color=color, linestyle="-", label=label)
        axs[0].plot(loss, color=color, linestyle="--", alpha=0.2)

        axs[1].plot(p1, color=color, linestyle="-", alpha=0.75)
        axs[1].plot(p2, color=color, linestyle="-", alpha=0.75)
        axs[1].plot(p3, color=color, linestyle="-", alpha=0.75)

        if short:
            return handle

        axs[1].plot(p4, color=color, linestyle="-", alpha=0.75)
        axs[1].plot(p5, color=color, linestyle="-", alpha=0.75)
        axs[1].plot(p6, color=color, linestyle="-", alpha=0.75)

        return handle

    # Set up fig
    plt.close("all")
    fig, axs = plt.subplots(
        1, 2, figsize=(toFigSize(pageFracWidth, aspectRatio)), layout="constrained"
    )
    handles, _ = axs[0].get_legend_handles_labels()

    for i, allLearn in enumerate(allLearns):
        handle = pipeLine(
            axs,
            allLearn[0],
            allLearn[1],
            allLearn[2],
            allLearn[3],
            allLearn[4],
            allLearn[5],
            allLearn[6],
            allLearn[7],
            splitColours[i + 3],
            splitNames[i + 3],
            allLearn[8],
        )
        handles.extend(handle)

    axs[0].set_xlabel(r"Training iteration")
    axs[0].grid(which="major", color="#CCCCCC", linewidth=1.0)
    axs[0].grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=0.7)

    axs[1].set_xlabel(r"Training iteration")
    axs[1].grid(which="major", color="#CCCCCC", linewidth=1.0)
    axs[1].grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=0.7)

    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_ylabel(r"Loss")
    axs[1].set_xscale("log")
    axs[1].set_ylabel(r"$\gamma_k$")
    axs[0].legend(handles=handles, loc="best")

    fig.savefig(
        name + fileType, bbox_inches="tight", transparent=True, dpi=getDPI(fileType)
    )


############################
### FIGURE 13: allOptims ###
############################
def plotAllOptims(
    pageFracWidth=0.85, aspectRatio=2.0, fileType=".png", name="allOptims"
):
    datas = loadParamLossOptims()

    def pipeLine(axs, data):
        loss, x, y, z, color, label = data
        axs[0].plot(loss, color=color, linestyle="-")
        axs[1].plot(x, color=color, linestyle="-", label=label)
        axs[2].plot(y, color=color, linestyle="-")
        axs[3].plot(z, color=color, linestyle="-")

    # Set up fig
    plt.close("all")
    fig, axs = plt.subplots(
        1, 4, figsize=(toFigSize(pageFracWidth, aspectRatio)), layout="constrained"
    )

    for data in datas:
        pipeLine(axs, data)

    for ax in axs.reshape(-1):
        ax.set_xlabel(r"Train iter")
        ax.grid(which="major", color="#CCCCCC", linewidth=1.0)
        ax.grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=0.7)

    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_ylabel(r"Loss")
    axs[1].set_ylabel(r"$\gamma_1$")
    axs[2].set_ylabel(r"$\gamma_2$")
    axs[3].set_ylabel(r"$\gamma_3$")
    fig.legend(loc="center", ncols=4, bbox_to_anchor=(0.5, -0.15))

    fig.savefig(
        name + fileType, bbox_inches="tight", transparent=True, dpi=getDPI(fileType)
    )


#############################
### FIGURE R1: lossConv2D ###
#############################
def plotLossConv2D(
    pageFracWidth=0.85, aspectRatio=2.0, fileType=".png", name="lossConv2D"
):
    splitNames, losses, numExps, lowQs, highQs, _ = loadLossConv(
        "data/lossConv2D.npz", True
    )
    # minInds = [np.argmin(loss)+1 for loss in losses] # TODO: Check why the +1 does not overflow
    minInds = [len(loss) for loss in losses]  # Can set to all data

    # Set up fig
    plt.close("all")
    fig, ax = plt.subplots(
        1, 1, figsize=(toFigSize(pageFracWidth, aspectRatio)), layout="constrained"
    )
    ax.set_xlabel(r"number of exponentials")
    ax.set_ylabel(r"func $L_2$ norm")

    for i, splitCol in enumerate(getSplitColours(False, True)):
        plotTo = minInds[i]
        filtExps = numExps[i][:plotTo]
        ax.loglog(
            filtExps,
            losses[i][:plotTo],
            splitCol,
            linestyle="-",
            marker="",
            alpha=1.0,
            label=splitNames[i],
        )
        ax.fill_between(
            filtExps,
            lowQs[i][:plotTo],
            highQs[i][:plotTo],
            color=splitCol,
            alpha=0.2,
            linewidth=0,
        )

    ax.loglog(
        numExps[0][:5], 5 * [2], "black", linestyle="-", alpha=0.5, label=r"Unitarity"
    )
    orderLineWidth = np.array([500, 1000])
    ax.loglog(
        orderLineWidth,
        (2.00e-3 / (orderLineWidth[0] ** -1.0)) * np.power(orderLineWidth, -1.0),
        "black",
        linestyle=":",
        alpha=0.5,
        label=r"Order 1, 2, 4",
    )
    ax.loglog(
        orderLineWidth,
        (1e-3 / (orderLineWidth[0] ** -2.0)) * np.power(orderLineWidth, -2.0),
        "black",
        linestyle=":",
        alpha=0.5,
    )
    ax.loglog(
        orderLineWidth,
        (5e-4 / (orderLineWidth[0] ** -4.0)) * np.power(orderLineWidth, -4.0),
        "black",
        linestyle=":",
        alpha=0.5,
    )

    ax.legend(loc="best", ncols=1)
    ax.grid(which="major", color="#CCCCCC", linewidth=1.0)
    ax.grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=0.7)
    ax.set_xscale("log")
    ax.set_xlim(100, 2800)
    ax.set_ylim(1e-5, 4)
    fig.savefig(
        name + fileType, bbox_inches="tight", transparent=True, dpi=getDPI(fileType)
    )


#####################################
### FIGURE R2: trainDataSizeOptim ###
#####################################
def plotTrainDataSizeOptim(pageFracWidth=0.85, aspectRatio=2.0, fileType='.png', name='trainDataSizeOptim'):
    trainingSetSizes, trainL2Loss, valL2Loss, _ = loadTrainDataSizeCorr()
    numTrainSizes, numEpochs = trainL2Loss.shape
    maxTrainSizeCol = getSplitColours()[4]
    xRange = np.arange(numEpochs)+1
    alphas = np.linspace(0.3, 1.0, numTrainSizes)

    def pipeLine(axs, trainLoss, valLoss, trainSize, color, alpha, infDataThreshhold = 10**5):
        infSymb = r'$\infty$'
        handle = axs[0].plot(xRange, trainLoss, color=color, linestyle='-', alpha=alpha, label=f'{trainSize if trainSize < infDataThreshhold else infSymb}')
        axs[1].plot(xRange, valLoss, color=color, linestyle='-', alpha=alpha)
        return handle

    # Set up fig
    plt.close('all')
    fig, axs = plt.subplots(1, 2, figsize=(toFigSize(pageFracWidth, aspectRatio)), layout='constrained')
    handles, _ = axs[0].get_legend_handles_labels()

    for i in range(numTrainSizes):
        handle = pipeLine(axs, trainL2Loss[i], valL2Loss[i], trainingSetSizes[i], maxTrainSizeCol, alphas[i])
        handles.extend(handle)
        
    axs[0].set_xlabel(r'Training iteration')
    axs[0].grid(which='major', color='#CCCCCC', linewidth=1.0)
    axs[0].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
        
    axs[1].set_xlabel(r'Training iteration')
    axs[1].grid(which='major', color='#CCCCCC', linewidth=1.0)
    axs[1].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)

    axs[0].set_yscale('log')
    axs[0].set_ylabel(r'Train $L_2$ error')
    axs[1].set_yscale('log')
    axs[1].set_ylabel(r'Validation $L_2$ error')
    axs[0].legend(handles=handles, loc='best',ncol=2)
    for ax in axs:
        ax.set_ylim(1E-3,0.7)

    fig.savefig(name+fileType, bbox_inches='tight', transparent=True, dpi=getDPI(fileType))

###################################################################################################################################

##################
### CALL PLOTS ###
##################
plotParamTransform(0.85, 3.7, ".png", "paramTransform")
plotLossLandscape(0.85, 1.2, ".png", "lossLandscape")
plotLoss2dPlanes(0.85, 1.7, ".png", "loss2dPlanes")
plotSplitVisLearn(0.85, 2.8, ".png", "splitVisLearn")
plotLossConv(0.85, 2.0, ".png", "lossConv")
plotLossRelAdv(0.85, 2.4, ".png", "lossRelAdv")
plotLossConvGen(0.85, 2.0, ".png", "lossConvGen")
plotLossLandGen(0.85, 1.2, ".png", "lossLandGen")
plotBestFitCoefs(0.85, 2.4, ".png", "bestFitCoefs")
plotLoss2dPlanesGen(0.85, 1.7, ".png", "loss2dPlanesGen")
plotSampleInitConds(0.85, 2.0, ".png", "sampleInitConds")
plotParamOptim(0.85, 3.2, ".png", "paramOptim")
plotAllOptims(0.85, 4.0, ".png", "allOptims")
plotLossConv2D(0.55, 1.2, ".png", "lossConv2D")
plotTrainDataSizeOptim(0.9, 4.0, '.png', 'trainDataSizeOptim')