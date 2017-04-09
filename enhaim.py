# ...:::;;;||| ENHAIM v1.0  #
# Gustavo Santiago, 8937416 #
import cv2
import numpy as np
from matplotlib import pyplot as plt

# DEFINE #
# ---------------------------------------------------------------------------- #
VERBOSE = 1
# ---------------------------------------------------------------------------- #

# TOOLS #
# ---------------------------------------------------------------------------- #
# Verify if f and g have the same dimensions
def checkDimensions(f, g):
    height, width = src.shape[:2]
    tmpY, tmpX = src.shape[:2]
    # If the dimensions deviate, return 0
    if(width != tmpX or height != tmpY):
        return 0
    # Else, return 1
    else:
        return 1

# Histogram calculus
def imHistogram(f):
    hst = [0] * 256
    height, width = src.shape[:2]

    for i in range(0, height):
        for j in range(0, width):
            hst[f[i][j]] = hst[f[i][j]] + 1

    hst = np.array(hst)
    return hst

# Histogram cummulative
def imCummulativeHist(hst):
    cmh = [0] * 256
    cmh[0] = hst[0]

    for i in range(1, 256):
        cmh[i] = cmh[i-1] + hst[i]

    cmh = np.array(cmh)
    return cmh

# Histogram plot (output: "histogram.jpg")
def plotHistogram(hst, name):
    plt.title("Histogram: " + name)
    plt.plot(np.arange(0, 256), hst, label = name)
    plt.legend()
    plt.savefig("hst_" + name.lower() + ".jpg")
    plt.gcf().clear()

# Histogram show
def showHistogram(hst, name):
    plotHistogram(hst, name)
    hstimg = cv2.imread("hst_" + name.lower() + ".jpg")
    cv2.imshow("Histogram: " + name, hstimg)

# RMSD calculus function
def rmsd(f, g):
    if(checkDimensions(f, g) == 0): return
    height, width = f.shape[:2]

    nm = height * width

    out = 0
    # Sum of (f[x,y] - g[x,y]) ^ 2 / n * m
    for i in range(0, height):
        for j in range(0, width):
            diff = np.float32(f[i][j])
            diff = diff - np.float32(g[i][j])
            out = out + np.float32(diff * diff / nm)

    out = np.sqrt(out) # Final RMSD root

    return out
# ---------------------------------------------------------------------------- #

# FILTERS #
# ---------------------------------------------------------------------------- #
# Logarithmic enhancement
def imLog(f):
    height, width = src.shape[:2]
    out = np.zeros((height, width, 1), np.float32)

    c = np.float32(255 / np.log(1 + f.max()))

    for i in range(0, height):
        for j in range(0, width):
            out[i][j] = np.float32(c * np.log(1 + np.absolute(f[i][j])))
    out = cv2.convertScaleAbs(out)

    return out

# Gamma correction
def imGamma(f, gamma):
    height, width = src.shape[:2]
    out = np.zeros((height, width, 1), np.float32)

    for i in range(0, height):
        for j in range(0, width):
            out[i][j] = np.float32(np.power(f[i][j], gamma))
    out = cv2.convertScaleAbs(out)

    return out

# Histogram equalization
def imEqualHist(f):
    height, width = src.shape[:2]
    out = np.zeros((height, width, 1), np.float32)

    hst = imHistogram(f)
    cmh = imCummulativeHist(hst)

    tmp = np.float32(255 / (height * width))

    for i in range(0, height):
        for j in range(0, width):
            out[i][j] = np.float32(tmp * cmh[f[i][j]])
    out = cv2.convertScaleAbs(out)

    return out

# Sharpening filter
def imSharp(f, a, b):
    height, width = src.shape[:2]
    ker = np.matrix([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]])
    np.flip(ker, -1)

    cnv = np.zeros((height, width, 1), np.float32)
    out = np.zeros((height, width, 1), np.float32)

    cnv = cv2.filter2D(f, -1, ker)
    cnv = np.float32(cnv)

    for i in range(0, height):
        for j in range(0, width):
            diff = np.float32(cnv[i][j] - f[i][j])
            out[i][j] = np.float32(a * f[i][j]) + np.float32(b * diff)
    out = cv2.convertScaleAbs(out)

    return out
# ---------------------------------------------------------------------------- #

# Main operations handler
def execute(src, gamma, a, b, show):
    height, width = src.shape[:2]
    if(VERBOSE):
        print("Image dimensions: X=" + str(width) + ", Y=" + str(height))

    if(show == 1):
        cv2.imshow("Source", src)

        hst = imHistogram(src)
        showHistogram(hst, "Original")
        # Plot along with original histogram
        plt.plot(np.arange(0, 256), hst, label = "Original")
        cmh = imCummulativeHist(hst)
        # Normalize CMH
        cmh = cmh * (np.max(hst) / np.max(cmh))
        showHistogram(cmh, "Cummulative")

        print("RMSD")

        # LOG ENHANCEMENT #
        # -------------------------------------------------------------------- #
        log = imLog(src)
        print("L=%.4f" % rmsd(src, log))
        log = np.clip(log, 0, 255).astype("uint8")
        # Plot along with original histogram
        plt.plot(np.arange(0, 256), hst, label = "Original")
        showHistogram(imHistogram(log), "Log Enhancement")
        cv2.imshow("Logarithmic Enhancement", log)
        # -------------------------------------------------------------------- #

        # GAMMA CORRECTION #
        # -------------------------------------------------------------------- #
        gmma = imGamma(src, gamma)
        print("G=%.4f" % rmsd(src, gmma))
        gmma = np.clip(gmma, 0, 255).astype("uint8")
        # Plot along with original histogram
        plt.plot(np.arange(0, 256), hst, label = "Original")
        showHistogram(imHistogram(gmma), "Gamma Correction")
        cv2.imshow("Gamma Correction", gmma)
        # -------------------------------------------------------------------- #

        # HISTOGRAM EQ #
        # -------------------------------------------------------------------- #
        eqh = imEqualHist(src)
        print("H=%.4f" % rmsd(src, eqh))
        eqh = np.clip(eqh, 0, 255).astype("uint8")
        # Plot along with original histogram
        plt.plot(np.arange(0, 256), hst, label = "Original")
        showHistogram(imHistogram(eqh), "Histogram Equalization")
        cv2.imshow("Histogram Equalization", eqh)
        # -------------------------------------------------------------------- #

        # SHARPENING FILTER #
        # -------------------------------------------------------------------- #
        sha = imSharp(src, a, b)
        print("S=%.4f" % rmsd(src, sha))
        sha = np.clip(sha, 0, 255).astype("uint8")
        # Plot along with original histogram
        plt.plot(np.arange(0, 256), hst, label = "Original")
        showHistogram(imHistogram(sha), "Sharpening Filter")
        cv2.imshow("Sharpening Filter", sha)
        # -------------------------------------------------------------------- #

        cv2.waitKey()
        cv2.destroyAllWindows()

# MAIN #
# ---------------------------------------------------------------------------- #
if(VERBOSE): print("...:::;;;||| ENHAIM v1.0")
if(VERBOSE): fname = input("Type in the file name: ")
else: fname = input()

src = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
if src is None:
    if(VERBOSE): print("[ERROR]: File \"" + fname + "\" does not exist.")
else:
    if(VERBOSE): gamma = input("Type in the gamma value: ")
    else: gamma = input()
    if(VERBOSE): a = input("Type in the A value: ")
    else: a = input()
    if(VERBOSE): b = input("Type in the B value: ")
    else: b = input()
    if(VERBOSE): show = input("Show image windows? (0 or 1): ")
    else: show = input()

    gamma = float(gamma)
    a = float(a)
    b = float(b)
    show = int(show)
    execute(src, gamma, a, b, show)
# ---------------------------------------------------------------------------- #
