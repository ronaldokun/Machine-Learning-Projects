import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import sys
import argparse

# np.set_printoptions(precision=4) # Print only four digits

'''
    Author: Samuel Prevost
    Date:   21/02/2018 22:33:13 UTC+1
    Title:  Automatic Principal Component Analysis
    Desc:   This script automatically performs PCA on any CSV dataset and outputs
            interesting graphs as well as the projected data as a new and more compact dataset (dimension reduction)

    Main source: http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html
'''


def main(argv):
    ## ------- WELCOMING C.L.I. ------- ##
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputFile", help="input file in Comma Separated Value format (with or without headers)")
    parser.add_argument(
        "outputFile", help="will contain the data projected in the Principal Components' dimensions, output as CSV")
    parser.add_argument("-t", "--varThreshold",
                        help="cumulative variance threshold after which to drop the useless principal components vectors, default: 0.8", type=float)
    parser.add_argument("-ev", "--explainedVar",
                        help="output the explained variance along with the cumulative variance, should be a path where to save the image, if '-' the image will be shown but not save")
    parser.add_argument("-pd", "--projectedData", help="output the data projected in the first two Principal Components' dimensions as a graph of scattered point, the most spread the better, should be a path where to save the image, if '-' the image will be shown but not save")
    parser.add_argument("-lc", "--labelCol", help="Column number (counting from 0) containing the name of the class to which the sample belongs, add colours to graph of projected data if provided, should be a positive integer", type=int)
    parser.add_argument("-pm", "--projectionMat",
                        help="path where to save the projection matrix used to project the data into PCs' dimensions (as as binary Numpy file .npy)")
    parser.add_argument("-dn", "--dropNa", help="exclude every row containing an null, invalid or infinite value. Solves the 'Input contains NaN, infinity or a value too large for dtype('float64')' issue", action="store_true")
    parser.add_argument(
        "-v", "--verbose", help="enable verbose and show graph as they get generated while still saving them", action="store_true")
    args = parser.parse_args()

    ## ------- ARGUMENTS ------- ##
    inputFile = args.inputFile
    outputFile = args.outputFile
    varThreshold = 0.8 if not args.varThreshold else args.varThreshold
    explainedVarPath = args.explainedVar
    projectedDataPath = args.projectedData
    labelColIndex = np.abs(
        int(args.labelCol)) if not args.labelCol is None else None
    projectionMatPath = args.projectionMat
    dropNa = args.dropNa
    verbose = args.verbose

    ## ------- INPUTS ------- ##
    data = pd.read_csv(inputFile)
    if dropNa:  # Drop every lines containing a NaN val, default: disabled
        data.dropna(inplace=True)
    # Drop empty lines at file-end
    data.dropna(how="all", inplace=True)
    labelVect = None
    if not labelColIndex is None:
        labelVect = data.ix[:, labelColIndex].values
        data.drop(data.columns[[labelColIndex]], axis=1,
                  inplace=True)  # Remove label's col
    # Only keep rows with numerical data (float or int)
    numData = data.ix[:, :]._get_numeric_data()
    if verbose:
        print("Numerical data from input:\n", numData.head(), "\n...")
    # Standardise to mean = 0 var = 1
    numData = StandardScaler().fit_transform(numData)

    ## ------- CALC COV MAT ------- ##
    #meanVect = np.mean(numData, axis=0)
    #covMat = (numData - meanVect).T.dot((numData - meanVect))/(numData.shape[0]-1)
    # ^^^^ Same as doing vvvvv
    covMat = np.cov(numData.T)
    if verbose:
        print("Cov mat:\n", covMat)

    ## ------- CALC EIGEN VALS AND VECT ------- ##
    eigVals, eigVects = np.linalg.eig(covMat)  # <==== CORE OF P.C.A.
    if verbose:
        print("Eigen Vals:\n", eigVals)
        print("Eigen Vects\n", eigVects)

    # P.S.: Correlation Matrix == normalized covariance matrix

    ## ------- TESTING EIGEN VECTS ------- ##
    if verbose:
        print("Eigen Vects should have a magnitude of 1, checking...", end="\t")
        for eigVect in eigVects:
            np.testing.assert_array_almost_equal(1.0, np.linalg.norm(eigVect))
        print("... Success !")

    ## ------- SORT EIGEN VECTS BY EIGEN VALS ------- ##
    # "The eigenvectors with the lowest eigenvalues bear the least information about the distribution of the data"
    # Let's drop 'em

    # List of (eigVal, eigVect) tuples
    eigPairs = [(np.abs(eigVals[i]), eigVects[:, i])
                for i in range(len(eigVals))]
    # Sort it
    eigPairs.sort(key=lambda x: x[0], reverse=True)
    if verbose:
        print("List of eigen vals in descending order:",
              [i[0] for i in eigPairs])

    ## ------- EXPLAINED VARIANCE ------- ##
    eigSum = sum(eigVals)
    explnVar = [(i / eigSum) * 100 for i in sorted(eigVals, reverse=True)]
    cumulativeExplnVar = np.cumsum(explnVar)

    ## ------- GRAPH EXPLAINED VAR ------- ##
    # Graph of the explained variance compared to the cumulative
    if not explainedVarPath is None:
        with plt.style.context("seaborn-whitegrid"):
            plt.figure(figsize=(9, len(eigVals)))
            plt.bar(range(len(eigVals)), explnVar, alpha=0.5,
                    align="center", label="individual explained variance")
            plt.step(range(len(eigVals)), cumulativeExplnVar,
                     where="mid", label="cumulative explained variance")
            plt.ylabel("Explained variance ratio")
            plt.xlabel("Principal components")
            plt.legend(loc="best")
            plt.tight_layout()
            if verbose or explainedVarPath == "-":
                plt.show()
            if explainedVarPath != "-":
                print("Explained Variance saved under : {}".format(explainedVarPath))
                plt.savefig(explainedVarPath)

    ## ------- PROJECTION MATRIX ------- ##
    amountOfEigVectsToKeep = 0
    sortedEigVals = sorted(eigVals, reverse=True)
    while sum([sortedEigVals[i] / eigSum for i in range(amountOfEigVectsToKeep)]) < varThreshold:
        amountOfEigVectsToKeep += 1
    if verbose:
        varConserved = sum([sortedEigVals[i] / eigSum *
                            100 for i in range(amountOfEigVectsToKeep)])
        print("Amount of Principal Components to keep to keep >={}% of information is {}, keeping {}% variance".format(
            varThreshold * 100, amountOfEigVectsToKeep, varConserved))

    # Create the projection matrix using the minimum amount of eigen vects to keep to reach the threshold
    eigVectsForPCA = []
    for i in range(amountOfEigVectsToKeep):
        eigVectLen = len(eigPairs[i][1])
        # Tilt the eig vects on their side to get vectors and not lists
        eigVectsForPCA.append(eigPairs[i][1].reshape((eigVectLen, 1)))
    # Combine each eig vects horizontally to get a numberOfInputFeatures x amountOfEigVectsToKeep projection matrix
    # in which the data dimension is optimally reduced
    matW = np.hstack(tuple(eigVectsForPCA))
    if verbose:
        print("Projection Matrix (matrix W):\n", matW)

    ## ------- SAVE PROJECTION MATRIX ------- ##
    if not projectionMatPath is None:
        print("Projection Matrix (matrix W) saved under: {} in Numpy binary format (.npy)".format(
            projectionMatPath))
        np.save(projectionMatPath, matW)

    ## ------- PERFORM PCA ------- ##
    # Perform projection in the new PCA projected space
    Y = numData.dot(matW)
    print("Input dimensions: {0:}\nOutput dimensions: {1:}\nReduction: {2:.2%}".format(
        numData.shape[1], Y.shape[1], 1 - Y.shape[1] / numData.shape[1]))
    columns = ["PC{}".format(i + 1) for i in range(Y.shape[1])]
    if not labelVect is None:
        columns.append("class")
        labelCol = labelVect.reshape(labelVect.shape[0], 1)
        Y = np.append(Y, labelCol, axis=1)
    Y = pd.DataFrame(Y, columns=columns)
    Y.to_csv(outputFile, sep=",", encoding="utf-8", index=False)
    #np.savetxt(outputFile, Y, delimiter=",")
    Y = Y.values
    print("Projected data saved under: {}".format(outputFile))

    ## ------- GRAPH PROJECTION DATA ------- ##
    # Show a representation in 2D
    if not projectedDataPath is None and Y.shape[1] >= 2:
        with plt.style.context("seaborn-whitegrid"):
            plt.figure(figsize=(40, 20))
            ax = plt.subplot(111)
            # Generate unique label colour
            dicoLabelColor = dict()
            if not labelVect is None:
                listColors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                uniqueLabels = list(set(labelVect))
                for i, uniqueLabel in enumerate(uniqueLabels):
                    dicoLabelColor[uniqueLabel] = listColors[i %
                                                             len(listColors)]

                for lab in tuple(uniqueLabels):
                    # Col 0 is first PC, col 1 is second PC
                    plt.scatter(Y[labelVect == lab, 0], Y[labelVect == lab, 1],
                                alpha=0.5, label=lab, c=dicoLabelColor[lab], marker=".", s=500)
            else:
                for i in range(Y.shape[0]):
                    plt.scatter(Y[i, 0], Y[i, 1], c='b')
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            legend = plt.legend(loc="upper right", fancybox=True)
            legend.get_frame().set_alpha(0.5)
            plt.title(
                "PCA: {} projection onto the first 2 principal components".format(inputFile))
            # hide axis ticks
            plt.tick_params(axis='both', which="both", bottom="off", top="off",
                            labelbottom="on", left="off", right="off", labelleft="on")
            # remove axis spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            plt.grid()
            plt.tight_layout()
            if verbose or projectedDataPath == "-":
                plt.show()
            if projectedDataPath != "-":
                print("Projection in 2D using the new PC axis saved under : {}".format(
                    projectedDataPath))
                plt.savefig(projectedDataPath)


if __name__ == "__main__":
    main(sys.argv[1:])