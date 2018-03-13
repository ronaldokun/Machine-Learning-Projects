import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys, argparse

np.set_printoptions(precision=4) # Print only four digits

'''
    Author: Samuel Prevost
    Date:   22/02/2018 14:22:53 UTC+1
    Title:  Automatic Linear Discriminant Analysis
    Desc:
        - Aim of LDA: project a feature space (a dataset n-dimensional samples) onto a smaller subspace k (k <= n-1)
            while maintaining the class-discriminatory information.
        - LDA requires knowing the classes of the samples
        - It can be good to first reduce the dimension using PCA, then project by class using LDA
    Main source: http://sebastianraschka.com/Articles/2014_python_lda.html
'''


def main(argv):
    ## ------- WELCOMING C.L.I. ------- ## 
    parser = argparse.ArgumentParser()
    parser.add_argument("inputFile", help="input file in Comma Separated Value format (with or without headers)")
    parser.add_argument("outputFile", help="will contain the data projected in the Linear Discriminants' dimensions, output as CSV")
    parser.add_argument("labelCol", help="Column number (counting from 0) containing the name of the class to which the sample belongs, should be a positive integer", type=int)
    parser.add_argument("-t", "--varThreshold", help="cumulative variance threshold after which to drop the useless eigen vectors, default: 0.8", type=float)
    parser.add_argument("-ev", "--explainedVar", help="output the explained variance along with the cumulative variance, should be a path where to save the image, if '-' the image will be shown but not save")
    parser.add_argument("-pd", "--projectedData", help="output the data projected in the first two Linear Discriminants' dimensions as a graph of scattered point, the most spread the better, should be a path where to save the image, if '-' the image will be shown but not save")
    parser.add_argument("-pm", "--projectionMat", help="path where to save the projection matrix used to project the data into PCs' dimensions (as as binary Numpy file .npy)")
    parser.add_argument("-dn", "--dropNa", help="exclude every row containing an null, invalid or infinite value. Solves the 'Input contains NaN, infinity or a value too large for dtype('float64')' issue", action="store_true")
    parser.add_argument("-v", "--verbose", help="enable verbose and show graph as they get generated while still saving them", action="store_true")
    args = parser.parse_args()

    ## ------- ARGUMENTS ------- ## 
    inputFile = args.inputFile
    outputFile = args.outputFile
    labelColIndex = np.abs(int(args.labelCol))
    varThreshold = 0.8 if not args.varThreshold else args.varThreshold
    explainedVarPath = args.explainedVar
    projectedDataPath = args.projectedData
    projectionMatPath = args.projectionMat
    dropNa = args.dropNa
    verbose = args.verbose

    ## ------- INPUTS ------- ##
    data = pd.read_csv(inputFile)
    if dropNa: # Drop every lines containing a NaN val, default: disabled
        data.dropna(inplace=True)
    # Drop empty lines at file-end
    data.dropna(how="all", inplace=True)
    # Array containing each row's label (as string)
    strLabelVect = data.ix[:,labelColIndex].values
    # Transform the labels to integer (starting from 1)
    enc = LabelEncoder()
    labelEncoder = enc.fit(strLabelVect)
    labelVect = labelEncoder.transform(strLabelVect) + 1
    
    # Ex : labelDict = {1: 'class 1', 2: 'class 2', 3: 'class 3' ... etc }
    labelDict = dict()
    for key, val in zip(labelVect, strLabelVect):
        if not key in labelDict:
            labelDict[key] = val
    if verbose:
        print("Identified classes: ", labelDict)

    # Only keep rows with numerical data (float or int)
    data.drop(data.columns[[labelColIndex]], axis=1, inplace=True) # Remove label's col
    numData = data.ix[:,:]._get_numeric_data()
    if verbose:
        print("Numerical data from input:\n", numData.head(), "\n...")
    # Convert from pandas dataframe to numpy array
    numData = numData.values
    ## COMPUTE D-DIMENSIONAL MEAN VECTOR ##
    # This vector contains the mean vector of each class
    # The mean vector is the mean of each feature of this class
    meanVects = []
    for label in set(labelVect):
        meanVects.append(np.mean(numData[labelVect==label], axis=0))
        if verbose:
            print("Mean vect class {}: \n{}".format(label, meanVects[label-1]))

    ## COMPUTE SCATTER MATRICES ##
    # -- Within-class scatter matrix (called sW)
    featureCount = numData.shape[1]
    sW = np.zeros((featureCount, featureCount))
    for label, meanVect in zip(set(labelVect), meanVects):
        classScatterMat = np.zeros_like(sW) # scatter matrix for class
        for row in numData[labelVect == label]:
            row = row.reshape(row.shape[0], 1)
            meanVect = meanVect.reshape(meanVect.shape[0], 1) # make col vect
            classScatterMat += (row-meanVect).dot((row-meanVect).T)
        sW += classScatterMat
    if verbose:
        print("Within-class scatter matrix: \n", sW)

    # -- Between-class scatter matrix (called sB)
    totalMean = np.mean(numData, axis=0)
    sB = np.zeros((featureCount, featureCount))
    for i, meanVect in enumerate(meanVects):
        n = numData[labelVect == i+1, :].shape[0]
        meanVect = np.array(meanVect) # avoid strange warning
        meanVect = meanVect.reshape(meanVect.shape[0], 1) # make col vect
        totalMean = np.array(totalMean) # avoid strange warning
        totalMean = totalMean.reshape(totalMean.shape[0], 1) # make col vect
        sB += n * (meanVect - totalMean).dot((meanVect - totalMean).T)

    if verbose:
        print("Between-class scatter matrix: \n", sB)

    ## CORE OF LDA ##
    eigVals, eigVects = np.linalg.eig(np.linalg.inv(sW).dot(sB))
    if verbose:
        for i in range(len(eigVals)):
            eigVect = eigVects[:,i].reshape(eigVects.shape[0], 1) # make col vect
            if eigVect.shape[0] < 7:
                print("Eigen vect {}: \n{}".format(i+1, eigVect.real))
            print("Eigen val {}: \n{:.2e}".format(i+1, eigVals[i].real))
            if i > 7:
                break
        # Checking if eigen vectors/values are alright
        print("Eigen Vects should be valid solution of sW^(-1)*sB*EigVect = EigVal*EigVect, checking...", end="\t")
        for i in range(len(eigVals)):
            eigVect = eigVects[:,i].reshape(eigVects.shape[0], 1) # make col vect
            np.testing.assert_array_almost_equal(np.linalg.inv(sW).dot(sB).dot(eigVect).real,
                                                (eigVals[i] * eigVect).real,
                                                decimal=6,
                                                err_msg="Strange, the eigen vectors/values are wrong ?! This often occurs when the values are more than e+10, since numpy checks for differences in decimals regardless of the scale",
                                                verbose=True)
        print("... Success !")

    ## ------- SORT EIGEN VECTS BY EIGEN VALS ------- ##
    # "The eigenvectors with the lowest eigenvalues bear the least information about the distribution of the data"
    # Let's drop 'em

    # List of (eigVal, eigVect) tuples
    eigPairs = [(np.abs(eigVals[i]), eigVects[:,i]) for i in range(len(eigVals))]
    # Sort it
    eigPairs.sort(key=lambda x: x[0], reverse=True)
    if verbose and len(eigPairs) < 8:
        print("List of eigen vals in descending order:", [i[0] for i in eigPairs])

    ## ------- EXPLAINED VARIANCE ------- ##
    eigSum = sum(eigVals)
    explnVar = [(i/eigSum)*100 for i in sorted(eigVals, reverse=True)]
    cumulativeExplnVar = np.cumsum(explnVar)
    if verbose:
        for i,j in enumerate(eigPairs):
            print("Eigen value {0:}: {1:.2%}".format(i+1, (j[0]/eigSum).real))
            if i > 7:
                break
    ## ------- GRAPH EXPLAINED VAR ------- ##
    # Graph of the explained variance compared to the cumulative
    if not explainedVarPath is None:
        with plt.style.context("seaborn-whitegrid"):
            plt.figure(figsize=(9, len(eigVals)))
            plt.bar(range(len(eigVals)), explnVar, alpha=0.5, align="center", label="individual explained variance")
            plt.step(range(len(eigVals)), cumulativeExplnVar, where="mid", label="cumulative explained variance")
            plt.ylabel("Explained variance ratio")
            plt.xlabel("Eigen vects")
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
    while sum([sortedEigVals[i]/eigSum for i in range(amountOfEigVectsToKeep)]) < varThreshold:
        amountOfEigVectsToKeep += 1
    if verbose:
        varConserved = sum([sortedEigVals[i]/eigSum for i in range(amountOfEigVectsToKeep)]).real
        print("Amount of eigen vectors to keep to keep >={0:.2%} of information is {1:}, keeping {2:.2%} variance".format(varThreshold, amountOfEigVectsToKeep, varConserved))
    
    # Create the projection matrix using the minimum amount of eigen vects to keep to reach the threshold
    eigVectsForLDA = []
    for i in range(amountOfEigVectsToKeep):
        eigVectLen = len(eigPairs[i][1])
        # Tilt the eig vects on their side to get vectors and not lists
        eigVectsForLDA.append(eigPairs[i][1].reshape(eigVectLen, 1))
    # Combine each eig vects horizontally to get a numberOfInputFeatures x amountOfEigVectsToKeep projection matrix
    # in which the data dimension is optimally reduced
    matW = np.hstack(tuple(eigVectsForLDA)).real
    if verbose:
        print("Projection Matrix (matrix W):\n", matW)
    
    ## ------- SAVE PROJECTION MATRIX ------- ##
    if not projectionMatPath is None:
        print("Projection Matrix (matrix W) saved under: {} in Numpy binary format (.npy)".format(projectionMatPath))
        np.save(projectionMatPath, matW)

    ## PROJECT DATA ONTO NEW SUBSPACE ##
    Y = numData.dot(matW).real
    print("Input dimensions: {0:}\nOutput dimensions: {1:}\nReduction: {2:.2%}".format(numData.shape[1], Y.shape[1], 1-Y.shape[1]/numData.shape[1]))
    assert Y.shape == (numData.shape[0], amountOfEigVectsToKeep), "The matrix is not {}x{} dimensional !!".format(numData.shape[0], amountOfEigVectsToKeep)
    ## ------- GRAPH PROJECTION DATA ------- ##
    # Show a representation in 2D
    if not projectedDataPath is None and Y.shape[1] >= 2:
        with plt.style.context("seaborn-whitegrid"):
            plt.figure(figsize=(40, 20))
            ax = plt.subplot(111)
            # Generate unique label colour
            dicoLabelColor = dict()
            listColors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for i in range(max(labelVect)):
                dicoLabelColor[i+1] = listColors[(i+1)%len(listColors)]

            Y_round = np.around(Y, decimals=2)
            for lab in set(labelVect):
                # Col 0 is first PC, col 1 is second PC
                x = Y_round[lab==labelVect, 0]
                y = Y_round[lab==labelVect, 1]
                plt.scatter(x, y, alpha=0.5, label=labelDict[lab], c=dicoLabelColor[lab], marker=".", s=500)
            plt.xlabel("LD 1")
            plt.ylabel("LD 2")
            legend = plt.legend(loc="upper right", fancybox=True)
            legend.get_frame().set_alpha(0.5)
            plt.title("LDA: {} projection onto the first 2 linear discriminant".format(inputFile))
            # hide axis ticks
            plt.tick_params(axis='both', which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
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
                print("Projection in 2D using the new LD axis saved under : {}".format(projectedDataPath))
                plt.savefig(projectedDataPath)

    columns = ["LD{}".format(i+1) for i in range(Y.shape[1])]
    columns.append("class")
    labelCol = np.array([labelDict[label] for label in labelVect]).reshape(labelVect.shape[0], 1)
    Y = np.append(Y, labelCol, axis=1)
    Y = pd.DataFrame(Y, columns=columns)
    Y.to_csv(outputFile, sep=",", encoding="utf-8", index=False)
    print("Projected data saved under: {}".format(outputFile))
if __name__ == "__main__":
    main(sys.argv[1:])