bool TernaryScheme<T>::checkFlipReduce(int i, int j, int index1, int index2, int sign) {
    int cmpI = uvw[i][index1].compare(uvw[i][index2]);
    if (cmpI == sign && uvw[j][index1].limitSum(uvw[j][index2], j != 2)) {
        reduceAdd(j, index1, index2);
        return true;
    }

    if (cmpI == -sign && uvw[j][index1].limitSub(uvw[j][index2], false)) {
        if (j == 2 || uvw[j][index1].positiveFirstNonZeroSub(uvw[j][index2]))
            reduceSub(j, index1, index2);
        else
            reduceSub(j, index2, index1);
        return true;
    }

    int cmpJ = uvw[j][index1].compare(uvw[j][index2]);
    if (cmpJ == sign && uvw[i][index1].limitSum(uvw[i][index2], i != 2)) {
        reduceAdd(i, index1, index2);
        return true;
    }

    if (cmpJ == -sign && uvw[i][index1].limitSub(uvw[i][index2], false)) {
        if (i == 2 || uvw[i][index1].positiveFirstNonZeroSub(uvw[i][index2]))
            reduceSub(i, index1, index2);
        else
            reduceSub(i, index2, index1);
        return true;
    }

    return false;
}