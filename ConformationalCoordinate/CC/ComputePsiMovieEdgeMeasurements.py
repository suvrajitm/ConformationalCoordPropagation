import numpy as np
import logging, sys

sys.path.append('../')

import myio
import p
import datetime
import ComputeOpticalFlowPrDAll
import ComputeMeasureEdgeAll
from subprocess import call
import os
import time

#from pyface.qt import QtGui, QtCore
#os.environ['ETS_TOOLKIT'] = 'qt4'

# this rescaling function should ensure to keep the exp(-M) values within a certain range as to prevent
# numerical overflow/underflow
def reScaleLinear(M,edgeNumRange,mvalrange):
    numE = np.max(edgeNumRange)#M.shape[0]
    nm = np.zeros(numE+1).astype(int)
    all_m=[]
    #print 'edgeNumRange',edgeNumRange
    for e in edgeNumRange:
        #print 'scale e',e
        meas = M[e].flatten()
        #print meas.shape
        nm[e] = meas.shape[0]
        all_m.append(meas)

    all_m = np.squeeze(all_m).flatten()

    #print 'min',np.min(all_m),'max',np.max(all_m)

    # determine if there are outliers in the all_m array
    #mean_val = np.mean(all_m)
    #sd_val = np.std(all_m)
    #upper_thresh = mean_val+3*sd_val

    q1, q3 = np.percentile(all_m,[25,75])
    iqr = q3 - q1
    upper_thresh =  q3 + (1.5 * iqr)

    #print 'upper_thresh',upper_thresh
    #outlier_ids = all_m>upper_thresh
    #all_m[outlier_ids]= upper_thresh
    #print 'max-without outliers',np.max(all_m[all_m<=upper_thresh])
    all_m[all_m>upper_thresh]= upper_thresh
    #print all_m
    ## linear scaling of values within the range 'mvalr', min and max to mapped to min(mvalr) and max(mvalr)
    scaled_all_m = np.interp(all_m,(np.min(all_m), np.max(all_m)),mvalrange)

    M_scaled = np.empty(M.shape,dtype=object)
    #print  M_scaled.shape
    for e in edgeNumRange:
        if e==0:
            nm_ind = range(0,nm[e])
        else:
            nm_ind = range(e*nm[e-1],e*nm[e-1]+nm[e])

        M_scaled[e]= np.reshape(scaled_all_m[nm_ind],np.shape(M[0]))

        #print 'e',e
        #print 'meas:',M_scaled[e]
    return M_scaled



def op(G, nodeRange, edgeNumRange, *argv):

    nodeEdgeNumRange = [nodeRange,edgeNumRange]

    # Step 1. Compute Optical Flow Vectors
    # Save the optical flow vectors for each psi-movie of individual projection direction
    if p.getOpticalFlow:
        print ('\n1.Now computing optical flow vectors for all (selected) PrDs...\n')
        #Optical flow vectors for each psi-movies of each node are saved to disk
        if argv:

            ComputeOpticalFlowPrDAll.op(nodeEdgeNumRange,argv[0])
        else:
            ComputeOpticalFlowPrDAll.op(nodeEdgeNumRange)

    # Step 2. Compute the pairwise edge measurements
    # Save individual edge measurements
    if p.getAllEdgeMeasures:
        print ('\n2.Now computing pairwise edge-measurements...\n')
        # measures for creating potentials later on
        # edgeMeasures files for each edge (pair of nodes) are saved to disk
        if argv:
            ComputeMeasureEdgeAll.op(G,nodeEdgeNumRange,argv[0])
        else:
            ComputeMeasureEdgeAll.op(G,nodeEdgeNumRange)

    # Step 3. Extract the pairwise edge measurements
    # to be used for node-potential and edge-potential calculations
    print ('\n3.Reading all the edge measurements from disk...')
    # load the measurements file for each edge separately
    #edgeMeasures = np.empty((len(edgeNumRange)),dtype=object)

    #in case there are some nodes/edges for which we do not want to calculate the measures,the number of edges and
    # max edge indices may not match, so use the full G.nEdges as the size of the edgeMeasures. The edges which are not
    # calculated will remain as empty
    edgeMeasures = np.empty((G['nEdges']),dtype=object)
    edgeMeasures_tblock = np.empty((G['nEdges']),dtype=object)
    badNodesPsisBlock = np.zeros((G['nNodes'],p.num_psis))
    for e in edgeNumRange:
        currPrD = G['Edges'][e,0]
        nbrPrD =  G['Edges'][e,1]
        #print 'Reading Edge:',e,'currPrD:',currPrD,'nbrPrD:',nbrPrD
        CC_meas_file = '{}{}_{}_{}'.format(p.CC_meas_file,e,currPrD,nbrPrD)
        data = myio.fin1(CC_meas_file)
        measureOFCurrNbrEdge = data['measureOFCurrNbrEdge']
        measureOFCurrNbrEdge_tblock = data['measureOFCurrNbrEdge_tblock']
        bpsi = data['badNodesPsisBlock']
        badNodesPsisBlock = badNodesPsisBlock + bpsi
        edgeMeasures[e] = measureOFCurrNbrEdge
        edgeMeasures_tblock[e] = measureOFCurrNbrEdge_tblock


    ###Test 30aug2019
    #print 'before:',edgeMeasures[0][:,:]
    #print 'before:',edgeMeasures[2][:,:]
    scaleRange = [5.0,30.0]
    edgeMeasures = reScaleLinear(edgeMeasures,edgeNumRange,scaleRange)
    #print 'badNodesPsisBlock',badNodesPsisBlock[0:10,:]
    #print 'after rescale:',edgeMeasures[0][:,:]
    #print 'after rescale:',edgeMeasures[2][:,:]
    #print 'badNodesPsis',badNodesPsis
    #print 'edgeMeasures_tblock',type(edgeMeasures_tblock)
    return edgeMeasures,edgeMeasures_tblock,badNodesPsisBlock
