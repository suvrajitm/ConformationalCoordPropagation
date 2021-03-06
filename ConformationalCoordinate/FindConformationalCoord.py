import numpy as np
import logging, sys
from subprocess import call
sys.path.append('./CC/')

import myio
import random
import ComputePsiMovieEdgeMeasurements
import datetime
import runGlobalOptimization
import FindCCGraphPruned
#_logger = logging.getLogger(__name__)
#_logger.setLevel(logging.DEBUG)
import os,os.path
import set_params
import time
import q2Spider
import rotatefill
import math
#from pyface.qt import QtGui, QtCore
#os.environ['ETS_TOOLKIT'] = 'qt4'


''' Suvrajit Maji,sm4073@cumc.columbia.edu
    Columbia University
    Created: Dec 2017. Modified:Aug 16,2019
'''

'''
def psi_ang(PD):

    lPD = sum(PD**2)
    Qr = np.array([1 + PD[2], PD[1], -PD[0], 0])
    Qr = Qr / np.sqrt(np.sum(Qr**2))
    phi,theta,psi = q2Spider.op(Qr)

    psi = np.mod(psi,2*np.pi)*(180/np.pi)
    return psi

def rotate_psi(PrDs,psinums):

    for prD in PrDs:
        dist_file = '{}prD_{}'.format(p.dist_file,prD)
        data = myio.fin1(dist_file)
        PD = data['PD']
        psi = psi_ang(PD)
        for psinum in psinums:
            imgPsiFileName = '{}prD_{}_psi_{}'.format(p.psi2_file,prD,psinum)
            data_IMG = myio.fin1(imgPsiFileName)
            img = data_IMG["IMG1"].T #transpose or not
            img = rotatefill.op(img, -psi, visual=False) # plus or minus

'''
def op(*argv):
    import p
    time.sleep(5)
    set_params.op(1)

    # file i/o
    CC_file = '{}/CC_file'.format(p.CC_dir)
    if os.path.exists(CC_file):
        os.remove(CC_file)

    nodeOutputFile = '{}/comp_psi_sense_nodes.txt'.format(p.CC_dir)
    if os.path.exists(nodeOutputFile):
        os.remove(nodeOutputFile)

    nodeBelFile1 = '{}/nodeAllStateBel_rc1.txt'.format(p.CC_dir)
    if os.path.exists(nodeBelFile1):
        os.remove(nodeBelFile1)

    nodeBelFile2 = '{}/nodeAllStateBel_rc2.txt'.format(p.CC_dir)
    if os.path.exists(nodeBelFile2):
        os.remove(nodeBelFile2)

    CC_graph_file_pruned = '{}_pruned'. format(p.CC_graph_file)
    if os.path.exists(CC_graph_file_pruned):
        os.remove(CC_graph_file_pruned)

    #if OF_CC, OF_CC_fig directory doesn't exist then
    if not os.path.exists(p.CC_OF_dir):
      call(["mkdir", "-p", p.CC_OF_dir])


    trash_list = np.array(p.trash_list)
    #print 'p.trash_list',trash_list
    trash_list_PDs = np.nonzero(trash_list==1)[0]
    t=0
    for i in p.trash_list:
        if int(i)==int(1):
            t=t + 1
    if t>0:
        print 'Number of trash PDs',trash_list_PDs.shape[0]
        CC_graph_file = CC_graph_file_pruned
        G,Gsub = FindCCGraphPruned.op(CC_graph_file)

    else:
        CC_graph_file = p.CC_graph_file
        data = myio.fin1(CC_graph_file)
        G=data['G']
        Gsub=data['Gsub']

    numConnComp = len(G['NodesConnComp'])
    #print "Number of connected component:",numConnComp

    anchorlist = [a[0] for a in p.anch_list]
    anchorlist = [a - 1 for a in anchorlist] # we need labels with 0 index to compare with the node labels in G, Gsub

    print '\nAnchor list:',anchorlist,', Number of anchor nodes:',len(anchorlist),'\n'

    if any(a in anchorlist for a in G['Nodes']):
        if len(anchorlist)+t==G['nNodes']:
            print '\nAll nodes have been manually selected (as anchor nodes). ' \
                 'Reaction-coordinate propagation is not required. Exiting this program.\n'


            psinums = np.zeros((2,G['nNodes']),dtype='int')
            senses = np.zeros((2,G['nNodes']),dtype='int')
            #CC1
            #psinums[0,:]=[a[1]-1 for a in p.anch_list]
            #senses[0,:]=[a[2] for a in p.anch_list]
	    
            for a in p.anch_list: #for row in anch_list; e.g. [1, 1, -1, 0]
                psinums[0,a[0]-1] = a[1]-1
                senses[0,a[0]-1] = a[2]


            idx = 0
            for t in p.trash_list: #for row in trash_list; e.g. [36, True] means PD 36 is Trash		

                if t == 1:
                    psinums[0,idx] = -1
                    senses[0,idx] = 0

                idx+=1


            #print 'psinums',psinums
            #print 'senses',senses
           		
            #CC2
            #psinums[1,:]=[a[4] for a in p.anch_list] # a[col], whatever column number is in the p.anch_list for CC2
            #senses[2,:]=[a[5] for a in p.anch_list]

            print '\nFind CC: Writing the output to disk...\n'
            p.CC_file = '{}/CC_file'.format(p.CC_dir)
            myio.fout1(p.CC_file, ['psinums', 'senses'],[psinums,senses])

            p.allAnchorPassed=1

            if argv:
                progress5 = argv[0]
                # We introduce some time delay for proper communication between the execution of this code and the GUI
                for i in xrange(101):
                    time.sleep(0.01)
                    progress5.emit(int(i))

            return
    else:
        print 'Some(or all) of the anchor nodes are NOT in the Graph node list.'
        return


    nodelCsel = []
    edgelCsel = []
    # this list keeps track of the connected component (single nodes included) for which no anchor was provided
    connCompNoAnchor = []
    for i in xrange(numConnComp):
        nodesGsubi = Gsub[i]['originalNodes']
        edgelistGsubi = Gsub[i]['originalEdgeList']
        edgesGsubi = Gsub[i]['originalEdges']
        #print 'Checking connected component ','i=',i,', Gsub[i]',', Original node list:',nodesGsubi,\
        #  'Original edge list:',edgelistGsubi[0],'Original edges:', edgesGsubi, 'No. edges:',len(edgesGsubi)

        #print 'Checking connected component ','i=',i,', Gsub[i]',', Original node list:',nodesGsubi,\
        #    'Original edge list:',edgelistGsubi[0],'Size Edges:',len(edgesGsubi)

        if any(x in anchorlist for x in nodesGsubi):
            #print 'Atleast one anchor node in connected component',i,'is selected.\n'
            nodelCsel.append(nodesGsubi.tolist())
            edgelCsel.append(edgelistGsubi[0])
        else:
            connCompNoAnchor.append(i)
            #print 'Anchor node(s) in connected component',i,' NOT selected.'
            #print '\nIf you proceed without atleast one anchor node for the connected component',i,\
            #    ', all the corresponding nodes will not be assigned with reaction coordinate labels.\n'

    G.update(ConnCompNoAnchor=connCompNoAnchor)
    #print connCompNoAnchor,G['ConnCompNoAnchor']

    #nodeRange = nodelCsel
    #edgeNumRange = edgelCsel
    nodeRange = [y for x in nodelCsel for y in x] #flatten list another way?
    edgeNumRange = [y for x in edgelCsel for y in x] #flatten list another way?

    #nodeRange = range(G['nNodes'])
    #edgeNumRange = range(G['nEdges'])
    #nodeRange = [171,172,173,174,175,176]
    #edgeNumRange = [1247,1248,1262,1267]
    #nodeRange = [0,1,2]
    #edgeNumRange = [0,1]
    nodeRange = np.sort(nodeRange)
    edgeNumRange= np.sort(edgeNumRange)
    # adding these two params to the CC graph file *Hstau Aug 19
    data = myio.fin1(CC_graph_file)
    extra = dict(nodeRange=nodeRange,edgeNumRange=edgeNumRange)
    data.update(extra)
    myio.fout2(CC_graph_file,data)
    #print 'Selected graph G nodes and edges in the connected components for which atleast one anchor node were selected:'
    #print 'nodeRange',nodeRange
    #print 'edgeNumRange',edgeNumRange
    # compute all pairwise edge measurements
    # Step 1: compute the optical flow vectors for all prds
    # Step 2: compute the pairwise edge measurements for all psi - psi movies
    # Step 3: Extract the pairwise edge measurements to be used for node-potential and edge-potential calculations
    if argv:
        edgeMeasures,edgeMeasures_tblock,badNodesPsisBlock = ComputePsiMovieEdgeMeasurements.op(G, nodeRange, edgeNumRange, argv[0])
    else:
        edgeMeasures,edgeMeasures_tblock,badNodesPsisBlock = ComputePsiMovieEdgeMeasurements.op(G, nodeRange, edgeNumRange)
    #print '\nedgeMeasures:\n',edgeMeasures

    # Setup and run the Optimization: Belief propagation
    print '\n4.Running Global optimization to estimate state probability of all nodes ...'

    BPoptions = dict(maxProduct = 1, verbose = 0, tol = 1e-4, maxIter = 300,eqnStates = 1.0, alphaDamp = 1.0)


    #reaction coordinate number rc = 1,2
    psinums = np.zeros((2,G['nNodes']),dtype='int')
    senses = np.zeros((2,G['nNodes']),dtype='int')
    cc = 1
    print ('\nFinding CC for Dim:1')
    nodeStateBP_cc1,psinums_cc1,senses_cc1,OptNodeBel_cc1,nodeBelief_cc1 = runGlobalOptimization.op(G, BPoptions, edgeMeasures,edgeMeasures_tblock,badNodesPsisBlock,cc)
    psinums[0, :]= psinums_cc1
    senses[0, :] = senses_cc1

    if p.dim == 2:
        cc = 2
        print ('\nFinding CC for Dim:2')
        nodeStateBP_cc2,psinums_cc2,senses_cc2,OptNodeBel_cc2,nodeBelief_cc2 = runGlobalOptimization.op(G,BPoptions,edgeMeasures,edgeMeasures_tblock,cc,nodeStateBP_cc1)
        psinums[1, :]= psinums_cc2
        senses[1, :] = senses_cc2

    # save
    print '\nFind CC: Writing the output to disk...\n'
    p.CC_file = '{}/CC_file'.format(p.CC_dir)
    myio.fout1(p.CC_file, ['psinums', 'senses'],[psinums,senses])


    if p.dim == 1:# 1 dimension
        node_list = np.empty((G['nNodes'],4))
        node_list[:,0] = range(1,G['nNodes']+1) # 1 indexing
        node_list[:,1] = psinums[0,:]+1 # indexing 1
        node_list[:,2] = senses[0,:]
        node_list[:,3] = OptNodeBel_cc1

        # save the found psinum , senses also as text file
        #node_list is variable name with columns: if dim =1 : (PrD, CC1, S1) + (CC2, S2) if dim =2
        np.savetxt(nodeOutputFile, node_list, fmt='%i\t%i\t%i\t%f', delimiter='\t')

        nodeBels1 = np.empty((nodeBelief_cc1.T.shape[0],nodeBelief_cc1.T.shape[1]+1))
        nodeBels1[:,0] = range(1,G['nNodes']+1)
        nodeBels1[:,1:] = nodeBelief_cc1.T
        np.savetxt(nodeBelFile1,nodeBels1, fmt='%f', delimiter='\t')

    elif p.dim == 2: # 2 dimension
        node_list = np.empty((G['nNodes'],7))
        node_list[:,0] = range(1,G['nNodes']+1) # 1 indexing
        node_list[:,1:3] = (psinums+1).T # indexing 1 and indices 1:3 means 1 & 2
        node_list[:,3:5] = senses.T #indices 3:5 means 3 & 4
        node_list[:,5] = OptNodeBel_cc1
        node_list[:,6] = OptNodeBel_cc2

        # save the found psinum , senses also as text file
        #node_list is variable name with columns: if dim =1 : (PrD, CC1, S1) + (CC2, S2) if dim =2
        np.savetxt(nodeOutputFile, node_list, fmt='%i\t%i\t%i\t%i\t%i\t%f\t%f', delimiter='\t')

        nodeBels2 = np.empty((nodeBelief_cc2.T.shape[0],nodeBelief_cc2.T.shape[1]+1))
        nodeBels2[:,0] = range(1,G['nNodes']+1)
        nodeBels2[:,1:] = nodeBelief_cc2.T
        np.savetxt(nodeBelFile2,nodeBels2, fmt='%f', delimiter='\t')


    if argv:
        progress5 = argv[0]
        progress5.emit(int(100))
    time.sleep(0.05)

if __name__ == '__main__':
    import p, os, sys
    p.init()
    p.user_dir = '../'

    sys.path.append('../')
    #p.proj_name='testData'
    p.proj_name='testvATPase'
    p.num_psis=1
    p.out_dir = os.path.join(p.user_dir, 'outputs_{}/'.format(p.proj_name))
    p.tess_file = '{}/selecGCs'.format(p.out_dir)
    p.nowTime_file = os.path.join(p.out_dir, '/nowTime')
    p.create_dir()
    #p.anch_list = [[1,1,1]]#p.anch_list
    p.anch_list = [[19,1,1],[28,1,-1],[108,1,-1],[126,1,-1],[206,1,-1],[217,2,-1],[289,2,-1],[300,1,-1],[312,1,-1],[322,1,-1]]#p.anch_list
    sys.path.append('../')
    op()
