import numpy as np
import MRFGeneratePotentials
import MRFBeliefPropagation
from MRFBeliefPropagation import createBPalg
from scipy.sparse import csr_matrix
from scipy.sparse import tril
from scipy.io import loadmat
import logging
import os,sys
sys.path.append('../../')
import myio

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


'''
function list = rearrange(seeds,nn)
% function to return a list of nodes ordered according to the
% degree of separation from the nodes in the seeds array
%
% INPUT: nn is an Nx(K+1) matrix, where N is # nodes and K # neighbors
%        seeds is vector with the seed node indeces
%
% OUTPUT: list is the list of nodes ordered according to the degree of
% separation from the seed nodes.
%
% E.g., seeds = (6, 15)                        1  2  3  4
%        nn = (1, 2, 5                         5  6  7  8
%              2, 1, 3, 6                      9 10 11 12
%              3, 2,4, 7                      13 14 15 16
%              4, 3, 8
%              ...)
%
%        then list = (6,15,2,5,7,10,11,14,16,1,3,9,8,7,12,15,13,4)
% initialize list
%
% Hstau Liao
% Columbia University
% Created: Feb 24,2018. Modified:Feb 24,2018
%
'''


def rearrange(seeds,nn):
    nodelist = []
    for i in xrange(len(seeds)):
        nodelist.append(seeds[i])

    cur_nodes = seeds
    #seeds
    next_nodes = [] # % one degree of sepration from cur_nodes

    #%while length(list) < size(nn,1)
    for szlist in xrange(nn.shape[0]):
        for j in xrange(len(cur_nodes)):
            #%
            j_neigh = nn[cur_nodes[j],:]
            j_neigh = j_neigh[j_neigh!=-100]  #% remove -100 as neighbors, indexing starts with 0, so a negative number was used in nn for fillers.

            for k in xrange(len(j_neigh)):
                probe = j_neigh[k]
                if probe not in nodelist: #%&& probe > 0
                   nodelist.append(probe)
                   next_nodes.append(probe)



        cur_nodes = next_nodes
        next_nodes = []

    #% add the remaining nodes which were not visited, to the final list
    remnodes = set(range(nn.shape[0]))-set(nodelist)
    #print 'Nodes not visited:',list(remnodes) # isolated nodes not visited
    if len(remnodes)>0:
        nodelist = nodelist + list(remnodes)

    #print 'nodelist:',nodelist
    nodelist=np.array(nodelist)

    return nodelist




'''
function [nodeOrder,G] = createNodeOrder(G,anchorNodes,nodeOrderType)
% Generate a ordering of node numbers for visiting the nodes
% Input:
%   G: Input graph / adjacency
%   anchorNodes: The anchor nodes
%   nodeOrderType: Node order type
%       'default' :Sequential order 1...nNodes
%       'minSpan' :Minimum spanning tree order with 'BFS' or 'DFS' search
%       'multiAnchor': The nodes arrangement will be start from the anchors
%        and then progressively with the neihbors of the anchor nodes
% Output:
%   nodeOrder: Reordered node numbers
%
%
% Suvrajit Maji,sm4073@cumc.columbia.edu
% Columbia University
% Created: Feb 28,2018. Modified: Mar 01,2018
%
%
Copyright (c) Columbia University Suvrajit Maji 2018 (original matlab version an debugging)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)

'''
def createNodeOrder(G,anchorNodes,nodeOrderType):

    print 'nodeOrderType:',nodeOrderType

    if not nodeOrderType or nodeOrderType==[]:
        #% default sequential order
        nodeOrderType ='default'

    A = G['AdjMat']
    nodeOrder = np.empty(G['nNodes'])

    if nodeOrderType=='default':
        nodeOrder = np.array(range(A.shape[0]))

    elif nodeOrderType=='multiAnchor':
        nnMatCell = G['nnMat']
        nnMatCell = np.reshape(nnMatCell,(G['nNodes'],-1))
        #print 'nnMatCell',nnMatCell

        #Sz = cell2mat(cellfun(@(x) size(x,2),nnMatCell,'UniformOutput',False))
        Sz = np.apply_along_axis(lambda x: len(x[0]),1, nnMatCell)
        maxSz = max(Sz)
        #print 'Sz',Sz,maxSz
        #nnMat = cell2mat(cellfun(@(x) [x zeros(1,maxSz - size(x,2))],nnMatCell,'UniformOutput',false));
        nnMat = np.apply_along_axis(lambda x: np.append(x[0], -100*(np.ones((1,maxSz-len(x[0]))))).tolist(),1,nnMatCell).astype(int) # put -100 as filler since indexing starts with 0
        #print nnMat
        nodeOrder = rearrange(anchorNodes,nnMat)

    return nodeOrder




'''
function [psinums,senses] = getPsiSensesfromNodeLabels(nodeState,NumPsis)
% variables for final psi-nums and senses
%psinums = zeros(2,length(PrDs));
%senses = zeros(2,length(PrDs));
Copyright (c) Columbia University Suvrajit Maji 2018
(original matlab version and python debugging)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
'''

def getPsiSensesfromNodeLabels(nodeState,NumPsis):

    nodePsiLabels = nodeState
    psinums = (nodePsiLabels % NumPsis)
    psinums[psinums==0] = NumPsis
    senses = (nodePsiLabels <= NumPsis) + (nodePsiLabels > NumPsis)*(-1)

    return (psinums,senses)


def readBadNodesPsisTau(badNodesPsisTaufile):
    try:
        dataR = myio.fin1(badNodesPsisTaufile)
        badNodesPsisTau = dataR['badNodesPsisTau']
    except:
        badNodesPsisTau = np.empty((0,0))

    return badNodesPsisTau

'''
function [nodeStateBP,psinums_cc,senses_cc,OptNodeBel] = runGlobalOptimization (G,BPoptions,edgeMeasures,cc)
Copyright (c) Columbia University Suvrajit Maji 2018
(original matlab version and python debugging)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Copyright (c) Columbia University Hstau Liao 2018 (python version)
'''

def op(G,BPoptions,edgeMeasures,edgeMeasures_tblock,badNodesPsis,cc,*argv):

    '''    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Generate the node and edge potentials for the Markov Random Field
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    import p
    NumPsis = p.num_psis
    #always make sure that the number of states maxState = 2*nPsiModes, because
    #we have two levels: up and down state for each psiMode.
    maxState = 2*NumPsis
    G.update(nPsiModes=p.num_psis)# update the nPsiModes in case p.num_psis is changed in the later steps
    G.update(maxState=maxState)

    if cc == 1:
        #print p.anch_list
        # format: PrD,CC1,S1 for 1D
        # p.anch_list = np.array([[1,1,1],[2,1,-1]])  #TEMP should PrD and CC1 start from 1 or 0?
        p.anch_list = np.array(p.anch_list)
        #print "type of anchor list", type(p.anch_list)
        #print "anchors= ", p.anch_list

        IndStatePlusOne = p.anch_list[:,2]==1
        IndStateMinusOne = p.anch_list[:,2]==-1
        anchorNodesPlusOne = p.anch_list[IndStatePlusOne,0]-1
        anchorNodesMinusOne = p.anch_list[IndStateMinusOne,0]-1
        anchorStatePlusOne = p.anch_list[IndStatePlusOne,1]-1
        anchorStateMinusOne = p.anch_list[IndStateMinusOne,1] + NumPsis-1

    elif cc == 2:
        if argv:
            nodeStateBP_cc1=argv[0]

        # format: PrD,CC1,S1,CC2,S2 for 2D
        IndStatePlusOne = p.anch_list[:, 4] == 1
        IndStateMinusOne = p.anch_list[:, 4] == -1
        anchorNodesPlusOne = p.anch_list[IndStatePlusOne, 0]-1
        anchorNodesMinusOne = p.anch_list[IndStateMinusOne, 0]-1
        anchorStatePlusOne = p.anch_list[IndStatePlusOne, 3]-1
        anchorStateMinusOne = p.anch_list[IndStateMinusOne, 3] + NumPsis-1

    #anchorNodes = [anchorNodesPlusOne, anchorNodesMinusOne]

    anchorNodeMeasuresPlusOne = np.zeros((maxState,len(anchorNodesPlusOne)))
    anchorNodeMeasuresMinusOne = np.zeros((maxState,len(anchorNodesMinusOne)))

    anchorNodes = anchorNodesPlusOne.tolist() + anchorNodesMinusOne.tolist()
    print 'anchorNodes:',anchorNodes

    anchorNodePotValexp = 110

    for u in xrange(len(anchorNodesPlusOne)):
        anchorNodeMeasuresPlusOne[anchorStatePlusOne[u],u] = anchorNodePotValexp

    for v in xrange(len(anchorNodesMinusOne)):
        anchorNodeMeasuresMinusOne[anchorStateMinusOne[v],v] = anchorNodePotValexp

    anchorNodeMeasures = np.hstack((anchorNodeMeasuresPlusOne,anchorNodeMeasuresMinusOne))
    nodePot,edgePot = MRFGeneratePotentials.op(G,anchorNodes,anchorNodeMeasures,edgeMeasures,edgeMeasures_tblock)

    # Set potential value to zero for bad psi-movies
    badNodesPsisTaufile = '{}badNodesPsisTauFile'.format(p.CC_dir)
    badNodesPsisTau  = readBadNodesPsisTau(badNodesPsisTaufile)
    #print 'bp-badNodesPsisTau',badNodesPsisTau[0:10,:]

    # from bad taus and split block movies
    if (badNodesPsis.shape[0]== badNodesPsisTau.shape[0]) and (badNodesPsis.shape[1]== badNodesPsisTau.shape[1]):
        badNodesPsis2 =  badNodesPsis + badNodesPsisTau
    else:
        badNodesPsis2 =  badNodesPsis

    # if badNodesPsis exists
    if badNodesPsis2.shape[0]>0:
        #print 'bp-badNodesPsis2',badNodesPsis2[0:10,:]
        for n in xrange(badNodesPsis2.shape[0]): # row has prd numbers, column has psi number so shape is (num_prds,2)
            #remember that badNodePsis has index starting with 1 ??
            badPsis = np.nonzero(badNodesPsis2[n,:]<=-100)[0]
            #print 'n',n,'badPsis',badPsis
            for k in badPsis:
                if k < NumPsis:
                    nodePot[k,n] = 1e-16
                    nodePot[k+NumPsis,n] = 1e-16
                else:
                    nodePot[k,n] =1e-16
                    nodePot[k-NumPsis,n] = 1e-16

            #print 'nodePot',nodePot[:,n]

    if cc==2:
        for n in xrange(nodeStateBP_cc1.shape[0]): # as nodeStateBP_cc1 is row vector so shape is (num_prds,)
            k = nodeStateBP_cc1[n] - 1 # remember that nodeStateBP_cc1 has index starting with 1 from previous run
            if k < NumPsis:
                nodePot[k,n] = 1e-16
                nodePot[k+NumPsis,n] = 1e-16
            else:
                nodePot[k,n] = 1e-16
                nodePot[k-NumPsis,n] = 1e-16

    print 'nodePot.shape:',nodePot.shape,'edgePot.shape',edgePot.shape

    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Global optimization with Belief propagation
    % Local pairwise measures for the projection direction/psi Topos and movies 
    % are encoded in the undirected probabilistic graphical model as Markov Random Field(MRF) 

    % options for belief propagation iterations
    % options: options for belief propagation iterations
    % options.tol
    % options.verbose
    % options.maximize
    % options.maxIter
    % options.eqnStates 
    '''


    options = dict(maxProduct = BPoptions['maxProduct'], verbose = BPoptions['verbose'], tol = BPoptions['tol'],
                   maxIter = BPoptions['maxIter'],eqnStates = BPoptions['eqnStates'],
                   alphaDamp = BPoptions['alphaDamp'])



    #;%.98;%0.99; %0.99 use damping factor (< 1) when message oscillates and do not converge

    # %%%%% For debug with samll example
    if options['maxProduct']:
        bplbl = 'maxprod'
    else:
        bplbl = 'sumprod'

    #BPalg['anchorNodes'] = anchorNodes
    G['anchorNodes'] = anchorNodes

    #nodeOrderType = 'default' # sequential order
    nodeOrderType = 'multiAnchor'

    graphNodeOrder = createNodeOrder(G,anchorNodes,nodeOrderType)
    G['graphNodeOrder'] = graphNodeOrder
    #print 'graphNodeOrder:',graphNodeOrder

    BPalg = createBPalg(G,options)
    BPalg['anchorNodes'] = anchorNodes

    nodeBelief,edgeBelief,BPalg = MRFBeliefPropagation.op(BPalg,nodePot,edgePot)
    #print 'nodeBelief:\n',nodeBelief
    OptNodeLabels = np.argsort(-nodeBelief,axis=0)
    nodeStateBP = OptNodeLabels[0,:] # %max-marginal
    OptNodeBel = nodeBelief[nodeStateBP,xrange(0,len(nodeStateBP))]

    #print 'nodeBelief',nodeBelief
    #print 'OptNodeLabels',OptNodeLabels
    #print  'OptNodeBel',OptNodeBel
    #%%%%% Determine the Psi's and Senses %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    print '\nDetermining the psinum and senses from node labels ...'
    nodeStateBP = nodeStateBP + 1 # indexing from 1 as matlab
    #print 'nodeStateBP:',nodeStateBP
    psinumsBP,sensesBP = getPsiSensesfromNodeLabels(nodeStateBP,NumPsis)

    psinums_cc = np.zeros((1,G['nNodes']),dtype='int')
    senses_cc = np.zeros((1,G['nNodes']),dtype='int')

    noAnchorCC = G['ConnCompNoAnchor']
    #print 'noAnchoccc',noAnchorCC
    nodesEmptyMeas=[]
    for c in noAnchorCC:
        #print 'c',c,'Gcc[c]',G['NodesConnComp'][c]
        nodesEmptyMeas.append(G['NodesConnComp'][c])

    nodesEmptyMeas = [y for x in nodesEmptyMeas for y in x]
    nodesEmptyMeas = np.array(nodesEmptyMeas)
    print 'nodesEmptyMeas:',nodesEmptyMeas

    psinums_cc[:] = psinumsBP-1 #python starts with 0
    senses_cc[:] = sensesBP

    psinums_cc = psinums_cc.flatten()
    senses_cc = senses_cc.flatten()

    if len(nodesEmptyMeas)>0:
        # put psinum/senses value to -1, for the nodes 'nodesEmpty' for which there were no calcuations done.
        psinums_cc[nodesEmptyMeas]=-1
        senses_cc[nodesEmptyMeas]=0

    # %%%% compare with the manual labels
    compareLabelAcc = 0
    if compareLabelAcc:

        '''
        from matlab
        recoFile = '../../Results/dataS2/ReCo.mat' # change your path for manual labels
        recoData = loadmat(recoFile)
        psiNumsAll = recoData['psiNumsAll']
        sensesAll = recoData['sensesAll']
        '''

        labelfile = '../../outputs_testvATPase/CC/temp_anchors_20190805-210129.txt'
        recoFileLabel = np.loadtxt(labelfile)

        nodes = recoFileLabel[:,0].astype(int)
        psinums = recoFileLabel[:,1].astype(int)
        senses = recoFileLabel[:,2].astype(int)

        psiNumsAll = (-1)*np.ones((G['nNodes']))
        sensesAll = np.zeros((G['nNodes']))

        #for n in range(0,psiNumsAll.shape[0]):
        #    if i==
        psiNumsAll[nodes-1] = psinums-1
        sensesAll[nodes-1]= senses



        Acc = sum(((psiNumsAll[:,cc-1] - psinumsBP.T)==0)*((sensesAll[:,cc-1] - sensesBP.T)==0))/\
              float(psiNumsAll.shape[0]-sum(psiNumsAll[:,cc-1]==-1))
        print '\nAccuracy: {}'.format(Acc)

        correctPsiSensePrds = np.nonzero(((psiNumsAll[:,cc-1] - psinumsBP.T)==0)*((sensesAll[:,cc-1] - sensesBP.T)==0))
        diffPsiPrds = np.setdiff1d(range(0,psiNumsAll.shape[0]),correctPsiSensePrds)

        np.savetxt('diffPsiPrds.txt', diffPsiPrds, fmt='%i\t', delimiter='\t')
        np.savetxt('correctPsiSensePrds.txt',correctPsiSensePrds, fmt='%i\t', delimiter='\t')




    return (nodeStateBP,psinums_cc,senses_cc,OptNodeBel,nodeBelief)
