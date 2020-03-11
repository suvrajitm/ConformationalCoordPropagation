import numpy as np
import logging, sys
import myio
import datetime
import multiprocessing
from functools import partial
from contextlib import contextmanager
from subprocess import Popen
import operator
import os
import time
import OpticalFlowMovie
from OpticalFlowMovie import getOrientMag
import LoadPrDPsiMoviesMasked
sys.path.append('../')
import mrcfile
import set_params
import p
import copy

#from pyface.qt import QtGui, QtCore
#os.environ['ETS_TOOLKIT'] = 'qt4'

sys.path.append('../')

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


@contextmanager

def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

'''
# this only works when PrD are in sequence
# here N is a number (total nodes)
def divide(N):
    ll = []
    for prD in xrange(N):
        #CC_OF_file = '{}/prD_{}'.format(p.CC_OF_dir, prD)
	    CC_OF_file = '{}{}'.format(p.CC_OF_file, prD)
        ll.append([CC_OF_file,prD])
    return ll
'''

# changed Nov 30, 2018, S.M.
# here N is a list of (node) numbers
def divide1(R):
    ll = []
    for prD in R:
        CC_OF_file = '{}{}'.format(p.CC_OF_file, prD)
        if os.path.exists(CC_OF_file):
            data = myio.fin1(CC_OF_file)
            if data is not None:
                continue
        ll.append([CC_OF_file,prD])
    return ll


def count1(R):
    c = 0
    for prD in R:
        CC_OF_file = '{}{}'.format(p.CC_OF_file, prD)
        if os.path.exists(CC_OF_file):
            data = myio.fin1(CC_OF_file)
            if data is not None:
                continue
        c+=1
    return c

'''
function ComputeOpticalFlowPrDAll
% Suvrajit Maji,sm4073@cumc.columbia.edu
% Columbia University
% Created: May 2018. Modified:Aug 16,2019
'''


def stackDicts(a, b, op=operator.concat):
    #op=lambda x,y: np.dstack((x,y),axis=2)
    op=lambda x,y: np.dstack((x,y))
    mergeDict = dict(a.items() + b.items() + [(k, op(a[k], b[k])) for k in set(b) & set(a)])
    return mergeDict


def ComputePsiMovieOpticalFlow(Mov,opt_movie,prds_psinums):

    OFvisualPrint = [opt_movie['OFvisual'],opt_movie['printFig']]
    Labels = ['FWD','REV']

    computeOF = 1
    blockSize_avg = 5 #how many frames will used for normal averaging
    currPrD = prds_psinums[0]
    psinum_currPrD = prds_psinums[1]
    prd_psinum = [currPrD, psinum_currPrD]

    #print '\nprd_psinum',prd_psinum
    MFWD = copy.deepcopy(Mov) #FWD
    #MREV = Mov[::-1,:] #REV
    numFrames = Mov.shape[0]
    #overlapFrames =  np.ceil(0.40*numFrames).astype(int)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Compute the Optical Flow vectors for each movie
    # For complicated motion involving some rotation component, the 2d movie can be misleading if we get the
    # optical flow vector added over the entire movie , so we might split the movie into blocks and compare
    # the vectors separtely , with splitmovie = 1, this is experimental now.

    # at present the stacking of the dictionary for two blocks has been checked, for others needs to be verified
    splitmovie = 0
    FlowVecFWD = []
    FlowVecREV = []
    if computeOF:

        if splitmovie:
            # number of frames in each blocks
            #blockSize_split = 20#25#np.ceil(numFrames/2).astype(int)+1 # except for the last block, remaining blocks will be of this size
            numBlocks_split = 3
            overlapFrames = 0 #12

            blockSize_split = np.round(np.float(numFrames + (numBlocks_split-1)*overlapFrames + 1)/(numBlocks_split)).astype(int)
            #numBlocks_split = np.round(np.float(numFrames - blockSize_split  + 1)/(blockSize_split - overlapFrames)).astype(int) + 1 + 1

            # In case we fix blockSize, it should be noted that the numBlocks will be different for different
            # blocksize and overlap values
            # Also, one extra block is used in case there is 1 or 2 frames left over after frameEnd is close to
            # numFrames and a new block is created with overlapping frames till frameEnd = numFrames
            # TO DO:better handling of this splitting into overlapping blocks
            for b in range(0,numBlocks_split):
                #frameStart = max(0,b*overlapFrames)
                #frameEnd = min(b*overlapFrames + blockSize_split - 1,numFrames)
                frameStart = max(0,b*(blockSize_split-overlapFrames))
                frameEnd = min(b*(blockSize_split-overlapFrames) + blockSize_split - 1,numFrames)

                if numFrames-frameEnd<5:
                    frameEnd = numFrames
                # check this criteria
                if  frameEnd - frameStart > 0:
                    #print 'frameStart,frameEnd', frameStart,frameEnd
                    blockMovieFWD = MFWD[frameStart:frameEnd,:]
                    #blockMovieREV = MREV[frameStart:frameEnd,:]

                    #print 'Computing Optical Flow for Movie Forward ...'
                    FlowVecFblock = OpticalFlowMovie.op(blockMovieFWD,prd_psinum,blockSize_avg,Labels[0]+'-H'+ str(b),OFvisualPrint)
                    if b==0:
                        FlowVecFWD = copy.deepcopy(FlowVecFblock)
                    else:
                        FlowVecFWD = stackDicts(FlowVecFWD,FlowVecFblock)

                    # print 'Computing Optical Flow for Movie Reverse ...'
                    # blockMovieFWD is used but due to label of 'REV', the negative vectors will be used after computing the FWD vectors
                    # If FWD vectors are provided, then reverse flow vectors are not going to be recomputed but will
                    # be obtained by reversing the FWD vectors (-Vx,-Vy)
                    # use FlowVecFblock as it is just one block, FlowVecFWD for multiple blocks has multidimensional Vx and Vy--stacked
                    FlowVecRblock = OpticalFlowMovie.op(blockMovieFWD,prd_psinum,blockSize_avg,Labels[1]+'-H'+ str(b),OFvisualPrint,FlowVecFblock)

                    if b==0:
                        FlowVecREV = copy.deepcopy(FlowVecRblock)
                    else:
                        FlowVecREV = stackDicts(FlowVecREV,FlowVecRblock)

                if frameEnd==numFrames:
                    break

        else:
            FlowVecFWD = OpticalFlowMovie.op(MFWD,prd_psinum,blockSize_avg,Labels[0],OFvisualPrint)

            #FlowVecREV = OpticalFlowMovie.op(MREV,prd_psinum,blockSize_avg,Labels[1],OFvisualPrint)
            # MFWD is used but due to label of 'REV', the negative vectors will be used after getting the FWD vectors
            FlowVecREV = OpticalFlowMovie.op(MFWD,prd_psinum,blockSize_avg,Labels[1],OFvisualPrint,FlowVecFWD)


    else:
        print('')
        #print 'Using the previously computed Optical Flow vectors for Movie A ...'
        #print 'Using the previously computed Optical Flow vectors for Movie B ...'

    #print 'FlowVecFWD.shape',np.shape(FlowVecFWD['Vx']),FlowVecFWD['Vx']
    FlowVec = dict(FWD=FlowVecFWD,REV=FlowVecREV)

    return FlowVec


def ComputeOptFlowPrDPsiAll1(input_data):
    CC_OF_file = input_data[0]
    currPrD = input_data[1]
    FlowVecPrD = np.empty(p.num_psis,dtype=object)
    psiSelcurrPrD = range(p.num_psis)

    #print 'currPrD',currPrD
    #load movie and tau param first
    moviePrDPsi,badPsis = LoadPrDPsiMoviesMasked.op(currPrD)

    #print 'curr PD',currPrD
    badPsis = np.array(badPsis)
    #print 'badPsis',badPsis,len(badPsis)
    if badPsis.shape[0]>0:
    #print 'badPsis for prD',currPrD,badPsis
        badNodesPsisTaufile = '{}badNodesPsisTauFile'.format(p.CC_dir)
        try:
            time.sleep(5)
            dataR = myio.fin1(badNodesPsisTaufile)
            badNodesPsisTau = dataR['badNodesPsisTau']
            #print  'read badNodesPsisTau', badNodesPsisTau,len(badPsis)
            badNodesPsisTau[currPrD,np.array(badPsis)] = -100
            #print 'write badNodesPsisTau',badNodesPsisTau[currPrD,:]
            myio.fout1(badNodesPsisTaufile, ['badNodesPsisTau'], [badNodesPsisTau])
            time.sleep(5)
        except:
            print('badNodes File: ',badNodesPsisTaufile,', does not exist.')


    #calculate OF for each psi-movie
    #loop over for psi selections for current prD
    for psinum_currPrD in psiSelcurrPrD:
        IMGcurrPrD = moviePrDPsi[psinum_currPrD]

        #print 'Current-PrD:{}, Current-PrD-Psi:{}'.format(currPrD, psinum_currPrD)
        prds_psinums = [currPrD, psinum_currPrD]
        FlowVecPrDPsi = ComputePsiMovieOpticalFlow(IMGcurrPrD,p.opt_movie,prds_psinums)
        FlowVecPrD[psinum_currPrD] =  FlowVecPrDPsi

    #print 'Writing OpticalFlow-Node {} data to file\n\n'.format(currPrD)
    CC_OF_file = '{}'.format(CC_OF_file)
    myio.fout1(CC_OF_file,['FlowVecPrD'],[FlowVecPrD])
    #return FlowVecPrD

# If computing for a specified set of nodes, then call the function with nodeRange
def op(nodeEdgeNumRange, *argv):
    time.sleep(5)
    set_params.op(1)

    '''
    #set_params.op(-1)
    if not os.path.exists(p.CC_OF_dir):
        from subprocess import call
        call(["mkdir", "-p", p.CC_OF_dir])
    '''
    if argv:
        progress5 = argv[0]

    nodeRange = nodeEdgeNumRange[0]
    edgeNumRange = nodeEdgeNumRange[1]
    numberofJobs = len(nodeRange) + len(edgeNumRange)

    findBadPsiTau = 1
    if findBadPsiTau:
        #initialize and write to file badpsis array
        #print '\nInitialize and write a file to record badPsis'
        offset_OF_files = len(nodeRange) - count1(nodeRange)
        #print 'offset_OF_files',offset_OF_files
        if offset_OF_files==0:
            badNodesPsisTaufile = '{}badNodesPsisTauFile'.format(p.CC_dir)
            if os.path.exists(badNodesPsisTaufile):
                os.remove(badNodesPsisTaufile)

        CC_graph_file_pruned = '{}_pruned'.format(p.CC_graph_file)
        if os.path.exists(CC_graph_file_pruned):
            dataG = myio.fin1(CC_graph_file_pruned)
        else:
            dataG = myio.fin1(p.CC_graph_file)

        if offset_OF_files==0:
            G = dataG['G']
            badNodesPsisTau = np.zeros((G['nNodes'],p.num_psis)).astype(int)
            myio.fout1(badNodesPsisTaufile, ['badNodesPsisTau'], [badNodesPsisTau])

    if p.machinefile:
        print('using MPI with {} processes'.format(p.ncpu))
        Popen(["mpirun", "-n", str(p.ncpu), "-machinefile", str(p.machinefile),
              "python", "modules/CC/ComputeOpticalFlowPrDAll_mpi.py"],close_fds=True)
        if argv:
            progress5 = argv[0]
            offset = 0
            while offset < len(nodeRange):
                offset = len(nodeRange) - count1(nodeRange)
                progress5.emit(int((offset / float(numberofJobs)) * 100))
                time.sleep(15)
    else:

        input_data = divide1(nodeRange) # changed Nov 30, 2018, S.M.
        if argv:
            offset = len(nodeRange) - len(input_data)
            #print 'optical offset',offset
            progress5.emit(int((offset / float(numberofJobs)) * 100))

        if p.ncpu == 1:  # avoids the multiprocessing package
            for i in xrange(len(input_data)):
                ComputeOptFlowPrDPsiAll1(input_data[i])
                if argv:
                    offset += 1
                    progress5.emit(int((offset / float(numberofJobs)) * 100))
        else:
            with poolcontext(processes=p.ncpu,maxtasksperchild=1) as pool:
                for i, _ in enumerate(pool.imap_unordered(partial(ComputeOptFlowPrDPsiAll1), input_data),1):
                    if argv:
                        offset += 1
                        progress5.emit(int((offset / float(numberofJobs)) * 100))
                    time.sleep(0.05)
                pool.close()
                pool.join()
