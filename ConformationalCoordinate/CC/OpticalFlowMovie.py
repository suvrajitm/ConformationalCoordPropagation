import numpy as np
import os,sys
import cv2
from cv2 import calcOpticalFlowFarneback,calcOpticalFlowPyrLK
from subprocess import call
import copy
sys.path.append('../')
import myio
import p
from scipy.ndimage import median_filter
import hornschunck_simple
from hornschunck_simple import lowpassfilt

''' Suvrajit Maji,sm4073@cumc.columbia.edu
    Columbia University
    Created: May 2018. Modified:Aug 16,2019
'''

def getOrientMag(X,Y):
    orient = np.arctan2(Y,X) % 2*np.pi
    mag = np.sqrt(X**2 + Y**2)
    return orient,mag

def normalizeRescaleVector(f,normalizeVec,rescaleRange):

    dims = np.shape(f)
    F = np.empty(dims)
    if rescaleRange: # rescaleRange= 0 if no scaling range [min,max] provided
        rescaleVec = 1
    else:
        rescaleVec = 0
        

    if  len(dims) > 2:
        #print 'Normalizing MxMx2 vector'
        fx = f[:,:,0]
        fy = f[:,:,1]
        l = np.sqrt(fx**2 + fy**2)+1e-10

        if normalizeVec:
            if rescaleVec:
                s = np.interp(l, (np.min(l), np.max(l)),rescaleRange)
            else:
                s = 1
            fx = np.multiply(s,np.divide(fx,l))
            fy = np.multiply(s,np.divide(fy,l))

        F[:,:,0] = fx
        F[:,:,1] = fy
    else:
        F = np.interp(f, (np.min(f), np.max(f)),rescaleRange)

    return F



def  writeOpticalFlowImage(outfile,label,img,flow,OFvisualPrint,step=4):
 
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.colorbar as mcolorbar

    displayFlow = OFvisualPrint[0]
    printFlowFig = OFvisualPrint[1]

    h, w = img.shape[:2]
    #reshape(2,-1) flattens the matrix
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    dim = np.sqrt(fx.shape[0]).astype(int)
    o,l = getOrientMag(fx,fy)
    #make sure that l is flattened, so that C is an array to be used in color bar
    l = l.flatten()
    C = normalizeRescaleVector(l,1,[0.0, 1.0])
    nz = mcolors.Normalize()
    nz.autoscale(C)

    if displayFlow or printFlowFig:

        #Only normalize (no rescale) the vectors for display
        flow = normalizeRescaleVector(flow,1,0)
        fx, fy = flow[y,x].T
        label_size = 20
        fig = plt.figure('OpticalFlow',figsize=(10,10))
        fig.clf()
        ax = fig.add_subplot(111)
        plt.imshow(img,cmap='gray')
        quiverPlot = 1
        streamPlot = 0
        if quiverPlot:
            plt.quiver(x,y,fx,fy,units='width',pivot='tail',scale=32,headwidth=3.0,headlength=5.25,headaxislength=5.0,width=0.0025,linewidth=0.05,color=cm.jet(nz(C)))
        if streamPlot:
            ax.streamplot(x.reshape((dim,dim)),y.reshape((dim,dim)),fx.reshape((dim,dim)),fy.reshape((dim,dim)),arrowsize=2,color=C.reshape((dim,dim)),density = 10)
        cax,_ = mcolorbar.make_axes(plt.gca())
        cb = mcolorbar.ColorbarBase(cax, cmap=cm.jet, norm=nz) #extend='max'
        cb.set_label('Relative magnitude',fontsize=label_size)
        cb.ax.tick_params(labelsize=label_size)
        ax.tick_params(axis='both',labelsize=label_size)
        ax.set_title('Sense:'+label,fontsize= label_size)
        if displayFlow:
            plt.show()
        if printFlowFig:
            fig.savefig(outfile + '.png')



def plot_optical_flow(img,U,V):

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.colorbar as mcolorbar
    print 'plotting optical flow'

    '''
    Plots optical flow given U,V and one of the images
    '''
    # Change t if required, affects the number of arrows
    # t should be between 1 and min(U.shape[0],U.shape[1])
    t=10
    # Subsample U and V to get visually pleasing output
    U1 = U[::t,::t]
    V1 = V[::t,::t]
    # Create meshgrid of subsampled coordinates
    r, c = img.shape[0],img.shape[1]
    cols,rows = np.meshgrid(np.linspace(0,c-1,c), np.linspace(0,r-1,r))
    cols = cols[::t,::t]
    rows = rows[::t,::t]
    # Plot optical flow
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.quiver(cols,rows,U1,V1)
    plt.quiver(cols,rows,U1,V1,units='width',pivot='tail',scale=32,headwidth=3.0,headlength=5.25,headaxislength=5.0,width=0.0025,linewidth=0.05)
    plt.show()



def figurePlot(mat):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(mat,cmap='gray')
    plt.show()



def SelectFlowVec(FlowVec,flowVecPctThresh):
    # this will work for all elements for any ndarray [m x m x d],
    # Here for half-split movies we have d = 2, i.e. stack of two m x m matrix

    VxM = FlowVec['Vx']
    VyM = FlowVec['Vy']

    FOrientMat = FlowVec['Orient']
    FMagMat = FlowVec['Mag']
    #print 'flowVecPctThresh',flowVecPctThresh
    FMag = FMagMat.flatten()
    #print  'FMag', np.max(FMag),np.min(FMag)
    magThresh = np.percentile(FMag,flowVecPctThresh)
    #print 'magThresh',magThresh
    not_magThIdx = np.where(FMagMat <= magThresh)

    FMagSel = FMagMat
    FMagSel[not_magThIdx] = 0.

    FOrientSel = FOrientMat
    FOrientSel[not_magThIdx] = -np.Inf

    #selecting which flow vectors to keep for analysis
    VxMSel = VxM
    VyMSel = VyM
    VxMSel[not_magThIdx]= 0.
    VyMSel[not_magThIdx]= 0.

    FlowVecSel = dict(Vx=VxMSel,Vy=VyMSel,Orient=FOrientSel,Mag=FMagSel)
    return FlowVecSel


def movingAverage(X, window_size):
    #window_mask = np.ones(int(window_size))/float(window_size) # 1D only
    #X_ma = np.convolve(X, window_mask, 'same',axis=2)
    from scipy.ndimage import uniform_filter
    X_ma  = uniform_filter(X, size=window_size, origin=-1,mode='wrap')
    return X_ma

def anisodiff3(stack,niter=1,kappa=50,gamma=0.1,step=(1.,1.,1.),option=1,ploton=False):

    import warnings

    '''
    3D Anisotropic diffusion.

    Usage:
    stackout = anisodiff(stack, niter, kappa, gamma, option)

    Arguments:
            stack  - input stack
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (z,y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the middle z-plane will be plotted on every
                 iteration

    Returns:
            stackout   - diffused stack.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x,y and/or z axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference:
    P. Perona and J. Malik.
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    '''

    #...you could always diffuse each color channel independently if you
    # really want
    if stack.ndim == 4:
        warnings.warn("Only grayscale stacks allowed, converting to 3D matrix")
        stack = stack.mean(3)

    # initialize output array
    stack = stack.astype('float32')
    stackout = stack.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(stackout)
    deltaE = deltaS.copy()
    deltaD = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    UD = deltaS.copy()
    gS = np.ones_like(stackout)
    gE = gS.copy()
    gD = gS.copy()

    # create the plot figure, if requested
    if ploton:
        import pylab as pl
        from time import sleep

        showplane = stack.shape[0]//2

        fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
        ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)

        ax1.imshow(stack[showplane,...].squeeze(),interpolation='nearest')
        ih = ax2.imshow(stackout[showplane,...].squeeze(),interpolation='nearest',animated=True)
        ax1.set_title("Original stack (Z = %i)" %showplane)
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    for ii in np.arange(1,niter):

        # calculate the diffs
        deltaD[:-1,: ,:  ] = np.diff(stackout,axis=0)
        deltaS[:  ,:-1,: ] = np.diff(stackout,axis=1)
        deltaE[:  ,: ,:-1] = np.diff(stackout,axis=2)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gD = np.exp(-(deltaD/kappa)**2.)/step[0]
            gS = np.exp(-(deltaS/kappa)**2.)/step[1]
            gE = np.exp(-(deltaE/kappa)**2.)/step[2]
        elif option == 2:
            gD = 1./(1.+(deltaD/kappa)**2.)/step[0]
            gS = 1./(1.+(deltaS/kappa)**2.)/step[1]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[2]

        # update matrices
        D = gD*deltaD
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'Up/North/West' by one
        # pixel. don't as questions. just do it. trust me.
        UD[:] = D
        NS[:] = S
        EW[:] = E
        UD[1:,: ,: ] -= D[:-1,:  ,:  ]
        NS[: ,1:,: ] -= S[:  ,:-1,:  ]
        EW[: ,: ,1:] -= E[:  ,:  ,:-1]

        # update the image
        stackout += gamma*(UD+NS+EW)

        if ploton:
            iterstring = "Iteration %i" %(ii+1)
            ih.set_data(stackout[showplane,...].squeeze())
            ax2.set_title(iterstring)
            fig.canvas.draw()
        # sleep(0.01)

    return stackout


# Optical Flow computation
def op(Mov,prd_psinum,blockSize_avg,label,OFvisualPrint,*argv):
    # for display
    prD = prd_psinum[0]
    psinum_prD = prd_psinum[1]

    numFrames = Mov.shape[0]
    dim = int(np.sqrt(Mov.shape[1]))

    Mov = np.resize(Mov, (numFrames,dim, dim))

    # this is temporary as long as the transpose of the movies are not corrected before nlsa movies
    #Mov = np.swapaxes(Mov,1,2)

    # for display use original images
    #displayImg = Mov[0,:,:]
    #print 'Mov.shape',np.shape(Mov)
    if label[0:3]=='FWD':
        #print 'Optflow label',label[0:3]
        displayImg = Mov[0,:,:]

    elif label[0:3]=='REV':
        #print 'Optflow label',label[0:3]
        displayImg = Mov[numFrames-1,:,:]

    VxM = np.zeros((dim,dim))
    VyM = np.zeros((dim,dim))

    inputFWD = 0
    FlowVec=[]
    if argv:
        FlowVec = copy.deepcopy(argv[0])
        inputFWD=1

    if not inputFWD:
        # implement the sliding window at some point
        do_simpleAvg = 1
        do_movingAvg = 0
        do_filterImage = 0
        sig = 2.0 #sigma for lowpass gauss filter
        #OF_Type='GF' # Gunnar-Farneback
        #OF_Type='HS' # Horn-Schunck
        OF_Type ='both' #GF for initial estimates and then HS

        if OF_Type=='GF' or OF_Type =='both':
            do_filterImage = 1
            if OF_Type =='both':
                sig=1.5

        # if needed use a median fitler for the optical flow vector field
        #medfw = 7

        #print 'Apply diffusion filter to the movie'
        Mov = anisodiff3(Mov,niter=5,kappa=50,gamma=0.1,step=(3.,1.,1.),option=1,ploton=False)


        #print 'numFrames',numFrames,'blockSize_avg',blockSize_avg
        if do_simpleAvg: # perform averaging over blockSize frames
            #print 'Perform averaging of movie frames'
            numAvgFrames = np.ceil(np.float(numFrames)/blockSize_avg).astype(int)
            AvgMov = np.zeros((numAvgFrames,dim,dim))
            for b in range(0,numAvgFrames):
                frameStart = b*blockSize_avg
                frameEnd = min((b+1)*blockSize_avg,numFrames)
                #print 'frameStart',frameStart,'frameEnd',frameEnd
                blockMovie = Mov[frameStart:frameEnd,:,:]
                AvgMov[b,:,:] = np.mean(blockMovie,axis=0,dtype=np.float64)

        elif do_movingAvg:
            ma_window_size = [blockSize_avg,0,0] #use the same averaging window size
            AvgMov = movingAverage(Mov, ma_window_size)
            numAvgFrames = AvgMov.shape[0]


        else:# keep original movie
            #print 'No averaging of movies done'
            numAvgFrames = numFrames
            AvgMov = Mov
        #print 'AvgMov.shape',np.shape(AvgMov)

        #start Optical flow algorithm
        ImgFrame_prev = AvgMov[0,:,:]

        if do_filterImage:
            # test image filter
            # this params when using bliateral filter
            d = 9
            sc = 25
            sp = 25
            #ImgFrame_prev = cv2.bilateralFilter(np.float32(ImgFrame_prev),d,sc,sp)
            ImgFrame_prev  = lowpassfilt(ImgFrame_prev,sig)

            #figurePlot(ImgFrame_prev)

        for frameno in range(0,numAvgFrames):
            ImgFrame_curr = AvgMov[frameno,:,:]

            #print ImgFrame_curr[120:125,120:125].T

            if do_filterImage:
                #ImgFrame_curr = cv2.bilateralFilter(np.float32(ImgFrame_curr),d,sc,sp)
                ImgFrame_curr = lowpassfilt(ImgFrame_curr,sig)
            #print 'prev img'
            #figurePlot(ImgFrame_prev)
            #print 'curr img'
            #figurePlot(ImgFrame_curr)

            #print 'diff-image:'
            #diff_p_c = ImgFrame_curr - ImgFrame_prev
            #figurePlot(diff_p_c)

            if OF_Type=='GF' or OF_Type=='both':
            #flow = calcOpticalFlowFarneback(ImgFrame_prev, ImgFrame_curr, flow=None, pyr_scale=0.4, levels=1, winsize=15,iterations=10,poly_n=5, poly_sigma=1.2, flags=0)
            #flow = calcOpticalFlowFarneback(ImgFrame_prev, ImgFrame_curr, flow=None, pyr_scale=0.5, levels=5, winsize=25,iterations=10,poly_n=7, poly_sigma=1.5, flags=0)
                flow = calcOpticalFlowFarneback(ImgFrame_prev, ImgFrame_curr, flow=None, pyr_scale=0.4, levels=5, winsize=21,iterations=10,poly_n=7, poly_sigma=1.5, flags=0)
                Vx,Vy = flow[:,:,0], flow[:,:,1]


            if OF_Type=='HS' or OF_Type=='both':
                if OF_Type=='both':
                    uInit = Vx # if both methods are used , GF will provide intial estimates
                    vInit = Vy # if both methods are used , GF will provide intial estimates
                else:
                    uInit =np.zeros(ImgFrame_prev.shape)
                    vInit =np.zeros(ImgFrame_prev.shape)

                Vx,Vy = hornschunck_simple.op(ImgFrame_prev, ImgFrame_curr,uInit,vInit,sig,3.,200)

            VxM = VxM + Vx
            VyM = VyM + Vy

            ImgFrame_prev = copy.deepcopy(ImgFrame_curr)

        #VxM = median_filter(VxM,size=medfw)
        #VyM = median_filter(VyM,size=medfw)

        # store the flow vectors in a dictionary
        FlowVec = dict(Vx=[],Vy=[],Orient=[],Mag=[])
        FlowVec['Vx'] = VxM
        FlowVec['Vy'] = VyM

    else:
        # read input FWD vectors when available
        VxM = FlowVec['Vx']
        VyM = FlowVec['Vy']

    # temporary trial of getting negative vectors directly from FW vectors
    #print 'VxM',VxM[160:165,160:165]
    #print 'VyM',VyM[160:165,160:165]

    if label[0:3]=='REV':
        #print 'Optflow label if rev',label[0:3]
        VxM = copy.deepcopy(-1.0*VxM)
        VyM = copy.deepcopy(-1.0*VyM)
        FlowVec['Vx'] = VxM
        FlowVec['Vy'] = VyM

    # get orientation and magnitude of the flow vectors
    FOrientMat, FMagMat = getOrientMag(VxM,VyM)

    FlowVec['Orient'] = FOrientMat
    FlowVec['Mag'] = FMagMat

    if OFvisualPrint[0] or OFvisualPrint[1]:
        # uses indexing 1 for user output
        filename = "flow_prd_" + str(prD+1) + '_psi_' + str(psinum_prD+1) + '_' + str(label)
        CC_OF_fig_dir = os.path.join(p.CC_dir,'CC_OF_fig/PrD_'+str(prD+1)+'/')
        call(["mkdir", "-p", CC_OF_fig_dir])
        figOutfile = os.path.join(CC_OF_fig_dir,filename)

        dim = VxM.shape[0]
        flow = np.zeros((dim,dim,2))

        FlowVecSel = SelectFlowVec(FlowVec,p.opt_movie['flowVecPctThresh'])
        #print 'flowVecPctThresh',p.opt_movie['flowVecPctThresh']
        #visualize only the selected flow vectors
        VxMSel = FlowVecSel['Vx']
        VyMSel = FlowVecSel['Vy']
        flow[:,:,0] = VxMSel
        flow[:,:,1] = VyMSel

        #plot_optical_flow(displayImg,VxMSel,VyMSel)
        writeOpticalFlowImage(figOutfile,label,displayImg,flow,OFvisualPrint,step=4)

    #print 'Vx-',label[0:3],FlowVec['Vx'][150:155,150:155]
    return FlowVec
