"""
Homework 5
Submission Functions
"""
import numpy as np
import helper
import math
from scipy.signal import convolve2d
data = np.load("../data/some_corresp.npz")


"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    


    # find the transformation matrix
    T = np.zeros((3,3))
    T[0,0] = 1/M
    T[1,1] = 1/M
    T[2,2] = 1
    pairs = pts1.shape[0]
    # initialize A
    A = np.zeros((pairs,9))
    # read out x and xprime, y and y prime, normalize
    pts1 = np.hstack((pts1,np.ones((pairs,1))))
    pts1 = (T@pts1.T).T

    pts2 = np.hstack((pts2,np.ones((pairs,1))))
    pts2 = (T@pts2.T).T

    x = pts1[:,0]
    xprime = pts2[:,0]
    y = pts1[:,1]
    yprime = pts2[:,1]
    # plug in the values into A
    A[:,0] = np.multiply(x,xprime)
    A[:,1] = np.multiply(x,yprime)
    A[:,2] = x
    A[:,3] = np.multiply(y,xprime)
    A[:,4] = np.multiply(y,yprime)
    A[:,5] = y
    A[:,6] = xprime
    A[:,7] = yprime
    A[:,8] = 1
    u, s, v = np.linalg.svd(A)
    F = v[-1, :].reshape(3, 3)
    F = F.T
    # enforce rank2
    F = helper.refineF(F, pts1[:,0:2], pts2[:,0:2])

    # enforce rank2
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U.dot(np.diag(S).dot(V))
    F = (T.T@F)@T

    return F

    
    
    



    






    



"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):
    # find the epipolar line first
    # append to hetero
    height = im1.shape[0]
    width = im1.shape[1]
    pairs = pts1.shape[0]
    pts1 = np.hstack((pts1,np.ones((pairs,1))))
    # initialize pts2
    pts2 = np.zeros((pairs,2))
    # define window 
    w = 5

    for i in range (pairs):
        input = pts1[i,:]


        inputx = int(input[0])
        inputy = int(input[1])
        
        neighbours = im1[inputy - w:inputy+w+1, inputx - w:inputx+w+1]
        
        # find the line


        l = np.matmul(F,input.T)

        a = l[0]
        b = l[1]
        c = l[2]
        candidatesx = np.array(list(range(width)))
        candidatesx = candidatesx.astype(int)
        # ax+by+c = 0
        candidatesy = (a*candidatesx+c)/(-b)
        candidatesy = candidatesy.astype(int)
        currdiff = math.inf
        bestx, besty = -1,-1
        for j in range (len(candidatesx)):
            # check if it is out of bound
            cx = candidatesx[j]
            cy = candidatesy[j]
            
            if cx - w > 0 and cx + w + 1 < width and cy-w>0 and cy+w+1 < height:
                sliding = im2[(cy - w):(cy+w+1), (cx - w):(cx+w+1)]
                # calculate difference 
                diff = np.sum((neighbours-sliding)**2)
                if diff < currdiff:
                    bestx = cx
                    besty = cy
                    currdiff = min(diff,currdiff)
        pts2[i,0] = bestx
        pts2[i,1] = besty


    return pts2

        


    


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    return np.matmul(np.matmul(K2.T,F),K1)


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):

    # read in the x and y
    pairs = pts1.shape[0]
    X = pts1[:,0]
    Xprime = pts2[:,0]
    Y = pts1[:,1]
    Yprime = pts2[:,1]
    # initialize A 

    # read in the p1,p2,p3
    p1t = P1[0,:]
    p2t = P1[1,:]
    p3t = P1[2,:]

    p1tprime = P2[0,:]
    p2tprime = P2[1,:]
    p3tprime = P2[2,:]
    pts3d = np.zeros((pairs,3))
    for j in range (pairs):
        x = X[j]
        xprime = Xprime[j]
        y = Y[j]
        yprime = Yprime[j]

        a1 = y*p3t - p2t
        a2 = p1t - x*p3t
        a3 = yprime*p3tprime - p2tprime
        a4 = p1tprime - xprime*p3tprime
        A = np.vstack((a1,a2,a3,a4))
        u, s, v = np.linalg.svd(A)
        p = v[-1, :]
        z = p[-1]

        pts3d[j,:] = p[0:3]/z
    # compute the projection error
    totalloss1 = 0
    totalloss2 = 0
    for i in range (pairs):
        threedp = [pts3d[i,0],pts3d[i,1],pts3d[i,2],1]
        predict1 = np.matmul(P1,threedp)
        predict1 = predict1.T
        z1 = predict1[-1]
        # homo
        predict1 /= z1
        error1 = math.sqrt(((predict1[0] - pts1[i,0])**2+(predict1[1] - pts1[i,1])**2))

        predict2 = np.matmul(P2,threedp)
        predict2 = predict2.T
        z2 = predict2[-1]
        # homo
        predict2 /= z2
        error2 = math.sqrt(((predict2[0] - pts2[i,0])**2+(predict2[1] - pts2[i,1])**2))
        totalloss1 += error1
        totalloss2 += error2
    totalloss1/=pairs
    totalloss2/=pairs




    return pts3d,totalloss1,totalloss2






    

    


    

    



"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):

    c1 = -np.matmul(np.linalg.inv(np.matmul(K1,R1)), np.matmul(K1,t1))
    c2 = -np.matmul(np.linalg.inv(np.matmul(K2,R2)), np.matmul(K2,t2))
    # find the norm

    norm = np.sqrt(np.sum(np.square(c1-c2)))
    r1 = (c1-c2)/norm
 
    oldz = (R1[2,:]).T

    r2 = np.cross(oldz,r1)
    r3 = np.cross (r2,r1)
    r1 = r1.reshape((3,1))
    r2 = r2.reshape((3,1))
    r3 = r3.reshape((3,1))


    newR = np.hstack((r1,r2,r3))

    newR = newR.T
    newR1 = newR
    newR2 = newR
    newK1 = K2
    newK2 = K2
    newt1 = -np.matmul(newR,c1)
    newt2 = -np.matmul(newR,c2)
    newM1 = np.matmul(np.matmul(newK1,newR1),np.linalg.inv(np.matmul(K1,R1)))
    newM2 = np.matmul(np.matmul(newK2,newR2),np.linalg.inv(np.matmul(K2,R2)))
    return (newM1,newM2,newK1,newK2,newR1,newR2,newt1,newt2)


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):

    height = im1.shape[0]
    width = im1.shape[1]

    i1sq = np.multiply(im1,im1)
    dispM = np.ones((height,width))
    dispM *= math.inf
    res = np.zeros((height,width))
    


    
    window = np.ones((win_size,win_size))
    cv1 = convolve2d(i1sq,window,mode = 'same')
    for d in range (max_disp+1):


        im2 = np.roll(im2, -1, axis=1)


        i2sq = np.multiply(im2,im2)
        i1i2 = np.multiply(im1,im2)
        cv12 = convolve2d(i1i2,window,mode = 'same')
        cv2 = convolve2d(i2sq,window,mode = 'same')
        dp = cv1 - 2*cv12 + cv2
        res[dp < dispM] = d
        dispM[dp < dispM] = dp[dp < dispM]
        


    return res 


    '''
    for y in range (height):
        print(y)
        for x in range (width):
            argmin = math.inf
            
            for d in range(max_disp):
                # check if it is out of bound
                if y-w >= 0 and y+w<height and x-w >=0 and x+w<width and x-d>=0:

                    window1 = im1[(y - w):(y+w+1), (x - w):(x+w+1)]
                    window2 = im2[(y - w):(y+w+1), (x - w):(x+w+1)]
                    term1 = np.sum(np.square(window1))
                    term2 = np.sum(convolve2d(window1, window2[::-1, ::-1]))
                    term3 = np.sum(np.square(window2))
                    
                    slidingsum = term1 - 2*term2 + term3
                else:
                    slidingsum = 0
                argmin = min(argmin, slidingsum)
            dispM[y,x] = argmin

    return dispM
    '''
                
    


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    c1 = -np.matmul(np.linalg.inv(np.matmul(K1,R1)), np.matmul(K1,t1))
    c2 = -np.matmul(np.linalg.inv(np.matmul(K2,R2)), np.matmul(K2,t2))
    b = np.sqrt(np.sum(np.square(c1-c2)))
    f = K1[1,1]

    height = dispM.shape[0]
    width = dispM.shape[1]
    depthM= np.zeros((height,width))
    for y in range (height):
        for x in range (width):
            disp = dispM[y,x]
            if disp == 0:
                depthM[y,x] = 0
            else:
                depthM[y,x] = b*f/disp
    return depthM







"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    pass
