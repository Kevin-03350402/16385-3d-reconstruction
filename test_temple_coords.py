import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
import cv2

# 1. Load the two temple images and the points from data/some_corresp.npz
data = np.load("../data/some_corresp.npz")
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
# calculae the max of weight/height
M = max(im1.shape[0],im1.shape[1])

# 2. Run eight_point to compute F
eight_pointF = sub.eight_point(data['pts1'], data['pts2'], M)
print(eight_pointF)
#hlp.displayEpipolarF(im1,im2,eight_pointF)

# 3. Load points in image 1 from data/temple_coords.npz
images = np.load("../data/temple_coords.npz")
image1 = images['pts1']


# 4. Run epipolar_correspondences to get points in image 2
image2 = sub.epipolar_correspondences(im1, im2, eight_pointF, image1)

#hlp.epipolarMatchGUI(im1, im2, eight_pointF)



# 5. Compute the camera projection matrix P1
intrinsics = np.load("../data/intrinsics.npz")
K1 = intrinsics['K1']
K2 = intrinsics['K2']

E = sub.essential_matrix(eight_pointF,K1,K2)

M1 = np.zeros((3,4))
M1[0,0] = 1
M1[1,1] = 1
M1[2,2] = 1
# multiple the intrinsic with the extrinsic
P1 = np.matmul(K1,M1)
# 6. Use camera2 to get 4 camera projection matrices P2
# iterate all candidates

M2_candidates = hlp.camera2(E)




# find the correct matrix by counting the z
mostz = 0
bestmatrix = None
bestm2 = None
# 7. Run triangulate using the projection matrices
# 8. Figure out the correct P2
for i in range (4):
    M2 = M2_candidates[:,:,i]
    P2 = np.matmul(K2,M2)
    pts3d,totalloss1,totalloss2 = sub.triangulate(P1, image1, P2, image2)


    print(totalloss1,totalloss2)

    depth = pts3d[:,-1]
    frontd = (depth>0).sum()




    if frontd > mostz:
        mostz = frontd
        bestmatrix = pts3d
        bestm2 = M2
        print(i)





# 9. Scatter plot the correct 3D points
# cited from : https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(bestmatrix[:,0], bestmatrix[:,1], bestmatrix[:,2],color = 'black')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
R1 = M1[:,0:3]
R2 = bestm2[:,0:3]
t1 = M1[:,3]
t2 = bestm2[:,3]

np.savez('../data/extrinsics.npz', R1 =R1, R2= R2, t1 = t1, t2 = t2)

