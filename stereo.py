
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ======================
# CONFIGURATION
# ======================
kitti_base_path = "/mnt/6tbdisk/vopipeline/dataset/sequences/00"
left_images_path = os.path.join(kitti_base_path, "image_0")
right_images_path = os.path.join(kitti_base_path, "image_1")
gt_path = "/mnt/6tbdisk/vopipeline/dataset/poses/00.txt"     # Ground truth
save_traj_txt = "/mnt/6tbdisk/vopipeline/trajectory_00.txt"  # === plot.py expects this
save_pose_txt = "/mnt/6tbdisk/vopipeline/poses_00.txt"       # SE3 3x4 matrices

# Camera intrinsics KITTI 00
K = np.array([[718.856, 0, 607.1928],
              [0, 718.856, 185.2157],
              [0, 0, 1]])
baseline = 0.573  # meters

# ======================
# LOAD GROUND TRUTH
# ======================
def load_gt(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            T = np.fromstring(line, sep=' ').reshape(3, 4)
            T = np.vstack([T, [0, 0, 0, 1]])
            poses.append(T)
    return np.array(poses)

ground_truth = load_gt(gt_path)
print("Loaded GT poses:", len(ground_truth))

# ======================
# FEATURE MATCHING (ORB)
# ======================
orb = cv2.ORB_create(2500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

def match_features(img1, img2):
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    matches = bf.knnMatch(des1, des2, k=2)

    pts1, pts2 = [], []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    return np.array(pts1), np.array(pts2)

# ======================
# TRIANGULATION
# ======================
def triangulate(pts_L, pts_R):
    disparity = pts_L[:, 0] - pts_R[:, 0]
    valid = disparity > 1.0

    ptsL = pts_L[valid]
    disparity = disparity[valid]

    depth = (K[0, 0] * baseline) / (disparity + 1e-6)
    X = (ptsL[:, 0] - K[0, 2]) * depth / K[0, 0]
    Y = (ptsL[:, 1] - K[1, 2]) * depth / K[1, 1]
    Z = depth

    pts3D = np.column_stack((X, Y, Z))
    return pts3D, valid

# ======================
# PnP ESTIMATION
# ======================
def estimate(Rt_prev, pts3D, pts2D):
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3D, pts2D, K, None,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None, None, None

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec[:, 0]
    return T, R, tvec

# ======================
# LOAD IMAGES
# ======================
left_imgs = sorted([os.path.join(left_images_path, f)
                    for f in os.listdir(left_images_path) if f.endswith(".png")])
right_imgs = sorted([os.path.join(right_images_path, f)
                     for f in os.listdir(right_images_path) if f.endswith(".png")])

N = min(len(left_imgs), len(right_imgs))
print("Total Frames:", N)

# ======================
# INITIALIZATION
# ======================
poses_SE3 = []
trajectory_xyz = []

current_pose = np.eye(4)

poses_SE3.append(current_pose[:3, :].reshape(12))
trajectory_xyz.append(current_pose[:3, 3])

# ======================
# MAIN LOOP
# ======================
for i in range(1, min(N, 2000)):   # First 300 frames
    imgL_prev = cv2.imread(left_imgs[i - 1], 0)
    imgR_prev = cv2.imread(right_imgs[i - 1], 0)
    imgL = cv2.imread(left_imgs[i], 0)
    imgR = cv2.imread(right_imgs[i], 0)

    # ---- Stereo Match ----
    pts_L, pts_R = match_features(imgL_prev, imgR_prev)
    print(f"[Frame {i}] Stereo matches = {len(pts_L)}")

    # ---- Triangulate ----
    pts3D, valid_mask = triangulate(pts_L, pts_R)

    # ---- Temporal Match (L_prev → L_curr) ----
    pts_prev, pts_curr = match_features(imgL_prev, imgL)
    print(f"[Frame {i}] Temporal matches = {len(pts_prev)}")

    # Associate 3D→2D
    pts3D_align, pts2D_align = [], []
    for j, pt in enumerate(pts_L[valid_mask]):
        d = np.linalg.norm(pts_prev - pt, axis=1)
        idx = np.argmin(d)
        if d[idx] < 2.0:
            pts3D_align.append(pts3D[j])
            pts2D_align.append(pts_curr[idx])

    pts3D_align = np.array(pts3D_align)
    pts2D_align = np.array(pts2D_align)

    print(f"[Frame {i}] Valid 3D-2D = {len(pts3D_align)}")

    if len(pts3D_align) < 10:
        print(f"[Frame {i}] Skipping...")
        poses_SE3.append(current_pose[:3, :].reshape(12))
        trajectory_xyz.append(current_pose[:3, 3])
        continue

    # ---- Estimate Pose ----
    T, R, t = estimate(current_pose, pts3D_align, pts2D_align)
    if T is None:
        print(f"[Frame {i}] PnP failed.")
        poses_SE3.append(current_pose[:3, :].reshape(12))
        trajectory_xyz.append(current_pose[:3, 3])
        continue

    # ---- Accumulate Pose ----
    current_pose = current_pose @ np.linalg.inv(T)
    xyz = current_pose[:3, 3]

    print(f"[Frame {i}] Pose = X={xyz[0]:.2f}, Z={xyz[2]:.2f}")

    poses_SE3.append(current_pose[:3, :].reshape(12))
    trajectory_xyz.append(xyz)

# ======================
# SAVE RESULTS (MATCHES plot.py)
# ======================
trajectory_xyz = np.array(trajectory_xyz)
poses_SE3 = np.array(poses_SE3)

np.savetxt(save_traj_txt, trajectory_xyz, fmt="%.6f")
np.savetxt(save_pose_txt, poses_SE3, fmt="%.6f")

print("\nSaved:")
print(" →", save_traj_txt)
print(" →", save_pose_txt)

# ======================
# QUICK PLOT
# ======================
plt.plot(trajectory_xyz[:, 0], trajectory_xyz[:, 2], label="Estimated")
plt.plot(ground_truth[:len(trajectory_xyz), 0, 3],
         ground_truth[:len(trajectory_xyz), 2, 3],
         label="GT")
plt.legend()
plt.grid()
plt.title("Stereo VO Trajectory")
plt.show()
