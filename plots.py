import numpy as np
import matplotlib.pyplot as plt

# ===================== LOAD TRAJECTORIES ===================== #

gt_path = "/mnt/6tbdisk/vopipeline/dataset/poses/00.txt"
est_path = "trajectory_00.txt"     # your saved trajectory text

def load_gt(file):
    poses = []
    with open(file, "r") as f:
        for line in f.readlines():
            T = np.fromstring(line, sep=" ").reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            poses.append(T)
    return np.array([p[:3, 3] for p in poses])

def load_est(file):
    arr = []
    with open(file, "r") as f:
        for line in f.readlines():
            xyz = np.fromstring(line, sep=" ")
            arr.append(xyz)
    return np.array(arr)

gt = load_gt(gt_path)
est = load_est(est_path)

N = min(len(gt), len(est))
gt = gt[:N]
est = est[:N]

# ===================== ALIGN WITH UMeyama ===================== #

def umeyama_alignment(X, Y):
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)

    Xc = X - mu_X
    Yc = Y - mu_Y

    S = Xc.T @ Yc / X.shape[0]

    U, D, Vt = np.linalg.svd(S)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt

    var_X = (Xc ** 2).sum() / X.shape[0]
    scale = D.sum() / var_X

    t = mu_Y - scale * R @ mu_X

    return scale, R, t

scale, R, t = umeyama_alignment(est, gt)
est_aligned = (scale * est @ R.T) + t

# ===================== METRIC COMPUTATION ===================== #

def trajectory_length(traj):
    return np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))

total_dist_gt = trajectory_length(gt)
total_dist_est = trajectory_length(est)

start_gt = gt[0]
end_gt = gt[-1]

start_est = est[0]
end_est_raw = est[-1]
end_est_aligned = est_aligned[-1]

drift_raw = np.linalg.norm(end_est_raw - end_gt)
drift_aligned = np.linalg.norm(end_est_aligned - end_gt)

errors = np.linalg.norm(est_aligned - gt, axis=1)
mean_ate = np.mean(errors)
rmse_ate = np.sqrt(np.mean(errors**2))
max_ate = np.max(errors)

# ===================== PRINT METRICS ===================== #

print("\n====================== METRICS ======================")
print(f"Start Position (GT):   {start_gt}")
print(f"End Position (GT):     {end_gt}\n")

print(f"Start Position (Est):  {start_est}")
print(f"End Position (Est RAW):      {end_est_raw}")
print(f"End Position (Est Aligned):  {end_est_aligned}\n")

print(f"Total GT Distance:          {total_dist_gt:.2f} m")
print(f"Total Estimated Distance:   {total_dist_est:.2f} m\n")

print(f"Final Drift (RAW):          {drift_raw:.2f} m")
print(f"Final Drift (Aligned):      {drift_aligned:.2f} m\n")

print(f"Mean ATE:                   {mean_ate:.3f} m")
print(f"RMSE ATE:                   {rmse_ate:.3f} m")
print(f"Max ATE:                    {max_ate:.3f} m\n")

print("Alignment Scale:", scale)
print("Rotation Matrix:\n", R)
print("Translation Vector:", t)

# ===================== MAIN TRAJECTORY PLOT ===================== #

plt.figure(figsize=(14, 6))
plt.plot(gt[:, 0], gt[:, 2], 'b-', label="GT")
plt.plot(est[:, 0], est[:, 2], color='orange', label="Estimated (RAW)")
plt.plot(est_aligned[:, 0], est_aligned[:, 2], 'r--', label="Estimated (Aligned)")
plt.xlabel("X [m]")
plt.ylabel("Z [m]")
plt.title("Trajectory Comparison (GT vs Estimated)")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()

# ===================== ATE PLOT ===================== #

plt.figure(figsize=(12, 4))
plt.plot(errors)
plt.title("Absolute Trajectory Error (ATE) over Time")
plt.xlabel("Frame")
plt.ylabel("ATE [m]")
plt.grid(True)
plt.tight_layout()
plt.show()
