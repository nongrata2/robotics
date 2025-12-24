import os
import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt

URDF_FILE    = "./three_link.urdf.xml"
USE_JACOBIAN = True
dt           = 1 / 240
maxTime      = 5.0
kp           = 6.0
max_joint_vel = 8.0
target_pos   = np.array([0.1, 0.0, 1.0])

def as_str(x):
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf‑8")
    return str(x)

def ensure_connection():
    if not p.isConnected():
        p.connect(p.GUI)

ensure_connection()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -10)


full_path = os.path.abspath(URDF_FILE)
print("\nЗагружаем URDF из:", full_path)
robotId = p.loadURDF(full_path, useFixedBase=True)


numJoints = p.getNumJoints(robotId)


print("\nJoint map:")
name_to_idx = {}
for i in range(numJoints):
    ji = p.getJointInfo(robotId, i)
    joint_name = as_str(ji[1])
    child_link = as_str(ji[12])
    joint_type = ji[2]
    axis_str   = as_str(ji[13])
    print(f"{i:2d} name={joint_name:15s} link(child)={child_link:10s} "
          f"type={joint_type} axis={axis_str}")
    name_to_idx[joint_name] = i

ctrl_joint_indices = [
    name_to_idx["joint_0"],
    name_to_idx["joint_1"],
    name_to_idx["joint_2"],
]

eefLinkIdx = name_to_idx["joint_eef2"]


dof_joint_indices = []
for i in range(numJoints):
    ji = p.getJointInfo(robotId, i)
    if ji[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
        dof_joint_indices.append(i)

numDof = len(dof_joint_indices)
print("\nnumDof =", numDof)
print("dof_joint_indices =", dof_joint_indices)

dof_index_map = {jid: k for k, jid in enumerate(dof_joint_indices)}
ctrl_cols = [dof_index_map[jid] for jid in ctrl_joint_indices]

def get_dof_q_dq(body_id, dof_indices):
    """
    Возвращает списки позиций (q) и скоростей (dq) только для движущихся
    суставов, указанных в dof_indices. Длина списка = numDof
    """
    js = p.getJointStates(body_id, dof_indices)
    q  = [st[0] for st in js]
    dq = [st[1] for st in js]
    return q, dq


q0 = [0.5, 0.5, 0.5]
p.setJointMotorControlArray(bodyIndex=robotId,
                            jointIndices=ctrl_joint_indices,
                            controlMode=p.POSITION_CONTROL,
                            targetPositions=q0)

for _ in range(1000):
    p.stepSimulation()
    time.sleep(dt)

logTime = np.arange(0.0, maxTime, dt)
sz      = len(logTime)

logX = np.zeros(sz)
logY = np.zeros(sz)
logZ = np.zeros(sz)

for step in range(sz):
    if not p.isConnected():
        ensure_connection()
        p.setGravity(0, 0, -10)

    link_state = p.getLinkState(robotId, eefLinkIdx, computeLinkVelocity=True)
    pos = np.array(link_state[0], dtype=float)
    vel = np.array(link_state[6], dtype=float)

    logX[step], logY[step], logZ[step] = pos

    err = np.array([pos[0] - target_pos[0],
                    pos[1] - target_pos[1],
                    pos[2] - target_pos[2]], dtype=float).reshape(3, 1)  # 3×1

    if USE_JACOBIAN:
        q_dof, dq_dof = get_dof_q_dq(robotId, dof_joint_indices)

        local_pos = [0.0, 0.0, 0.0]
        jac_t, jac_r = p.calculateJacobian(robotId,
                                           eefLinkIdx,
                                           local_pos,
                                           q_dof,
                                           dq_dof,
                                           [0.0] * numDof)

        Jt = np.array(jac_t, dtype=float)
        J = Jt[:, ctrl_cols]

        lam = 1e-3
        JJt = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJt + (lam ** 2) * np.eye(3))

        w = (-kp) * (J_pinv @ err)

        w = np.clip(w.flatten(), -max_joint_vel, max_joint_vel)


        p.setJointMotorControlArray(bodyIndex=robotId,
                                    jointIndices=ctrl_joint_indices,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=w.tolist())
    else:
        ik_sol = p.calculateInverseKinematics(bodyUniqueId=robotId,
                                               endEffectorLinkIndex=eefLinkIdx,
                                               targetPosition=target_pos.tolist(),
                                               lowerLimits=[-3.14, -3.14, -3.14],
                                               upperLimits=[ 3.14,  3.14,  3.14],
                                               jointRanges =[6.28, 6.28, 6.28],
                                               restPoses   =[0, 0, 0])
        p.setJointMotorControlArray(bodyIndex=robotId,
                                    jointIndices=ctrl_joint_indices,
                                    targetPositions=ik_sol[:3],
                                    controlMode=p.POSITION_CONTROL)

    p.stepSimulation()
    time.sleep(dt)

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.title('X(t)')
plt.plot(logTime, logX, label='X')
plt.hlines(target_pos[0], logTime[0], logTime[-1], colors='r', linestyles='--')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.title('Y(t)')
plt.plot(logTime, logY, label='Y')
plt.hlines(target_pos[1], logTime[0], logTime[-1], colors='r', linestyles='--')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.title('Z(t)')
plt.plot(logTime, logZ, label='Z')
plt.hlines(target_pos[2], logTime[0], logTime[-1], colors='r', linestyles='--')
plt.grid(True)

plt.tight_layout()
plt.show()

# trajectory
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(logX, logY, logZ, '-b', label='trajectory')
ax.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]],
           c='r', marker='x', s=100, label='target')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

if p.isConnected():
    p.disconnect()
