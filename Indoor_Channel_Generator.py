import numpy as np
import random
import argparse
from itertools import product
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver, subcarrier_frequencies, Camera
import tensorflow as tf

# Argument parser 설정
parser = argparse.ArgumentParser(description='Indoor Channel Generator')
parser.add_argument('--seed', type=int, default=43, help='Random seed value (default: 43)')
parser.add_argument('--scene', type=int, default=0, help='Scene number (default: 0)')
args = parser.parse_args()

# 시드 설정
random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

from tqdm import tqdm

# Scene 로드
h_freq_want=[]
ue_pos_want=[]
h_tau_want=[]
obj_pos_want=[]
a_want=[]
tau_want=[]
x_vals = np.arange(1.5, 9, 1)
z_vals = -np.arange(1.5, 9, 1)

x_vals_ = np.arange(1, 9, 2)
z_vals_ = -np.arange(1, 9.5, 1.5)
ue_y = 1.2

ue_positions = list(product(x_vals_, [ue_y], z_vals_))
print(len(ue_positions))
scene = load_scene("Indoor.xml")

        # Configure antenna array for all transmitters
scene.tx_array = PlanarArray(num_rows=1,
                                    num_cols=3,
                                    vertical_spacing=0.2,
                                    horizontal_spacing=0.2,
                                    pattern="tr38901",
                                    polarization="H")

        # Configure antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.2,
                                    horizontal_spacing=0.2,
                                    pattern="tr38901", 
                                    polarization="V")

# Transmitter 생성
tx = Transmitter(name="tx",
                        position=[5.0, 2.5, -5.0],display_radius=0.5)
scene.add(tx)
scene_list=[]

def move_object(name, y_val):
    obj = scene.get(name)
    x = random.choice(x_vals)
    z = random.choice(z_vals)
    rot = random.uniform(0, 2*np.pi)

    obj.position = list(map(float, [x, y_val, z]))
    obj.orientation = [0.0, rot, 0.0]
    return list(map(float, [x, y_val, z])), rot

        # Instantiate a path solver
        # The same path solver can be used with multiple scenes
p_solver  = PathSolver()
max_num_paths=30
for i in tqdm(range(1)):
            # 움직이는 객체 포지션 설정 함수
    curtain_pos, curtain_rot = move_object("curtain", 1.5)
    cabinet_pos, cabinet_rot = move_object("kitchen_cabinet_closed_side", 1.0)
    
    for ue_pos in ue_positions:
        for j in range(30):
            rx = Receiver(name="rx", position=list(map(float,ue_pos)), display_radius=0.5)
            scene.add(rx)
        # Tx가 Rx를 바라보도록 설정
            
        # Reference Grid
        #ue_pos = random.choice(ue_positio
            
        

            # Compute propagation paths
            paths = p_solver(scene=scene,
                            max_depth=5,
                            los=True,
                            specular_reflection=True,
                            diffuse_reflection=False,
                            refraction=True,
                            synthetic_array=False,
                            max_num_paths_per_src=100000,
                            seed=random.randint(0, 1000000))

            # OFDM system parameters
            num_subcarriers = 30
            subcarrier_spacing=15e3

            # Compute frequencies of subcarriers relative to the carrier frequency
            frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)

            # Compute channel frequency response
            h_freq = paths.cfr(frequencies=frequencies,
                            normalize=True, # Normalize energy
                            normalize_delays=False,
                            out_type="numpy", num_time_steps=30)
            h_tau, _ = paths.cir(sampling_frequency=frequencies,
                               num_time_steps=30, 
                               normalize_delays= True, 
                               out_type="numpy")

            h_tau = np.squeeze(h_tau)
            h_tau = h_tau[:, :30, :]
            h_tau_want.append(h_tau)
            h_freq_want.append(np.squeeze(h_freq))
            ue_pos_want.append(ue_pos)
            obj_pos_want.append(np.concatenate([
                curtain_pos,  # curtain position (x,y,z)
                [curtain_rot],  # curtain rotation (1,)
                cabinet_pos,  # cabinet position (x,y,z) 
                [cabinet_rot]   # cabinet rotation
            ]))
            
            scene.remove("rx")


    np.savez(f"data/IndoorChannelGenerator_scene{args.scene}.npz", cir = h_freq_want, ue_pos = ue_pos_want, obj_pos = obj_pos_want)