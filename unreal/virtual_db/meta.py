# Some meta data for conversion, etc.
# TODO: need a way to match skeleton

# personnel / girl / trooper
skel1 = [
    "foot_r", "calf_r", "thigh_r", "thigh_l", "calf_l",
    "foot_l", "pelvis", "neck_01", None, None,
    "hand_r", "lowerarm_r", "upperarm_r", "upperarm_l", "lowerarm_l",
    "hand_l"
]
skel_edge = [
    (0,1), (1,2), # right leg
    (3,4), (4,5), # left leg
    (6,7), (7,8), (8,9), # body
    (10,11), (11,12), # right arm
    (13,14), (14,15), # left arm
    (2,6), (3,6), # leg to body
    (12,7), (13,7), # arm to body
]

# eric / alison
human_skel_list = [
    "foot_r", "lowerleg_r", "upperleg_r", "upperleg_l", "lowerleg_l", 
    "foot_l", "hip", "neck", "jaw_end", "head_end",
    "hand_r", "lowerarm_r", "upperarm_r", "upperarm_l", "lowerarm_l", 
    "hand_l"
]

# vision dataset, which one?
ref_skel = [
    'r ankle', 'r knee', 'r hip', 'l hip', 'l knee', 
    'l ankle', 'pelvis', 'thorax', 'upper neck', 'head top', 
    'r wrist', 'r elbow', 'r shoulder', 'l shoulder', 'l elbow', 
    'l wrist'
]

# arm_skel = [
#     'M1', 'Upper', 'Lower', 'Hand'
# ]

arm_skel = [
    'Rotation', 'Base', 'Elbow', 'Wrist'
]

arm_skel_edge = [
    (0,1), (1,2), (2,3)
]