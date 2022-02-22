import os
import h5py
from click import progressbar
import numpy as np

def read_skeleton_file(file_path: str):
    with open(file_path, 'r') as f:
        skeleton_sequence = {'numFrame': int(f.readline()), 'frameInfo': []}
        for _ in range(skeleton_sequence['numFrame']):
            frame_info = {'numBody': int(f.readline()), 'bodyInfo': []}
            for _ in range(frame_info['numBody']):
                body_info_key = [
                    'bodyID', 'clippedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for _ in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence

def parse_skeleton_data(file_path: str, joints: int = 25, persons = 2) -> None:
    
    sequence_data = read_skeleton_file(file_path)
    data = np.zeros((3, sequence_data['numFrame'], joints, persons), dtype=np.float32)
    
    for frame_number, frame in enumerate(sequence_data['frameInfo']):
        for body_number, body in enumerate(frame['bodyInfo']):
            for joint_number, joint in enumerate(body['jointInfo']):
                if body_number < persons and joint_number < joints:
                    data[:, frame_number, joint_number, body_number] = [joint['x'], joint['y'], joint['z']]

    return np.around(data, decimals=3)

def generate_skeleton_dataset(data_path: str, output_path: str) -> None:

    target_file_path = f"{output_path}/skeleton.h5"

    with h5py.File(target_file_path, "w") as target_file:
        progress_bar = progressbar(iterable=None, length=len(next(os.walk(data_path))[2]))
        for file_name in os.listdir(data_path):
            sequence_name = os.path.splitext(file_name)[0]

            skeleton_data = parse_skeleton_data(f"{data_path}/{file_name}")
            
            f = open(output_path + "log.txt", "w+")
            f.write(sequence_name)
            f.write("\r\n")
            f.close()

            target_file.create_group(sequence_name).create_dataset("skeleton", data=skeleton_data)
            progress_bar.update(1)
