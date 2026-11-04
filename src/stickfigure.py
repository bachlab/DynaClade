# ==============================================================================
# Author: Ulises Daniel Serratos Hernandez
# Contact: ulises.serratos@ucl.ac.uk
# ==============================================================================

"""
Attributes
---
name: str: 
        identifier label name for stickfigure variable
body: dict: 
        where ->
                key: str: segment name
                values: list: markers IDs/Labels defining segments,
                        this names are expected in data set unless 
                        remapped (explained in next 2 args)
data_mapping: dict: (OPTIONAL)
        where ->
                key: str: marker ID/label
                values: str: marker ID/label 
                       
"""

stickfigures: dict = {
    "coda": {
        "name": "coda",
        "body": {
            "RShoulder": ["Neck", "RSho"],
            "LShoulder": ["Neck", "LSho"],
            "RUpperarm": ["RSho", "RElb"],
            "LUpperarm": ["LSho", "LElb"],
            "RForearm": ["RElb", "RWri"],
            "LForearm": ["LElb", "LWri"],
            "Back": ["Neck", "MHip"],
            "RThigh": ["MHip", "RKne"],
            "LThigh": ["MHip", "LKne"],
            "RShin": ["RKne", "RAnk"],
            "LShin": ["LKne", "LAnk"],
        },
        "data_mapping": {
            "MHip": "6",
            "RWri": "10",
            "RElb": "11",
            "RSho": "12",
            "LWri": "16",
            "LElb": "17",
            "LSho": "18",
            "Neck": "22",
            "RAnk": "25",
            "RKne": "26",
            "LAnk": "27",
            "LKne": "28",
        },
    },
    "openpose": {
        "name": "openpose",
        "body": {
            "RShoulder": ["Neck", "RSho"],
            "LShoulder": ["Neck", "LSho"],
            "RUpperarm": ["RSho", "RElb"],
            "LUpperarm": ["LSho", "LElb"],
            "RForearm": ["RElb", "RWri"],
            "LForearm": ["LElb", "LWri"],
            "Back": ["Neck", "MHip"],
            "RThigh": ["MHip", "RKne"],
            "LThigh": ["MHip", "LKne"],
            "RShin": ["RKne", "RAnk"],
            "LShin": ["LKne", "LAnk"],
        },
    },
    "qualisys_tsv": {
        "name": "qualisys_tsv",
        "body": {
            "RShoulder": ["Neck", "RSho"],
            "LShoulder": ["Neck", "LSho"],
            "RUpperarm": ["RSho", "RElb"],
            "LUpperarm": ["LSho", "LElb"],
            "RForearm": ["RElb", "RWri"],
            "LForearm": ["LElb", "LWri"],
            "Back": ["Neck", "MHip"],
            "RThigh": ["MHip", "RKne"],
            "LThigh": ["MHip", "LKne"],
            "RShin": ["RKne", "RAnk"],
            "LShin": ["LKne", "LAnk"],
        },
        "data_mapping": {
            "MHip": "Hips",
            "RWri": "RightHand",
            "RElb": "RightForeArm",
            "RSho": "RightArm",
            "LWri": "LeftHand",
            "LElb": "LeftForeArm",
            "LSho": "LeftArm",
            "Neck": "Spine2",
            "RAnk": "RightFoot",
            "RKne": "RightLeg",
            "LAnk": "LeftFoot",
            "LKne": "LeftLeg",
        },
    },
    "qualisys_csv": {
        "name": "qualisys_csv",
        "body": {
            "RShoulder": ["Neck", "RSho"],
            "LShoulder": ["Neck", "LSho"],
            "RUpperarm": ["RSho", "RElb"],
            "LUpperarm": ["LSho", "LElb"],
            "RForearm": ["RElb", "RWri"],
            "LForearm": ["LElb", "LWri"],
            "Back": ["Neck", "MHip"],
            "RThigh": ["MHip", "RKne"],
            "LThigh": ["MHip", "LKne"],
            "RShin": ["RKne", "RAnk"],
            "LShin": ["LKne", "LAnk"],
        },
        "data_mapping": {
            "MHip": "Hips",
            "RWri": "RHand",
            "RElb": "RForeArm",
            "RSho": "RArm",
            "LWri": "LHand",
            "LElb": "LForeArm",
            "LSho": "LArm",
            "Neck": "Spine2",
            "RAnk": "RFoot",
            "RKne": "RLeg",
            "LAnk": "LFoot",
            "LKne": "LLeg",
        },
    },
    "template": {
        # === FEEL FREE TO MODIFY NAME === #
        "name": "template",
        # === DO NOT MODIFY HERE, THIS MARKERS ARE EXPECTED === #
        "body": {
            "RShoulder": ["Neck", "RSho"],
            "LShoulder": ["Neck", "LSho"],
            "RUpperarm": ["RSho", "RElb"],
            "LUpperarm": ["LSho", "LElb"],
            "RForearm": ["RElb", "RWri"],
            "LForearm": ["LElb", "LWri"],
            "Back": ["Neck", "MHip"],
            "RThigh": ["MHip", "RKne"],
            "LThigh": ["MHip", "LKne"],
            "RShin": ["RKne", "RAnk"],
            "LShin": ["LKne", "LAnk"],
        },
        # === MODIFY DICT VALUES IF REMAPPING IS NEEDED === #
        "data_mapping": {
            "MHip": "MHip",
            "RWri": "RWri",
            "RElb": "RElb",
            "RSho": "RSho",
            "LWri": "LWri",
            "LElb": "LElb",
            "LSho": "LSho",
            "Neck": "Neck",
            "RAnk": "RAnk",
            "RKne": "RKne",
            "LAnk": "LAnk",
            "LKne": "LKne",
        },
    },
}
