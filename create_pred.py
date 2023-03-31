import numpy as np
import nibabel as nib
from tqdm import tqdm


filenames = [
    "9908796_20041112_SAG_3D_DESS_LEFT_016610076606.nii.gz",
    "9902757_20041112_SAG_3D_DESS_RIGHT_016610077012.nii.gz",
    "9567704_20040319_SAG_3D_DESS_LEFT_016610013903.nii.gz"
    ]

"""
for file in tqdm(filenames):
    name = file[:-7]

    label = nib.load(f"/home/blg515/knee_data/test/labels/{file}").get_fdata()
    np.save(name + "_label", label)

    label = None

    pred = nib.load(f"/home/blg515/masters-thesis/predictions/{file}").get_fdata()
    pred = np.argmax(pred, axis=0)
    np.save(name + "_prediction", pred)

    pred = None
"""

# Verify files
for file in filenames:
    name = file[:-7]
    path = "/home/blg515/masters-thesis/"

    lab = np.load(path + name + "_label.npy")
    assert lab.shape == (384, 384, 160)
    lab = None

    pred = np.load(path + name + "_prediction.npy")
    assert pred.shape == (384, 384, 384)
    pred = None
finally:
    print("All files were correctly saved.")


"""

metrics = [
    { 'accuracy': [0.99965188, 0.99972418, 0.99925735, 0.99989556, 0.99995537,
       0.99982059, 0.99998661, 0.9995412 ],
       'precision': [0.9901622 , 0.9888606 , 0.96045198, 0.99340505, 0.99991515,
       0.96691511, 0.99999999, 0.99997306],
       'recall': [0.99396042, 0.97858268, 0.97964841, 0.99909846, 0.99585939,
       0.99644061, 0.89510489, 0.99953257],
       'specificity': [0.99977916, 0.99990547, 0.99950027, 0.99990677, 0.9999991 ,
       0.99983677, 1.        , 0.99965253],
       'dice': [0.99205767, 0.98369479, 0.96995522, 0.99624362, 0.99788315,
       0.98145585, 0.94464944, 0.99975277]},

    {'accuracy': [0.99999907, 1.        , 0.99965827, 0.99999907, 1.        ,
       0.99987523, 1.        , 0.9995335 ],
       'precision': [0.9997549 , 0.        , 0.62929293, 0.99999998, 0.        ,
       0.        , 0.999999  , 1.        ],
       'recall': [1.        , 0.        , 1.        , 0.97619045, 0.        ,
       0.        , 0.999999  , 0.99953143],
       'specificity': [0.99999907, 1.        , 0.99965808, 1.        , 1.        ,
       0.99987523, 1.        , 1.        ],
       'dice': [0.99987744, 0.        , 0.77247365, 0.9879518 , 0.        ,
       0.        , 0.9999995 , 0.99976566]},

       {'accuracy': [0.99945614, 0.9997411 , 0.99933257, 0.99947883, 0.99998823,
       0.99999832, 0.9999958 , 0.99962846],
       'precision': [0.99807161, 0.96147044, 0.95096207, 0.96613059, 1.        ,
       0.8947368 , 1.        , 1.        ],
       'recall': [0.98354937, 0.99960599, 0.96132044, 0.99859031, 0.99887387,
       0.99999994, 0.99824129, 0.99960016],
       'specificity': [0.99994196, 0.99974197, 0.99962224, 0.99949173, 1.        ,
       0.99999832, 1.        , 1.        ],
       'dice': [0.99075727, 0.98016742, 0.9561132 , 0.98209231, 0.99943662,
       0.94444442, 0.99911987, 0.99980004]}
]

label_keys = [
    "Lateral femoral cart.",
    "Lateral meniscus",
    "Lateral tibial cart.",
    "Medial femoral cartilage",
    "Medial meniscus",
    "Medial tibial cart.",
    "Patellar cart.",
    "Tibia"]

for file, metric in zip(filenames, metrics):
    print(file)
    print(f\"""
| Class | Accuracy | Precision | Sensitivity | Specificity | Dice |
| ----- | -------- | --------- | ----------- | ----------- | ---- |\""")
    for i, key in enumerate(label_keys):
        accuracy = round(metric["accuracy"][i], 4)
        precision = round(metric["specificity"][i], 4)
        sensitivity = round(metric["recall"][i], 4)
        specificity = round(metric["specificity"][i], 4)
        dice = round(metric["dice"][i], 4)

        print(f"| {key} | {accuracy} | {precision} | {sensitivity} | {specificity} | {dice} |")

    print()
"""