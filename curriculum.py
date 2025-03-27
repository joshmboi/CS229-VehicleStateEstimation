from save_csv import CSV_Saver
from train import Trainer
from test import Tester

# set surface, space, and version
surf = "dry"
space = "red"
ver = 4

state_save_fields = [
    "axCG_mps2",
    "ayCG_mps2",
    "azCG_mps2",
    "pitchAngle_rad",
    "pitchRate_radps",
    "posE_m",
    "posN_m",
    "posU_m",
    "rollAngle_rad",
    "rollRate_radps",
    "sideSlip_rad",
    "vxCG_mps",
    "vyCG_mps",
    "yawAngle_rad",
    "yawRate_radps"
]

control_save_fields = [
    "LFwheelSpeed_mps",
    "LRwheelSpeed_mps",
    "RFwheelSpeed_mps",
    "RRwheelSpeed_mps",
    "brakePressureFL_bar",
    "brakePressureFR_bar",
    "brakePressureRL_bar",
    "brakePressureRR_bar",
    "engineTorque_Nm",
    "massEstimate_kg",
    "pinionAngle_rad"
]

fields = [
    [
        [
            "axCG_mps2",
            "ayCG_mps2",
            "sideSlip_rad",
            "vxCG_mps",
            "vyCG_mps",
            "yawRate_radps"
        ],
        [
            "LFwheelSpeed_mps",
            "LRwheelSpeed_mps",
            "RFwheelSpeed_mps",
            "RRwheelSpeed_mps",
            "brakePressureFL_bar",
            "brakePressureFR_bar",
            "brakePressureRL_bar",
            "brakePressureRR_bar",
            "engineTorque_Nm",
            "pinionAngle_rad"
        ]
    ]
]

for i in range(len(fields)):
    print(i)
    csv_saver = CSV_Saver(space, fields[i][0], fields[i][1])
    csv_saver.save_csvs()

    trainer = Trainer(surf, space)

    nn_train_losses, nn_val_losses = trainer.train_nn()
    xgb_train_losses, xgb_val_losses = trainer.train_xgb()
    lstm_train_losses, lstm_val_losses = trainer.train_lstm()
    trainer.save_models()

    tester = Tester(surf, space, ver)

    tester.plot_traj(f"./sims/sim_{i}.png")
