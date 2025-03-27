import os
import scipy
import pandas as pd


class CSV_Saver:
    def __init__(
            self, space, state_save_fields=None, control_save_fields=None
    ):
        # set name of reduced state space files
        self.space = space

        # set save_fields and control_fields
        self.state_save_fields = state_save_fields
        self.control_save_fields = control_save_fields

        if not self.state_save_fields:
            self.state_save_fields = [
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

        if not self.control_save_fields:
            self.control_save_fields = [
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

    def data_from_mat(self, filepath):
        """
        Gets only the data that we want from the mat file.

        Params:
        filepath (string): path to mat file

        Returns:
        state (dict): dict with desired fields as keys and data as values
        control (dict): dict with desired fields as keys and data as values
        """

        # load in mat file
        mat = scipy.io.loadmat(filepath)

        # state variable and control variable
        state_var = mat["OxTSData"]
        control_var = mat["vehicleData"]

        # determine smallest dimension to match lengths of fields for both
        small_dim = len(state_var[self.state_save_fields[0]][0][0][0])
        for field in self.state_save_fields:
            small_dim = min(
                small_dim, len(state_var[field][0][0][0])
            )

        for field in self.control_save_fields:
            small_dim = min(
                small_dim, len(control_var[field][0][0][0])
            )

        # state and control dataframes
        state = pd.DataFrame(
            index=range(small_dim),
            columns=self.state_save_fields
        )

        control = pd.DataFrame(
            index=range(small_dim),
            columns=self.control_save_fields
        )

        # reassign values in each column for state and control
        for field in self.state_save_fields:
            div = len(state_var[field][0][0][0]) // small_dim
            state[field] = state_var[field][0][0][0][::div][:small_dim]

        for field in self.control_save_fields:
            div = len(control_var[field][0][0][0]) // small_dim
            control[field] = control_var[field][0][0][0][::div][:small_dim]

        return state, control

    def save_csvs(self):
        # get current working directory
        cur_path = os.getcwd()

        # get Vehicle Data dir and csv data dir
        vehicle_data_dir = os.path.join(cur_path, "Vehicle Data")
        out_data_dir = os.path.join(cur_path, "data")

        # get all data directories in "Vehicle Data" directory
        data_dirs = [
            dir for dir in os.listdir(vehicle_data_dir)
            if os.path.isdir(vehicle_data_dir)
        ]

        # get mat files for each data directory (Ice, Dry, Wet)
        for dir in data_dirs:
            # get data dir path
            mat_dir = os.path.join(vehicle_data_dir, dir)

            # get all mat files within each surface type
            mat_files = [
                    f for f in os.listdir(mat_dir)
                    if (
                        os.path.isfile(os.path.join(mat_dir, f)) and
                        os.path.join(mat_dir, f).lower().endswith("mat")
                    )
            ]

            # get state and control data from mat file
            for i, mat_file in enumerate(mat_files):
                state, control = self.data_from_mat(
                    os.path.join(mat_dir, mat_file)
                )

                dir_name = dir.lower().replace(" ", "_")

                state_filename = f"{dir_name}_state_{self.space}_{i + 1}.csv"
                ctrl_filename = f"{dir_name}_control_{self.space}_{i + 1}.csv"

                state.to_csv(
                    os.path.join(out_data_dir, state_filename),
                    index=False
                )

                control.to_csv(
                    os.path.join(out_data_dir, ctrl_filename),
                    index=False
                )
            print(f"saved {dir.lower()}")
