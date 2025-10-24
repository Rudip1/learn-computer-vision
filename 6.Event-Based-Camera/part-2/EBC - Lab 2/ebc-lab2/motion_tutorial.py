"""
Tutorial on how to apply motion compensation.
"""

import cv2
import h5py
import tonic
import numpy as np


def keep_windows_open() -> None:
    """
    Keeps visualization windows open till 'ESC' key press.
    """
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            break


# ===== Tutorial Code ===== #
if __name__ == "__main__":
    # Data reader imports
    from ewiz.core.reader import read_data, read_gt, clip_data
    from ewiz.core.sync import GroundTruthSynchronizer

    # ===== Replace variables here ===== #
    """
    Fill the empty string cells below. Choose your data directories, and '.hdf5' groups as
    stated in the lab's given.
    """
    # Directory and clip variables
    data_dir = "/home/chere/Academics/Hands_on_perception/lab2_ebc2/indoor_flying1_data.hdf5"
    gt_dir = "/home/chere/Academics/Hands_on_perception/lab2_ebc2/indoor_flying1_gt-002.hdf5"

    clip = [10, 10.2]

    # ----- Read dataset ----- #
    data_file = h5py.File(data_dir, "r")
    data = read_data(
        data_file=data_file,
        events_group=["davis", "left", "events"],
        nearest_events_group=["davis", "left", "image_raw_event_inds"],
        grayscale_images_group=["davis", "left", "image_raw"],
        grayscale_timestamps_group=["davis", "left", "image_raw_ts"],
        reset_time=False
    )
    data_file = None

    events = data[0]
    nearest_events = data[1]
    grayscale_images = data[2]
    grayscale_timestamps = data[3]

    # Save start time
    time_idx = events[0, 2]

    # Clip data
    data = clip_data(
        start_time=clip[0] + time_idx,
        end_time=clip[1] + time_idx,
        events=events,
        nearest_events=nearest_events,
        grayscale_images=grayscale_images,
        grayscale_timestamps=grayscale_timestamps
    )

    events = data[0]
    nearest_events = data[1]
    grayscale_images = data[2]
    grayscale_timestamps = data[3]
    # ------------------------ #

    # ----- Read Ground Truth Flows ----- #
    gt_file = h5py.File(gt_dir, "r")
    gt = read_gt(
        gt_file=gt_file,
        flow_group=["davis", "left", "flow_dist"],
        timestamps_group=["davis", "left", "flow_dist_ts"]
    )
    gt_flows = gt[0]
    gt_timestamps = gt[1]
    gt_synchronizer = GroundTruthSynchronizer(flows=gt_flows, timestamps=gt_timestamps)
    gt_flow = gt_synchronizer.sync(
        start_time=clip[0] + time_idx,
        end_time=clip[1] + time_idx
    )
    """
    NOTE: BELOW IS THE GROUND TRUTH FLOW YOU WILL BE USING FOR VALIDATION.
    THE FLOW HAS A SHAPE OF (2, H, W), THE FIRST INDEX OF DIMENSION 0 IS THE FLOW IN X,
    THE SECOND INDEX OF DIMENSION 0 IS THE FLOW IN Y.
    """
    gt_flow = gt_flow/(events[-1, 2] - events[0, 2])
    # ----------------------------------- #
    # ================================== #

    # ===== Apply motion compensation ===== #
    """
    NOTE: BELOW IS THE PREDICTED FLOW GIVEN BY MOTION COMPENSATION.
    THE FLOW HAS A SHAPE OF (2, H, W), THE FIRST INDEX OF DIMENSION 0 IS THE FLOW IN X,
    THE SECOND INDEX OF DIMENSION 0 IS THE FLOW IN Y.
    """
    comp_flow = None

    from ewiz.losses.loss import MotionCompensationLoss
    from ewiz.solvers.pyramidal import PyramidalPatchMotionCompensation

    # ----- Crop images ----- #
    """
    Before applying the algorithm we have to crop the images into square equivalents.
    We choose an output size of (256, 256).
    """
    from ewiz.transforms.events import EventsCenterCropUns
    from ewiz.transforms.flow import FlowCenterCrop

    events_transforms = tonic.transforms.Compose([
        EventsCenterCropUns(sensor_size=(346, 260), out_size=(256, 256))
    ])
    flow_transforms = tonic.transforms.Compose([
        FlowCenterCrop(out_size=(256, 256))
    ])

    # Transformed flow and events
    events = events_transforms(events)
    gt_flow = flow_transforms(gt_flow)
    # ----------------------- #

    # Loss function initialization
    loss_function = MotionCompensationLoss(
        image_size=(256, 256),
        losses=[
            "multifocal_normalized_image_variance",
            "multifocal_normalized_gradient_magnitude",
            "regularizer"
        ],
        weights=[1.0, 1.0, 0.01],
        gt_flow=gt_flow
    )

    # Initialize optimizer
    pyramidal_optimizer = PyramidalPatchMotionCompensation(
        image_size=(256, 256),
        optimizer="BFGS",
        init_method="random",
        loss_function=loss_function,
        scale_range=(4, 5)
    )

    # Run optimization
    patch_flows, optimizer_results = pyramidal_optimizer.optimize(events=events)
    # ===================================== #

    # ===== Metric Evaluation ===== #
    """
    IMPLEMENT THE AEE METRIC FUNCTION HERE.
    NOTE: REMEMBER BELOW IS THE PREDICTED FLOW.
    """
    comp_flow = pyramidal_optimizer.dense_flow.cpu().detach().numpy()

    def get_average_endpoint_error(
        gt_flow: np.ndarray = None,
        comp_flow: np.ndarray = None
    ) -> float:
        """
        Compute the Average Endpoint Error.

        Parameters
        ----------
        gt_flow : np.ndarray
            Ground truth flow of shape (2, H, W).
        comp_flow : np.ndarray
            Compensated flow of shape (2, H, W).
        """
        # Compute over valid points in ground truth data
        mask_flow = np.logical_and(
            np.logical_and(~np.isinf(gt_flow[[0], ...]), ~np.isinf(gt_flow[[1], ...])),
            np.logical_and(np.abs(gt_flow[[0], ...]) > 0, np.abs(gt_flow[[1], ...]) > 0)
        )
        gt_flow = gt_flow*mask_flow
        comp_flow = comp_flow*mask_flow

        # ==============================
        diff = gt_flow - comp_flow
        squared = diff[0]**2 + diff[1]**2
        aee = np.sum(np.sqrt(squared)) / np.sum(mask_flow[0])  # Use one mask channel for count
        return aee

        # ==============================

    # Compute AEE here
    aee = None
    aee = get_average_endpoint_error(gt_flow=gt_flow, comp_flow=comp_flow)
    loss = loss_function.loss
    print("Metrics,", "LOSS:", loss, "AEE:", aee)
    # ============================= #
    keep_windows_open()
