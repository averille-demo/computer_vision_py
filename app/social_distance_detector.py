"""
Computer vision module to flag social distancing violations based on pixel distance.
"""
import os
import time
from pathlib import Path
from typing import List, Tuple
import cv2
from cv2.dnn import blobFromImage, NMSBoxes, readNetFromDarknet, DNN_BACKEND_CUDA, DNN_TARGET_CUDA
import numpy as np
from scipy.spatial import distance as dist
from requests import get
from vision_lib import init_logger, parse_cmd_args, get_timestamp, get_files_by_keyword, limit_path

MODULE = Path(__file__).resolve().stem
CWD_PATH = Path.cwd()


def check_dependencies(yolo_paths: list) -> bool:
    """
    Download necessary dependencies (names, config, weights) prior to running pipeline

    names: 80 different class names used in the COCO dataset (common objects)
    cfg: deep neural network configuration
    weights: pre-trained deep neural network weights

    Return:
        bool: if all dependencies exist in ./yolo-coco directory
    """
    file_url_map = {
        "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
        "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
    }
    if yolo_paths and len(yolo_paths) > 0 and isinstance(yolo_paths[0], Path):
        is_config_ready = dict.fromkeys([path.name for path in yolo_paths], False)
        parent_dir = yolo_paths[0].parent
        # create 'yolo-coco' directory if it does not exist
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
        # compare filenames from pipeline to url_map (pathlib.name)
        yolo_filenames = [yp.name for yp in yolo_paths]
        if set(yolo_filenames) == set(file_url_map.keys()):
            for yolo_path in yolo_paths:
                if not yolo_path.is_file():
                    yolo_url = f"{file_url_map[yolo_path.name]}"
                    logger.info(f"\tdownloading missing: {yolo_url}")
                    resp = get(f"{yolo_url}")
                    if resp.status_code == 200:
                        with open(file=str(yolo_path), mode="wb") as fp:
                            fp.write(resp.content)
                            logger.info(f"SUCCESS: downloaded '{yolo_path}'")
                # validation check if all configuration files exist with data
                if yolo_path.is_file() and os.stat(yolo_path).st_size > 0:
                    is_config_ready[yolo_path.name] = True
            return all(is_config_ready.values())
    else:
        logger.error(f"invalid input: {yolo_paths}")
    return False


def cleanup_prior(path: Path, file_ext: str = ".mp4") -> None:
    """
    Delete prior output if it exists in directory.

    Args:
        path (pathlib.Path): directory location of prior output files
        file_ext (str): '.mp4' video file extension

    Returns:
        None: results printed to console
    """
    if isinstance(path, Path):
        try:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            src_paths = [
                p.absolute() for p in sorted(path.glob(f"*{file_ext}")) if p.is_file()
            ]
            if len(src_paths) > 0:
                for src_path in src_paths:
                    src_path.unlink(missing_ok=True)
                logger.info(msg=f"found ({len(src_paths)}) '{file_ext}'"
                                f" file(s) in: {limit_path(path)}")
        except PermissionError:
            logger.exception(msg=f"{path.name}")


def detect_people(
        image: np.ndarray,
        score_threshold: float = 0.3,
        nms_threshold: float = 0.3,
) -> List:
    """
    YOLO3 object detector on input video frame.

    Args:
        image (np.ndarray): numpy array with pixel data
        score_threshold (float): minimum confidence
        nms_threshold (float): non-maximum suppression threshold parameter

    Returns:
        List: categorized images with human objects
    """
    (H, W) = image.shape[:2]
    results = []
    blob = blobFromImage(
        image=image,
        scalefactor=(1.0 / 255.0),
        size=(416, 416),
        swapRB=True,
        crop=False
    )
    dnn.setInput(blob)
    layer_outputs = dnn.forward(names)
    bboxes = []
    centroids = []
    confidence_scores = []
    person_idx = labels.index("person")
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = float(scores[class_id])
            if (class_id == person_idx) and (confidence > score_threshold):
                box = detection[0:4] * np.array([W, H, W, H])
                (center_x, center_y, width, height) = box.astype("int")
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))
                bboxes.append([x, y, int(width), int(height)])
                centroids.append((center_x, center_y))
                confidence_scores.append(confidence)
    indices = NMSBoxes(
        bboxes=bboxes, scores=confidence_scores,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
    )
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (bboxes[i][0], bboxes[i][1])
            (w, h) = (bboxes[i][2], bboxes[i][3])
            r = (confidence_scores[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)
    return results


def setup_network() -> Tuple:
    """
    Setup deep neural network with pre-trained human/person model.

    Returns:
        darknet (readNetFromDarknet): pre-trained network stored in Darknet model
        coco_labels (List): COCO names of objects (person, bicycle, car, etc.)
        outer_layer_names (List): names of outer network layers
    """
    labels_path = Path(CWD_PATH, "yolo-coco", "coco.names")
    config_path = Path(CWD_PATH, "yolo-coco", "yolov3.cfg")
    weights_path = Path(CWD_PATH, "yolo-coco", "yolov3.weights")

    # check environment and download missing configuration files
    if check_dependencies(yolo_paths=[labels_path, config_path, weights_path]):
        logger.info(f"loading YOLO weights and model configuration:"
                    f" ({cmd_args['pixel']}-pixel distance)")
        with open(file=labels_path, mode='r', encoding='utf-8') as fp:
            coco_labels = fp.read().strip().split("\n")

        darknet = readNetFromDarknet(str(config_path), str(weights_path))
        if cmd_args["cuda"]:
            try:
                # requires custom OpenCV CMake install
                logger.info("*CUDA-GPU* processing selected")
                # for CMake with OPENCV_DNN_CUDA, WITH_CUDA and WITH_CUDNN
                darknet.setPreferableBackend(DNN_BACKEND_CUDA)
                darknet.setPreferableTarget(DNN_TARGET_CUDA)
            except AttributeError:
                logger.exception(msg="OPENCV_EXTRA_MODULES_PATH not configured "
                                     "reverting to *CPU-only* processing")
        else:
            logger.info("*CPU-only* processing selected")
        outer_layer_names = []
        # all layers of network
        dnn_layer_names = darknet.getLayerNames()
        # [200 227 254] ~ last layers of network
        uc_outer_layers = darknet.getUnconnectedOutLayers()
        # 199=yolo_82, 226=yolo_94, 253=yolo_106
        for uc_outer_layer in uc_outer_layers:
            outer_layer_names.append(dnn_layer_names[uc_outer_layer - 1])
    return darknet, coco_labels, outer_layer_names


def reduce_image_dim(
        reduction_percent: float = 1.0,
        width: int = 600,
        height: int = 400,
) -> Tuple:
    """
    Resize frame by scaled reduction percentage

    Args:
        reduction_percent (float): percentage to reduce image size
        width (int): number of pixels across horizontal dimension of image (N-columns)
        height (int): number of pixels across vertical dimension of image (N-rows)

    Returns:
        n_width_cols, n_height_rows (tuple): scaled/rounded dimensions of original image
    """
    n_width_cols = int(width)
    n_height_rows = int(height)
    if 0.1 <= float(reduction_percent) < 1.0:
        n_width_cols = int(width * reduction_percent)
        n_height_rows = int(height * reduction_percent)
    return n_width_cols, n_height_rows


def process_video(
        src_path: Path,
        dst_path: Path,
) -> bool:
    """
    process video with social distancing detector based on euclidean distance.

    Args:
        src_path (pathlib.Path): input file path location of source video
        dst_path (pathlib.Path): output directory path for tagged/processed video

    Returns:
        bool: true if video processed without exception
    """
    try:
        logger.info(f"starting: {limit_path(src_path)}")
        capture = cv2.VideoCapture(str(src_path))

        dst_width, dst_height = reduce_image_dim(
            reduction_percent=cmd_args['reduction_percent'],
            width=capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )
        frame_size = f"[{dst_width}x{dst_height}]"
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            f"size: {frame_size}\t ({total_frames:03} frames "
            f"@{capture.get(cv2.CAP_PROP_FPS)}fps)\tCUDA_GPU: {cmd_args['cuda']}"
        )
        # dynamically create output video filenames
        dst_path = Path(dst_path, f"{src_path.stem}_output{src_path.suffix}")

        # write the video to disk (convert pathlib object to string)
        writer = cv2.VideoWriter(
            str(dst_path),
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=25,
            frameSize=(dst_width, dst_height),
            isColor=True,
        )
        frame_num = 0
        while capture.isOpened():
            (status, frame) = capture.read()
            if not status:
                # finished all video frames
                break
            frame_num += 1
            # resize frame dimensions (if needed)
            if 10.0 <= cmd_args['reduction_percent'] < 100.0:
                frame = cv2.resize(
                    frame, (dst_width, dst_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            results = detect_people(image=frame)
            violation_set = set()
            if len(results) >= 2:
                centroids = np.array([r[2] for r in results])
                euclidean_dist = dist.cdist(
                    centroids, centroids, metric="euclidean"
                )
                for i in range(0, euclidean_dist.shape[0]):
                    for j in range(i + 1, euclidean_dist.shape[1]):
                        if euclidean_dist[i, j] < cmd_args['pixel']:
                            violation_set.add(i)
                            violation_set.add(j)

            # add colored rectangles around human/person objects
            for i, (prob, bbox, centroid) in enumerate(results, start=0):
                (start_x, start_y, end_x, end_y) = bbox
                # (cent_x, cent_y) = centroid
                # color=BGR format: distance OK = GREEN
                color = (0, 220, 0)
                if i in violation_set:
                    # color=BGR format: distance violation = RED
                    color = (0, 0, 220)
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 1)
                # add confidence percentage to label
                cv2.putText(
                    img=frame,
                    text=f"person: {prob * 100:0.2f}%",
                    org=(start_x, start_y - 3),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.25,
                    color=(128, 128, 128),
                    thickness=1,
                )

            # overlay colored text on video frames
            cv2.putText(
                img=frame,
                text=f"frame: [{frame_num:03} of {total_frames:03}]",
                # top left of frame
                org=(8, 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.70,
                color=(200, 30, 30),
                thickness=1,
            )
            cv2.putText(
                img=frame,
                text=f"violations: {len(violation_set):02}",
                # bottom left of frame
                org=(8, frame.shape[0] - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.70,
                color=(30, 30, 200),
                thickness=2,
            )
            if frame_num % 20 == 0:
                logger.info(f"frame: [{frame_num:04d} of {total_frames:04d}]\t"
                            f"'{src_path.name}'\t "
                            f"{int((frame_num / total_frames) * 100):02d}% complete")
            writer.write(frame)
        logger.info(f"completed: {limit_path(dst_path)}")
        capture.release()
        writer.release()
        cv2.destroyAllWindows()
        return True
    except cv2.error:
        logger.exception(msg=f"{src_path.name}")
    return False


def run_pipeline() -> None:
    """
    Step 1: remove prior output '.mp4' files
    Step 2: find relevant source video files (by keyword and extension)
    Step 3: setup deep neural network (dnn) using pre-trained darknet model
    Step 4: process video frames based on euclidean distance
    Step 5: export output videos with colored boxes:
          RED boxes: social distance between individuals TOO_CLOSE
        GREEN boxes: social distance between individuals OK
    """
    logger.info(f"{MODULE} triggered: {get_timestamp()}")
    timer = time.perf_counter()
    cleanup_prior(path=cmd_args["output_path"], file_ext=cmd_args["extension"])
    src_paths = get_files_by_keyword(
        path=cmd_args["input_path"],
        keyword=cmd_args["keyword"],
    )
    # track progress: initialize all paths to False
    is_processed = dict.fromkeys([path.name for path in src_paths], False)
    for i, src_path in enumerate(src_paths, start=1):
        logger.info(f"processing: [{i:02d} of {len(src_paths):02d}] "
                    f"'{limit_path(src_path)}'")
        is_processed[src_path.name] = process_video(
            src_path=src_path,
            dst_path=cmd_args["output_path"],
        )
    if all(is_processed.values()):
        logger.info("SUCCESS: all video processing complete")
    else:
        logger.error(f"FAILURE: video processing incomplete {is_processed}")
    logger.info(f"{MODULE} finished in {time.perf_counter() - timer:0.2f} seconds")


if __name__ == "__main__":
    logger = init_logger(log_name="opencv_sdd")
    cmd_args = parse_cmd_args()
    dnn, labels, names = setup_network()
    run_pipeline()
