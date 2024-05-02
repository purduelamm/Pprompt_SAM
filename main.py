import time
import torch
import argparse
import cv2 as cv
import numpy as np
import pyrealsense2 as rs

from utils import *
from gravitonf import *


def parse_args(argv=None) -> None:
    parser = argparse.ArgumentParser(description='pointSAM')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use CUDA for SAM.')
    parser.add_argument('--img_path', default='./images/test/rgb063.png', type=str,
                        help='path to an image for segmentation.')
    parser.add_argument('--use_camera', default=False, type=bool,
                        help='user a camera [Realsense d435 is used].')
    parser.add_argument('--cad_path', default='./objects/obj_05.ply', type=str,
                        help='path to an object cad model.')
    parser.add_argument('--checkpoint', default='./weights/sam_vit_l_0b3195.pth', type=str,
                        help='path to a checkpoint.')
    parser.add_argument('--model_type', default='vit_l', type=str,
                        help='model_type for SAM corresponding to the checkpoint.')
    parser.add_argument('--save_imgs', default='./result', type=str,
                        help='path to save output images.')

    global args
    args = parser.parse_args(argv)


def click_event(event, x: int, y: int, flags, params) -> None: 
    if event == cv.EVENT_LBUTTONDOWN: 
        coord = np.array([x, y])
        prompt.append(coord)
  
        cv.circle(img, coord, 2, (0, 0, 255), 3, cv.LINE_AA) 
        cv.imshow('image', img) 
  
    return


def interactor(img: np.array) -> None:
    """ 
    Load an interactive window for a user to select points.
    
    Parameters
    ----------
    img : obj : 'np.array'
        original image file we choose to process using SAM
            
    Returns
    -------
    None
    """
    cv.imshow('image', img) 
    cv.setMouseCallback('image', click_event) 
    cv.waitKey(0) 
    cv.destroyAllWindows()     

    return


def PpromptSAM(args, img: np.array) -> np.array:
    """ 
    Generate masks using SAM
    
    Parameters
    ----------
    img : obj : 'np.array'
        original image file we choose to process using SAM
            
    Returns
    -------
    masks : obj : 'np.array'
        grayscale mask (0 ~ 255)
    """
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    dst = img.copy()
    interactor(img=img)

    if args.save_imgs:
        cv.imwrite(f'{args.save_imgs}/image_pts.png', img)

    mask_predictor = sam_loader(checkpoint=args.checkpoint, model_type=args.model_type, device=device)
    mask_predictor.set_image(dst)

    masks, _, _ = mask_predictor.predict(point_coords=np.array(prompt), point_labels=np.ones(len(prompt)), multimask_output=False)

    result, masks = segmentation_visualizer(dst, masks)

    if args.save_imgs:
        cv.imwrite(f'{args.save_imgs}/segmented.png', result)

    return masks


def d435_initializer(index: int = 0):
    pipeline = rs.pipeline()
    config = rs.config()
    colorizer = rs.colorizer()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    depth_sensor = device.first_depth_sensor()
    desired_preset_index = index  # Example index, replace with your choice
    depth_sensor.set_option(rs.option.visual_preset, desired_preset_index)

    rs_witdh, rs_heigt = 1280, 720
    config.enable_stream(rs.stream.depth, rs_witdh, rs_heigt, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs_witdh, rs_heigt, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    pc = rs.pointcloud()

    time.sleep(1)

    try:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        color_image = cv.normalize(color_image, None, 0, 255, cv.NORM_MINMAX)
        
        img = cv.addWeighted(depth_colormap, 0.5, color_image, 0.7, 0)

        cv.imshow('blend', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

        points = pc.calculate(depth_frame)
        pc.map_to(depth_frame)

    finally:
        pipeline.stop()

    return color_image, points


if __name__=="__main__":
    parse_args()
    prompt = []
    
    if args.use_camera:
        print("Initialize Realsense d435...")
        img, points = d435_initializer()
    else:
        assert args.img_path is not None, "image_path is not provided!!!"
        img = cv.imread(args.img_path, 1)

    masks = PpromptSAM(args=args, img=img)

    pts = gen_pointcloud(points=points, segment_img=masks)
