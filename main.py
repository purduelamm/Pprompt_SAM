import torch
import argparse
import cv2 as cv
import numpy as np

from utils import *


def parse_args(argv=None) -> None:
    parser = argparse.ArgumentParser(description='pointSAM')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use CUDA for SAM.')
    parser.add_argument('--img_path', default='./images/test/rgb063.png', type=str,
                        help='path to an image for segmentation.')
    parser.add_argument('--checkpoint', default='./weights/sam_vit_h_4b8939.pth', type=str,
                        help='path to a checkpoint.')
    parser.add_argument('--model_type', default='vit_h', type=str,
                        help='model_type for SAM corresponding to the checkpoint.')
    parser.add_argument('--save_imgs', default='./images/result', type=str,
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


if __name__=="__main__":
    parse_args()
    prompt = []
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    img = cv.imread(args.img_path, 1) 
    dst = img.copy()
    interactor(img=img)

    if args.save_imgs:
        cv.imwrite(f'{args.save_imgs}/image_pts.png', img)

    mask_predictor = sam_loader(checkpoint=args.checkpoint, model_type=args.model_type, device=device)
    mask_predictor.set_image(dst)

    masks, scores, logits = mask_predictor.predict(point_coords=np.array(prompt), point_labels=np.ones(len(prompt)), multimask_output=False)

    result = segmentation_visualizer(dst, masks)

    if args.save_imgs:
        cv.imwrite(f'{args.save_imgs}/segmented.png', result)