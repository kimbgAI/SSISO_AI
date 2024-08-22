### Import Libraries ###
import os
import torch
from monai.transforms import (
    AsDiscrete, AsDiscreted
)


### Utilities ###
import nibabel as nib
import nrrd
import numpy as np

import monai
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose,
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged,
    Invertd, AsDiscreted, SaveImaged,
)
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference


def GetModelWithCP():

    model = SwinUNETR(
        in_channels=CFG.ImgInfo['Channel'],
        out_channels=len(CFG.MaskInfo),
        img_size=CFG.ImgInfo['Size'],
        feature_size=CFG.ModelInfo['FeatureSize'],
        use_checkpoint=True,
        use_v2=True
    ).to(CFG.Device)

    checkpoint = torch.load(CFG.ModelInfo['InferenceWeights'], map_location=CFG.Device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    del checkpoint

    return model



def PreTransforms():

    return Compose(
        [
            LoadImaged(keys=["image",]),
            EnsureChannelFirstd(keys=["image",]),
            Orientationd(keys=["image", ], axcodes="RAS"),
            Spacingd(keys=["image", ], pixdim=CFG.PreProcessing['PixDim'], mode=("bilinear", )),

            ScaleIntensityRanged(
                keys=["image"],
                a_min=CFG.PreProcessing['IntensityScale']['a_min'],
                a_max=CFG.PreProcessing['IntensityScale']['a_max'],
                b_min=CFG.PreProcessing['IntensityScale']['b_min'],
                b_max=CFG.PreProcessing['IntensityScale']['b_max'],
                clip=True,
            ),
        ]
    )



def Inference(model, input_path):

    img_dict = [{'image':input_path}]
    val_org_ds = Dataset(data=img_dict, transform=PreTransforms())
    val_org_loader = DataLoader(val_org_ds, batch_size=CFG.BatchSize, 
                                num_workers=CFG.NumWorkers)

    with torch.no_grad():
        for batch in val_org_loader:
            inputs = batch['image'].to(CFG.Device)
            batch['pred'] = sliding_window_inference(
                inputs, CFG.ImgInfo['Size'], CFG.PreProcessing['SlidingWindow']['BS'], 
                model, #overlap=0.7
            )
            # _ = [PostTransforms(save_path)(i) for i in decollate_batch(batch)]

    return batch



def PostTransforms():
    return Compose(
        [
            Invertd(
                keys="pred",
                transform=PreTransforms(),
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
                device="cpu",
            ),
            CFG.PostPred2Save, 
        ]
    )



def write_to_nrrd(img_array, img_affine, nrrd_file_path):

    ### def inputs ###
    nifti_image = img_array
    affine = img_affine

    def convert_orientation_to_str(orientation_tuple):
        # 방향 코드와 그에 해당하는 문자열을 딕셔너리로 정의
        orientation_dict = {
            'L': 'left',
            'R': 'right',
            'A': 'anterior',
            'P': 'posterior',
            'S': 'superior',
            'I': 'inferior'
        }
        # 튜플의 각 요소를 문자열로 변환하여 결합
        orientation_str = '-'.join(orientation_dict[char] for char in orientation_tuple)
        return orientation_str
    space = convert_orientation_to_str(nib.orientations.aff2axcodes(affine))
    # print(space)


    # RAS to LPS 변환 행렬
    ras_to_lps = np.diag([-1, -1, 1, 1])
    # NRRD에서 사용할 affine matrix 계산
    nrrd_affine = np.dot(ras_to_lps, affine)
    # nrrd_affine = affine
    # print(nib.orientations.aff2axcodes(nrrd_affine))

    # NRRD space directions와 space origin 추출
    space_directions = nrrd_affine[:3, :3]
    space_origin = nrrd_affine[:3, 3]


    # NRRD 헤더 생성
    nrrd_header = {
        'space': space,
        'sizes': list(nifti_image.shape),
        'space origin': space_origin.tolist(),
    }
    nrrd_header['dimension'] = len(nifti_image.shape)

    if nrrd_header['dimension'] == 3:
        nrrd_header['space directions'] = [list(space_directions[:, i]) for i in range(3)]

    if nrrd_header['dimension'] == 4:
        nrrd_header['space directions'] = [list(space_directions[:, i]) for i in range(3)] + [None]
        nrrd_header['kinds'] = ['domain', 'domain', 'domain', 'list'] # 마지막 축이 채널
        
    # NRRD 파일로 저장
    nrrd.write(nrrd_file_path, nifti_image, header=nrrd_header)
    print(f"NRRD file saved at {nrrd_file_path}")




class CFG:
    ModelInfo = {
            'Name':'SwinUNETR',
            'FeatureSize':24,
            # 'InferenceWeights':UserCFG.model_weight_path,
            'InferenceWeights':None,
        }
    # Device = torch.device(UserCFG.device)
    Device = None

    NumWorkers = 0
    BatchSize = 1
    PreProcessing = {
        'IntensityScale':{'a_max':300, 'a_min':0, 'b_max':1., 'b_min':0.},
        'SlidingWindow':{'BS':1},
        'PixDim':(0.8, 0.8, 1.),
    }    
    ImgInfo = {'Size':(192,192,128), 'Channel':1}
    MaskInfo = {
        'Background':0, 'LT_lateral':1, 'LT_medial': 2, 
        'RT_anterior':3, 'RT_posterior':4, 'SPG':5, 'HV':6, 'PV':7
    }
    ### Post-processing ###
    InputDim = 3
    PostPred = AsDiscrete(argmax=True, to_onehot=len(MaskInfo))
    PostLabel = AsDiscrete(to_onehot=len(MaskInfo))
    PostPred2Save = AsDiscreted(keys='pred', argmax=True)
    if InputDim == 4:
        PostPred = AsDiscrete(argmax=False, to_onehot=None, threshold=0.5, rounding=None)
        PostLabel = AsDiscrete(to_onehot=None)
        PostPred2Save = AsDiscreted(keys='pred', argmax=False, to_onehot=None, threshold=0.5, rounding=None)




import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="두 숫자의 합을 계산하는 프로그램")
    # 인자를 명시적으로 지정할 수 있도록 'option-style'로 추가
    # parser.add_argument("--image_path", type=str, default=None, required=True, help="must be .nii.gz format")
    # parser.add_argument("--model_weight_path", type=str, default=None, required=True)
    parser.add_argument("--image_path", type=str, default=r"sample_input.nii.gz",
                        # required=True, 
                        help="must be .nii.gz format")
    parser.add_argument("--model_weight_path", type=str, default=r"model\2819_1.0457_0.8379.pth",
                        #  required=True
                         )
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--results_save_dir_path", type=str, default=r"results",
                        #  required=True
                         )

    # args = parser.parse_args()
    UserCFG = parser.parse_args()
    # UserCFG.image_path = args.image_path
    # UserCFG.model_weight_path = args.model_weight_path
    # UserCFG.device = args.device
    # UserCFG.results_save_dir_path = args.results_save_dir_path

    print(UserCFG)
    CFG.Device = UserCFG.device
    CFG.ModelInfo['InferenceWeights'] = UserCFG.model_weight_path


    model = GetModelWithCP()
    prediction = Inference(model, UserCFG.image_path)

    prediction_ = [PostTransforms()(i) for i in decollate_batch(prediction)][0]
    img_array = prediction_['pred'].numpy()[0]
    img_affine = prediction_['pred'].meta['affine'].numpy()

    nrrd_file_path = os.path.join(UserCFG.results_save_dir_path, 'seg_'+os.path.basename(prediction['pred'].meta['filename_or_obj'][0]).split('.')[0]+'.nrrd')

    write_to_nrrd(img_array, img_affine, nrrd_file_path)


