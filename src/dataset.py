import numpy as np
import torch
import os
import nibabel as nib


class ParseDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, fns, transforms=None, mode='train'):           
        assert mode in ['train', 'test']
        assert os.path.isdir(base_path)

        self.base_path = base_path
        self.fns = fns
        self.transforms=transforms
        self.mode = mode       

    def __read_nifti__(self, path_to_nifti):
        return nib.load((path_to_nifti)).get_fdata()

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, index):
        sample = dict()
        fn = self.fns[index]
        
        img_fn = os.path.join(self.base_path, fn, 'image', fn + '.nii.gz')
        assert os.path.isfile(img_fn)

        if self.mode == 'train':
            label_fn = os.path.join(self.base_path, fn, 'label', fn + '.nii.gz')
            assert os.path.isfile(label_fn)
            img = self.__read_nifti__(img_fn)
            label = self.__read_nifti__(label_fn)

            sample['input'] = img
            sample['target'] = label            

        elif self.mode =='test':
            pass
      
        if self.transforms:
            sample = self.transforms(sample)

        return sample


if __name__ == "__main__":
    import os, random
    import pandas as pd

    base_path=r'C:\Users\bed1\src\parse2022\data\train'       
    df_fn = r'C:\Users\bed1\src\parse2022\data\data_split.csv'
    assert os.path.isdir(base_path)
    assert os.path.isfile(df_fn)

    df = pd.read_csv(df_fn) 
    df = df.loc[(df['kfold_idx']==0) & (df['mode']=='train')]['fn'].values.tolist()
    
    dataset = ParseDataset(
        base_path=base_path,
        fns=df,
        mode='train',
    )

    sample = random.choice(dataset)
    print('image shape : {}'.format(sample['input'].shape))
    print('mask shape : {}'.format(sample['target'].shape)) 