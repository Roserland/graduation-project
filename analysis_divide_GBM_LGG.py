import pandas as pd
from shutil import move as mv
import os

train_df = pd.read_csv('./coords/G_TwoTypes_Train.csv')
test_df  = pd.read_csv('./coords/G_TwoTypes_Test.csv')

def move_patchs(test_df, selected_patchs_dir = './selected/selected_patchs/',
                gbm_val_dir = './selected_patches/val_GBM',
                lgg_val_dir = './selected_patches/val_LGG',
                gbm_train_dir='./selected_patches/train_GBM',
                lgg_train_dir='./selected_patches/train_LGG',
                ):
    ids = test_df['Case ID']
    labels = test_df['TypeName']

    assert len(ids) == len(labels)

    hash_dict = {}
    for i in range(len(ids)):
        hash_dict[ids[i]] = labels[i]

    patchs_list = os.listdir(selected_patchs_dir)
    print(len(hash_dict), len(patchs_list))
    # assert len(hash_dict) == len(patchs_list)

    full_paths = [os.path.join(selected_patchs_dir, item) for item in patchs_list]

    patchs_id_list = [patch_case_id[:12] for patch_case_id in patchs_list]

    for i in range(len(full_paths)):
        patch_src_path = full_paths[i]

        patch_id = patchs_id_list[i]
        patch_name = patch_id + '.jpg'

        if patch_id not in hash_dict.keys():
            continue

        hash_res = hash_dict[patch_id]

        if hash_res == "GBM":
            mv(patch_src_path, os.path.join(gbm_train_dir, patch_name))
        elif hash_res == "LGG":
            mv(patch_src_path, os.path.join(lgg_train_dir, patch_name))
        else:
            raise KeyError

if __name__ == '__main__':
    move_patchs(test_df=train_df)




