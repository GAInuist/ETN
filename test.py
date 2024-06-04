import torch
from network.ETN import ETN
path = '/home/c402/backup_project/zyf/ETN/checkpoint/SUN/SUN_ETN_45.2_9149.pth'
pth_dict = torch.load(path)

model = ETN(dim=2048, attr_num=102, drop_rate=.3, n_head=8)
model_dict = model.state_dict()
replace_dict = {'refine_att': 'GAFR', "ffn": "VBL", "bias_learner": 'CPP', "mask_learner": "CDM", "block": 'APP', 'local_predictor': 'coarse_extractor',
                'global_predictor': 'W_g', 'W1': 'W_l', 'W2': 'W1'}
for key, value in pth_dict.items():
    key = key[5:]
    for str_key in replace_dict.keys():
        if str_key in key:
            key = key.replace(str_key, replace_dict[str_key])
    model_dict[key] = value
torch.save(model_dict, '/checkpoint/SUN/ETN_SUN_45.3_GZSL.pth')
