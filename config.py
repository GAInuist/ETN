from argparse import ArgumentParser
def get_config_parser():
    parser = ArgumentParser()
    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='0', type=str, help='The GPU index')
    parser.add_argument('--DATASET', default='SUN', choices=['AWA2', 'CUB', 'SUN'], type=str, help='Dataset type')
    parser.add_argument('--DATASET_path', default='/home/c402/backup_project/Dataset/SUN/images/', type=str,
                        help='the path of DATASET')
    parser.add_argument('--attr_num', default=102, choices=[85, 312, 102], type=int,
                        help='Attribute Num for your Dataset')
    parser.add_argument('--v_embedding', default=2048, type=int,
                        help='visual feature embedding according to resnet: [resnet101:2048|others:1024]')
    parser.add_argument('--region_num', default=196, type=int,
                        help='region number according to img size: [(224 * 224):49|(448 * 448):196]')

    ### CPP fusion coefficient
    parser.add_argument('--beta1', default=0.35, choices=[0.35, 0.15, 0.5], help='the coefficient of local weight')
    parser.add_argument('--beta2', default=0.65, choices=[0.65, 0.85, 0.5], help='the coefficient of global weight')

    # Training hyperparameter
    parser.add_argument('--is_train', default=False, type=bool, help='Whether to start training')
    parser.add_argument('--is_test', default=True, type=bool, help='Whether to start testing')
    parser.add_argument('--cs', default=False, type=bool, help='get calibrator stack')
    parser.add_argument('--gamma', default=0.5, choices=[0.85, 0.95, 0.5], type=float, help='Pre-specified gamma')
    parser.add_argument('--Num_Epochs', default=150, type=int, help='Training Epochs')
    parser.add_argument('--batch_size', default=32, type=float, help='the batch size of train loop')
    parser.add_argument('--test_batch', default=64, type=float, help='the batch size of test loop')
    parser.add_argument('--pretrain_path', default='/home/c402/backup_project/zyf/ETN/checkpoint/SUN/ETN_SUN_45.3_GZSL.pth', type=str,
                        help='the path of pretrain model')
    parser.add_argument('--drop_rate', default=.3, type=float, help='Dropout Layer Parameter Settings')
    parser.add_argument('--n_head', default=8, type=int, help='the number of heads in the attention mechanism')
    parser.add_argument('--lambda_', default=[1., 0.001, 0.001], type=list, help='Loss weight')
    parser.add_argument('--num_workers', default=4, type=int, help='The number of worker threads')
    parser.add_argument('--seed', default=None, type=int, help='the random seed')
    parser.add_argument('--use_w2v', default=True, type=bool,
                        help='Whether to use Word2Vector as auxiliary information')
    return parser