class Path_Hyperparameter:
    random_seed = 42

    # training hyper-parameter
    epochs: int = 300  # Number of epochs
    batch_size: int = 2  # Batch size
    inference_ratio = 2  # batch_size in val and test equal to batch_size*inference_ratio
    learning_rate: float = 1e-3  # Learning rate
    factor = 0.1  # learning rate decreasing factor
    patience = 12  # schedular patience
    warm_up_step = 1000  # warm up step
    weight_decay: float = 1e-3  # AdamW optimizer weight decay
    amp: bool = True  # if use mixed precision or not
    load: str = None  # Load model and/or optimizer from a .pth file for testing or continuing training
    max_norm: float = 20  # gradient clip max norm

    # evaluate and test hyper-parameter
    evaluate_epoch: int = 10  # start evaluate after training for evaluate epochs
    evaluate_inteval: int = 5  # evaluate every evaluate_inteval epoch 
    test_epoch: int = 30  # start test after training for test epochs
    stage_epoch = [0, 0, 0, 0, 0]  # adjust learning rate after every stage epoch
    save_checkpoint: bool = True  # if save checkpoint of model or not
    save_interval: int = 10  # save checkpoint every interval epoch
    save_best_model: bool = True  # if save best model or not


    # model hyper-parameter
    # RSM_SS tiny
    drop_path_rate = 0.2
    dims = 96
    depths = [ 2, 2, 9, 2 ]
    ssm_d_state = 16
    ssm_dt_rank = "auto"
    ssm_ratio = 2.0
    mlp_ratio = 4.0
    # downsample_version = "v3"
    # patchembed_version = "v2"

    # data parameter
    original_image_size = 1500
    target_image_size = 1024 #Size after downsampling, needs to be divisible by 8. (Because patch-embedding=4 and encoder division by 2)
    crop_size = 256 #Size of cropped tiles that the model will be trained on. Needs to divide evenly or be equal to 'target_image_size'
    downsample_ratio = (original_image_size/target_image_size)
    dataset_name = 'your_dataset_name'
    root_dir = '.'  # the root dir of your dataset


    # inference parameter
    log_path = './log_feature/'

    # log wandb hyper-parameter
    # log_wandb_project: str = 'train_whu_cd'  # wandb project name
    log_wandb_project: str = 'train_whu'  # wandb project name


    project_name = f'{log_wandb_project}_{target_image_size}_{learning_rate}'

    def state_dict(self):
        return {k: getattr(self, k) for k, _ in Path_Hyperparameter.__dict__.items() \
                if not k.startswith('_')}


ph = Path_Hyperparameter()
