from configs.default_configs import get_default_configs

def init_cfg(cfg_proj):
    config = get_default_configs()
    config.Note = None

    config.data.dim_out = 2
    config.training.epochs = 30
    config.training.batch_size = 512
    config.training.lr_init = 1.0e-3
    config.training.tol = 1e-4

    config.alpha1 = 0.5 # Adversarial Coefficient
    config.alpha2 = 0.5 # 
    config.alpha3 = 0.1

    config.l2_lambda = None #or None
    config.l1_lambda = None

    config.training.epochs_whiting = 60

    return config
