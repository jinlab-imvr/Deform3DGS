ModelParams = dict(
    extra_mark = 'endonerf',
    camera_extent = 10
)

OptimizationParams = dict(
    coarse_iterations = 0,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    iterations = 3000,
    percent_dense = 0.01,
    opacity_reset_interval = 3000,
    position_lr_max_steps = 4000,
    prune_interval = 3000
)

ModelHiddenParams = dict(
    curve_num = 17, # number of learnable basis functions. This number was set to 17 for all the experiments in paper (https://arxiv.org/abs/2405.17835)

    ch_num = 10, # channel number of deformable attributes: 10 = 3 (scale) + 3 (mean) + 4 (rotation)
    init_param = 0.01, )
