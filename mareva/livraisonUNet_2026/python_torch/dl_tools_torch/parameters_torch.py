"""Experiment parameters, PyTorch port of parameters.py.

The `opt` field is now a torch.optim.Optimizer instance, set by
dl_tools_torch.tools_torch.set_optimizer(model, myP) once the model exists
(unlike Keras optimizers, PyTorch optimizers need the model's parameters at
construction time, so this can no longer be done inside __init__).
"""


class Parameters:
    def __init__(self, MODEL="modular_u_net", batch_size=128, opt_name="rmsprop"):
        self.batch_size = batch_size
        self.MODEL = MODEL

        # architecture params
        self.nb_levels = 3
        self.nb_filters = 32
        self.sigma_noise = 0.01
        self.DataAugmentation = False

        self.skip = True
        self.res = True
        self.conv_sampling = False
        self.drop = 0.0
        self.batch_norm = True
        self.attention = False

        self.opt_name = opt_name  # choices: adadelta, sgd, rmsprop, adagrad, adam
        self.opt = None           # set by tools_torch.set_optimizer(model, myP)
        self.lr = 0.01

        # fit params
        self.nb_epoch = 100
        self.patience = 5
        self.nb_train_samples = 5000
        self.nb_val_samples = 1000
        self.nb_test_samples = 100

        # input data generator
        self.img_rows, self.img_cols = 32, 32
        self.img_channels = 1
        self.gauss_n_std = 40

        # majority class
        self.nb_obj_l, self.nb_obj_h = 1, 3
        self.r1_ring_l, self.r1_ring_h = 8, 14
        self.grey_l, self.grey_h = 50, 200

        # minority class
        self.nb2_obj_l, self.nb2_obj_h = 1, 2
        self.r2_ring_l, self.r2_ring_h = 3, 6
        self.grey2_l, self.grey2_h = 70, 150

        self.norm = 255  # normalization constant
        self.wo_bckg, self.wo_str = False, "w"
        self.showFlag = False
        self.classList = [1, 2]

        # loss function selector, consumed by tools_torch.set_loss
        self.lossF = "jac2"  # choices: jac, jac2, cce, focal, mse

        # bookkeeping: where results/checkpoints/plots get written, and the
        # base name used for those files (set these before training)
        self.dirRes = "."
        self.baseName = "run"

    def update_params(self, experience):
        for name, value in vars(experience).items():
            if "__" not in name:
                print(name, value)
                setattr(self, name, value)


class ExperienceClass:
    def __init__(self, MODEL="u_net_mod_lev"):
        self.MODEL = MODEL


def dump_class(myclass):
    for name, value in vars(myclass).items():
        print(name, "=", value)
