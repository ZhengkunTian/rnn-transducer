import torch
import logging
import editdistance


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item not in self:
            return None
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]

    def __setattr__(self, item, value):
        self.__dict__[item] = value


def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def computer_cer(preds, labels):
    dist = sum(editdistance.eval(label, pred) for label, pred in zip(labels, preds))
    total = sum(len(l) for l in labels)
    return dist, total


def get_saved_folder_name(config):
    return '_'.join([config.data.name, config.training.save_model])


def count_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' in name:
            dec += param.nelement()
    return n_params, enc, dec


def init_parameters(model, type='xnormal'):
    for p in model.parameters():
        if p.dim() > 1:
            if type == 'xnoraml':
                torch.nn.init.xavier_normal_(p)
            elif type == 'uniform':
                torch.nn.init.uniform_(p, -0.1, 0.1)
        else:
            pass


def save_model(model, optimizer, config, save_name):
    multi_gpu = True if config.training.num_gpu > 1 else False
    checkpoint = {
        'encoder': model.module.encoder.state_dict() if multi_gpu else model.encoder.state_dict(),
        'decoder': model.module.decoder.state_dict() if multi_gpu else model.decoder.state_dict(),
        'joint': model.module.joint.state_dict() if multi_gpu else model.joint.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': optimizer.current_epoch,
        'step': optimizer.global_step
    }

    torch.save(checkpoint, save_name)


def save_ctc_model(model, optimizer, config, save_name):
    multi_gpu = True if config.training.num_gpu > 1 else False
    checkpoint = {
        'encoder': model.module.encoder.state_dict() if multi_gpu else model.encoder.state_dict(),
        'project_layer': model.module.project_layer.state_dict() if multi_gpu else model.project_layer.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': optimizer.current_epoch,
        'step': optimizer.global_step
    }

    torch.save(checkpoint, save_name)


def save_language_model(model, optimizer, config, save_name):
    multi_gpu = True if config.training.num_gpu > 1 else False
    checkpoint = {
        'decoder': model.module.decoder.state_dict() if multi_gpu else model.decoder.state_dict(),
        'project_layer': model.module.project_layer.state_dict() if multi_gpu else model.project_layer.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': optimizer.current_epoch,
        'step': optimizer.global_step
    }

    torch.save(checkpoint, save_name)

