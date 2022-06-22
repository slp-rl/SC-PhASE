import logging
import os
import hydra
import wandb
from dset_builders.dset_builder_factory import DsetBuilderFactory
from external_files.executor import start_ddp_workers
from models.model_factory import ModelFactory
from solvers.solver_factory import SolverFactory

logger = logging.getLogger(__name__)


def _get_wandb_config(args):
    included_keys = ['eval_every', 'optim', 'lr', 'loss', 'epochs', 'num_workers']
    wandb_config = {k: args[k] for k in included_keys}
    wandb_config.update(**args)
    return wandb_config


def init_wandb(args):
    wandb_mode = os.environ['WANDB_MODE'] if 'WANDB_MODE' in os.environ.keys() else args.wandb.mode
    wandb.init(mode=wandb_mode, project=args.wandb.project, entity=args.wandb.wandb_entity, config=_get_wandb_config(args),
               group=args.experiment_name, resume=(args.continue_from != ""),
               name=args.experiment_name)


def _log_obj(name, obj, prefix):

    if name in ["wandb", "dset", "model"]:  # TODO: change this to include additional values in the log
        try:
            obj = vars(obj)["_content"]
        except Exception:
            return
    if isinstance(obj, dict):
        logger.info(f"{prefix}{name}:")
        for k, v in obj.items():
            _log_obj(k, v, prefix + "  ")
    else:
        logger.info(f"{prefix}{name}: {obj}")


def log_args(args):
    _log_obj("Args", vars(args)["_content"], "")


def init_hydra_and_logs(args):
    global __file__
    log_args(args)
    for key, value in args.dset.items():
        if isinstance(value, str) and key not in ["matching"]:
            args.dset[key] = hydra.utils.to_absolute_path(value)
    __file__ = hydra.utils.to_absolute_path(__file__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("denoise").setLevel(logging.DEBUG)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)


def run(args):
    import torch
    from external_files import distrib

    # init seed and distrib
    torch.manual_seed(args.seed)
    distrib.init(args)

    # construct model
    model = ModelFactory.get_model(args)
    wandb.watch(model, log=args.wandb.log, log_freq=args.wandb.log_freq)

    if args.show:
        logger.info(model)
        mb = sum(p.numel() for p in model.parameters()) * 4 / 2**20
        logger.info('Size: %.1f MB', mb)
        if hasattr(model, 'valid_length'):
            field = model.valid_length(1)
            logger.info('Field: %.1f ms', field / args.dset.sample_rate * 1000)
        return

    assert args.batch_size % distrib.world_size == 0
    args.batch_size //= distrib.world_size

    # build data loaders
    tr_loader, cv_loader, tt_loader, enh_loader = DsetBuilderFactory.get_loaders(args, model)
    enh_loader = tt_loader if enh_loader is None else enh_loader
    data = {"tr_loader": tr_loader, "cv_loader": cv_loader, "tt_loader": tt_loader, "enh_loader": enh_loader}

    if torch.cuda.is_available():
        model.cuda()

    # optimizer - TODO: change/add more optimizers
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta, args.beta2))
    else:
        logger.fatal('Invalid optimizer %s', args.optim)
        os._exit(1)

    # Construct Solver
    solver = SolverFactory.get_solver(data, model, optimizer, args)
    solver.train()


def _main(args):

    # init wandb
    init_wandb(args)

    # init hydra and logs
    init_hydra_and_logs(args)

    if args.ddp and args.rank is None:
        start_ddp_workers(args)
    else:
        run(args)


@hydra.main(config_path="configurations", config_name="main_config") #  for latest version of hydra=1.0
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)


if __name__ == "__main__":
    main()