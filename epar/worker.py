from argparse import Namespace
import importlib
import json
import logging
from pathlib import Path
import sys
import yaml
from .config import STAGES


ENGINE_MODULES = {
    'train_1st': 'epar.engines.feature_adaptation',
    'train_2nd': 'epar.engines.nsn_trainer',
    'inference': 'epar.engines.inference',
}


def main():
    args = Namespace(**json.load(sys.stdin))
    if args.stage not in STAGES:
        raise ValueError(f'Unknown stage: {args.stage}')
    run_dir = Path(args.output_dir) / args.stage / args.class_name
    run_dir.mkdir(parents=True, exist_ok=True)
    is_inference = args.stage == 'inference'
    if is_inference:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(),
                      logging.FileHandler(run_dir / 'run.log', encoding='utf-8')],
            force=True,
        )
        logging.info('Inference started: class=%s, dataset=%s',
                     args.class_name, args.dataset_path)
        logging.info('Checkpoint directories: stage1=%s, stage2=%s',
                     args.stage1_dir, args.stage2_dir)
    else:
        # Training keeps print/tqdm output, without Python logging or log files.
        logging.disable(logging.CRITICAL)

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('EPAR requires a CUDA GPU for KNN_CUDA, PointNet2 and GPU FAISS.')
        for directory in (args.stage1_dir, args.stage2_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)
        (run_dir / 'resolved_config.yaml').write_text(
            yaml.safe_dump(vars(args), sort_keys=False), encoding='utf-8')
        engine = importlib.import_module(ENGINE_MODULES[args.stage])
        engine.run(args)
    except Exception:
        if is_inference:
            logging.exception('Inference failed: class=%s', args.class_name)
        raise
    else:
        if is_inference:
            logging.info('Inference completed: class=%s', args.class_name)
    finally:
        if is_inference:
            logging.shutdown()


if __name__ == '__main__':
    main()
