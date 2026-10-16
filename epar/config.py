"""Load the fixed MVTec 3D-AD recipe; only the dataset route comes from CLI."""
from argparse import Namespace
from pathlib import Path
import math
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'config' / 'mvtec3d.yaml'
STAGES = ('train_1st', 'train_2nd', 'inference')
CLASSES = ('bagel', 'cable_gland', 'carrot', 'cookie', 'dowel', 'foam',
           'peach', 'potato', 'rope', 'tire')


def project_path(value):
    path = Path(value).expanduser()
    return str((ROOT / path).resolve())


def load_config():
    with CONFIG_PATH.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    required = {'classes', 'paths', 'common', 'stages', 'class_overrides'}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f'{CONFIG_PATH}: expected sections {sorted(required)}')
    classes = config['classes']
    if not isinstance(classes, list) or not classes or any(c not in CLASSES for c in classes):
        raise ValueError(f'classes must be a nonempty list drawn from {CLASSES}')
    if len(classes) != len(set(classes)):
        raise ValueError('classes contains duplicate entries')
    for section in required - {'classes'}:
        if not isinstance(config[section], dict):
            raise ValueError(f'{section} must be a mapping')
    if set(config['stages']) != set(STAGES):
        raise ValueError(f'stages must contain {STAGES}')
    if set(config['class_overrides']) - set(CLASSES):
        raise ValueError('Unknown class in class_overrides')
    for cls, overrides in config['class_overrides'].items():
        if not isinstance(overrides, dict) or set(overrides) - set(STAGES):
            raise ValueError(f'Invalid stage override for {cls}')
        for stage, values in overrides.items():
            allowed = set(config['common']) | set(config['stages'][stage])
            if not isinstance(values, dict) or set(values) - allowed:
                raise ValueError(f'Unknown option in class_overrides.{cls}.{stage}')
    return config


def resolve_args(config, stage, class_name, dataset_path):
    if stage not in STAGES or class_name not in config['classes']:
        raise ValueError(f'Unconfigured stage/class: {stage}/{class_name}')
    values = dict(config['common'])
    values.update(config['stages'][stage])
    values.update(config['class_overrides'].get(class_name, {}).get(stage, {}))
    # PyYAML may parse hand-written values such as 1e-6 as strings.
    for key, value in list(values.items()):
        if key.startswith(('lr_', 'start_lr_', 'final_lr_', 'bank_lr_', 'coreset_percentage_')) or key in (
                'vic_rate', 'weight_decay', 'final_weight_decay', 'ipe_scale'):
            values[key] = float(value)
            if not math.isfinite(values[key]) or values[key] < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
    for key in ('num_epochs', 'npred', 'min_keep', 'batch_size', 'img_size', 'patch_size'):
        if type(values[key]) is not int or values[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if values['batch_size'] != 1 or values['img_size'] != 224 or values['patch_size'] != 8:
        raise ValueError('The full-VIC recipe requires batch_size=1, img_size=224, patch_size=8')
    if values['usage_half_precision'] is not False:
        raise ValueError('This preserved training loop supports usage_half_precision=false only')
    if stage == 'inference' and values['mode'] != 'combined':
        raise ValueError('EPAR inference uses mode=combined')
    root = Path(dataset_path).expanduser().resolve()
    # Accept either datasets/ or the directory directly containing bagel/, etc.
    if (root / 'mvtec').is_dir():
        root = root / 'mvtec'
    paths = config['paths']
    expected_paths = {'output_dir', 'stage1_dir', 'stage2_dir',
                      'visual_encoder_ckpt_pth', 'point_encoder_ckpt_pth', 'dtd_dataset_path'}
    if set(paths) != expected_paths:
        raise ValueError(f'paths must contain {sorted(expected_paths)}')
    values.update({k: project_path(v) for k, v in paths.items() if k != 'dtd_dataset_path'})
    values['dtd_dataset_path'] = (project_path(paths['dtd_dataset_path'])
                                  if paths['dtd_dataset_path'] else str(root.parent / 'dtd'))
    values.update(dataset_path=str(root), camera_param_route=str(root), dataset_type='mvtec',
                  class_name=class_name, stage=stage)
    return Namespace(**values)


def checkpoint_paths(args, stage):
    suffix = f'{args.npred}blocks_full_{args.vic_rate}.pth'
    if stage == 'train_1st':
        return [Path(args.stage1_dir) / f'{args.class_name}_best_jepa_{mod}_main_{suffix}'
                for mod in ('rgb_dino', 'xyz_pmae')]
    return [Path(args.stage2_dir) / f'{args.class_name}_{mod}_bank_main_{suffix}'
            for mod in ('rgb', 'xyz')]


def validate_inputs(args):
    root = Path(args.dataset_path) / args.class_name
    splits = ('train', 'validation') if args.stage == 'train_1st' else (
        ('train', 'validation', 'test') if args.stage == 'train_2nd' else ('train', 'test'))
    for split in splits:
        if not (root / split).is_dir():
            raise FileNotFoundError(f'Missing dataset split: {root / split}')
    files = [Path(args.visual_encoder_ckpt_pth), Path(args.point_encoder_ckpt_pth)]
    if args.stage in ('train_2nd', 'inference'):
        files += checkpoint_paths(args, 'train_1st')
    if args.stage == 'inference':
        files += checkpoint_paths(args, 'train_2nd')
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f'Missing checkpoint: {path}. Check config/mvtec3d.yaml and stage order.')
