"""Capture raw Stage-1 epochs using the pinned official training method.

The method is copied at runtime from the hashed upstream source. The only
changes are instrumentation hooks and removal of the early-stop break; the
forward pass, loss, optimizer, validation, scheduler and data loaders remain
the upstream implementation.
"""
import argparse, copy, hashlib, inspect, json, pathlib, sys, textwrap, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from ops import *
import ts_benchmark.baselines.worldts.worldts as official


def tensor_hash(values):
    h = hashlib.sha256()
    for key, value in sorted(values.items()):
        h.update(key.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def instrument_official_method():
    original = textwrap.dedent(inspect.getsource(WorldTS._run_stage_train))
    modifications = [
        ("    for epoch in range(num_epochs):", "    self._audit_prepare_resume(stage, optimizer, early_stop)\n    if self._audit_start_epoch:\n        stage_best_loss = self._audit_prior_best_loss\n        stage_best_state = self._audit_prior_best_state\n    for epoch in range(self._audit_start_epoch, num_epochs):"),
        ("        self.model.train()\n        if stage.startswith('stage2'):", "        self._audit_epoch_start(epoch + 1)\n        self.model.train()\n        if stage.startswith('stage2'):"),
        ("            output = out_loss['output']", "            self._audit_train_batch(out_loss)\n            output = out_loss['output']"),
        ("                    if early_stop.early_stop:\n                        break", "                    if early_stop.early_stop:\n                        pass  # audit captures the prescribed raw epoch 6"),
        ("        if (\n            debug_batch_forecast_callback is not None", "        self._audit_epoch_end(stage, epoch + 1, optimizer, early_stop, stage_best_loss, stage_best_state, valid_loss)\n\n        if (\n            debug_batch_forecast_callback is not None"),
    ]
    revised = original
    for old, new in modifications:
        assert revised.count(old) == 1, old
        revised = revised.replace(old, new)
    namespace = dict(vars(official))
    exec(compile(revised, str(pathlib.Path(official.__file__)), 'exec'), namespace)
    WorldTS._run_stage_train = namespace['_run_stage_train']
    return [{'old': old, 'new': new} for old, new in modifications]


class Replay(WorldTS):
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset
        self.run_dir = ROOT / 'outputs/checkpoints' / dataset
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_records = []
        self._audit_start_epoch = 0
        self._audit_train_values = []
        self._audit_started = time.perf_counter()
        self._audit_initial_rng = None

    def _audit_prepare_resume(self, stage, optimizer, early_stop):
        assert stage == 'stage1_latent'
        latest = self.run_dir / 'stage1_latest.pt'
        if not latest.is_file():
            state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            original_decoder = torch.load(self.run_dir / 'untrained_decoder.pt', map_location='cpu', weights_only=False)['Decoder']
            init_decoder = {k[len('decoder.'):]: v for k, v in state.items() if k.startswith('decoder.')}
            initial = {'model_tensor_hash': tensor_hash(state),
                       'decoder_tensor_hash': tensor_hash(init_decoder),
                       'original_stage1_decoder_tensor_hash': tensor_hash(original_decoder),
                       'rng': rng(), 'dataset': self.dataset_name}
            torch.save(initial, self.run_dir / 'stage1_initial.pt')
            self._audit_initial_rng = initial['rng']
            print('AUDIT_INIT', self.dataset_name, initial['model_tensor_hash'],
                  'decoder_match_original', initial['decoder_tensor_hash'] == initial['original_stage1_decoder_tensor_hash'], flush=True)
            return
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        assert saved['dataset'] == self.dataset_name and 1 <= saved['epoch'] <= 6
        self.model.load_state_dict(saved['Model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        for key, value in saved['early_stop'].items(): setattr(early_stop, key, value)
        self._audit_start_epoch = saved['epoch']
        self.run_records = saved['run_records']
        self._audit_prior_best_loss = saved['stage_best_loss']
        self._audit_prior_best_state = saved['stage_best_state']
        restore_rng(saved['rng'])
        print('AUDIT_RESUME', self.dataset_name, self._audit_start_epoch, flush=True)

    def _audit_epoch_start(self, epoch):
        self._audit_train_values = []
        self._audit_epoch_start_rng = rng()
        print('AUDIT_EPOCH_START', self.dataset_name, epoch, flush=True)

    def _audit_train_batch(self, out_loss):
        self._audit_train_values.append({key: float(out_loss[key].detach().cpu())
                                         for key in ['additional_loss', 'latent_mse_loss',
                                                     'latent_cosine_loss', 'latent_sigreg_loss']})

    def _audit_epoch_end(self, stage, epoch, optimizer, early_stop,
                         stage_best_loss, stage_best_state, valid_loss):
        assert stage == 'stage1_latent' and len(self._audit_train_values) == 1112
        state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        by_key = {key: float(np.mean([row[key] for row in self._audit_train_values]))
                  for key in self._audit_train_values[0]}
        record = {'dataset': self.dataset_name, 'epoch': epoch,
                  'train_online': by_key, 'validation_official': float(valid_loss),
                  'validation_components': dict(getattr(self, '_last_stage1_valid_components', {})),
                  'optimizer_steps': epoch * 1112,
                  'lr_next_epoch': [float(g['lr']) for g in optimizer.param_groups],
                  'raw_model_tensor_hash': tensor_hash(state),
                  'best_loss_so_far': float(stage_best_loss),
                  'early_stop_counter': early_stop.counter,
                  'elapsed_seconds': time.perf_counter() - self._audit_started}
        self.run_records.append(record)
        payload = {'Model': state, 'epoch': epoch, 'dataset': self.dataset_name,
                   'identity': 'raw completed epoch before best restoration',
                   'optimizer': optimizer.state_dict(), 'rng': rng(),
                   'early_stop': {key: getattr(early_stop, key) for key in
                                  ['patience', 'counter', 'best_score', 'early_stop', 'val_loss_min', 'delta']},
                   'stage_best_loss': stage_best_loss,
                   'stage_best_state': {k: v.detach().cpu().clone() for k, v in stage_best_state.items()},
                   'run_records': list(self.run_records),
                   'epoch_start_rng': self._audit_epoch_start_rng,
                   'source_sha256': sha(official.__file__),
                   'protocol_sha256': sha(ROOT / 'configs/protocol.json')}
        target = self.run_dir / f'replay_stage1_epoch{epoch}.pt'
        temporary = target.with_suffix('.pt.tmp')
        torch.save(payload, temporary); temporary.replace(target)
        latest = self.run_dir / 'stage1_latest.pt'
        latest_tmp = latest.with_suffix('.pt.tmp')
        torch.save(payload, latest_tmp); latest_tmp.replace(latest)
        write_json(self.run_dir / 'stage1_history.json', self.run_records)
        print('AUDIT_EPOCH_DONE', self.dataset_name, epoch,
              'train', by_key['additional_loss'], 'validation', valid_loss,
              'tensor', record['raw_model_tensor_hash'], flush=True)


def execute(dataset):
    existing = ROOT / 'outputs/checkpoints' / dataset / 'replay_stage1_epoch6.pt'
    if existing.is_file():
        saved = torch.load(existing, map_location='cpu', weights_only=False)
        assert saved['epoch'] == 6 and saved['dataset'] == dataset
        print('AUDIT_STAGE1_ALREADY_COMPLETE', dataset, flush=True)
        return
    cfg, directory, metadata, checkpoint = config(dataset)
    kwargs = vars(cfg).copy(); kwargs.pop('criterion', None)
    kwargs.update(training_stage='stage1_latent', training_stage_plan=['stage1_latent'],
                  stage1_epochs=6, stage_save_dir=str(ROOT / 'outputs/checkpoints' / dataset / 'official_best'),
                  stage1_checkpoint_path='', final_checkpoint_path='')
    seed(2021)
    wrapper = Replay(dataset, **kwargs)
    loaded = data(dataset)
    df = loaded['df'].iloc[:loaded['tv']]
    frame = df.iloc[:, -1:]
    exog = df.iloc[:, :-1]
    assert len(frame) == 41932 and loaded['tr'] == 36690
    wrapper.forecast_fit(frame, covariates={'exog': exog}, train_ratio_in_tv=loaded['tr']/loaded['tv'])
    assert existing.is_file()
    save = torch.load(existing, map_location='cpu', weights_only=False)
    assert len(save['run_records']) == 6
    print('AUDIT_STAGE1_COMPLETE', dataset, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--dataset', choices=['NP', 'PJM'], required=True)
    args = parser.parse_args()
    modifications = instrument_official_method()
    write_json(ROOT / 'outputs/stage1_instrumentation.json',
               {'official_source_sha256': sha(official.__file__), 'modifications': modifications})
    execute(args.dataset)
