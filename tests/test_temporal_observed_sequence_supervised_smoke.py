from __future__ import annotations
import json
from pathlib import Path
import pytest

from rendering.temporal_bridge import TrainableTemporalConsistencyBackend
from training.temporal_observed_sequences_builder import build_temporal_manifest_from_observed_sequences
from training.temporal_trainer import TemporalTrainer
from training.types import TrainingConfig
from training.datasets import TemporalDataset


def test_temporal_observed_sequence_builder_example(tmp_path: Path) -> None:
    out = tmp_path / 'temporal_manifest.json'
    result = build_temporal_manifest_from_observed_sequences(observed_sequences_path='examples/temporal_sequences.example.json', output_path=str(out), strict=True)
    payload = json.loads(Path(result.manifest_path).read_text(encoding='utf-8'))
    assert payload['contract_version'] == 'temporal_refinement_manifest_v1'
    assert payload['manifest_type'] == 'temporal_refinement_manifest'
    assert payload['records'][0]['target_source'] == 'provided_ground_truth_temporal_frame'



def test_temporal_builder_strict_missing_changed_regions_raises(tmp_path: Path) -> None:
    inp = tmp_path / 'missing_regions.json'
    inp.write_text(json.dumps({'contract_version':'temporal_observed_sequence_manifest_input_v1','sequences':[{'sequence_id':'s1','frames':['examples/assets/source_0001.ppm','examples/assets/target_0001.ppm']}]}), encoding='utf-8')
    with pytest.raises(ValueError, match='changed_regions are required in strict mode'):
        build_temporal_manifest_from_observed_sequences(observed_sequences_path=str(inp), output_path=str(tmp_path / 'o.json'), strict=True)


def test_temporal_builder_strict_empty_changed_regions_raises(tmp_path: Path) -> None:
    inp = tmp_path / 'empty_regions.json'
    inp.write_text(json.dumps({'contract_version':'temporal_observed_sequence_manifest_input_v1','sequences':[{'sequence_id':'s1','frames':['examples/assets/source_0001.ppm','examples/assets/target_0001.ppm'],'changed_regions':[]}]}), encoding='utf-8')
    with pytest.raises(ValueError, match='changed_regions are required in strict mode'):
        build_temporal_manifest_from_observed_sequences(observed_sequences_path=str(inp), output_path=str(tmp_path / 'o.json'), strict=True)

def test_temporal_builder_non_strict_fallback_region_diagnostics(tmp_path: Path) -> None:
    inp = tmp_path / 'noregions.json'
    inp.write_text(json.dumps({'contract_version':'temporal_observed_sequence_manifest_input_v1','sequences':[{'sequence_id':'s1','frames':['examples/assets/source_0001.ppm','examples/assets/target_0001.ppm']}]}), encoding='utf-8')
    out = tmp_path / 'o.json'
    result = build_temporal_manifest_from_observed_sequences(observed_sequences_path=str(inp), output_path=str(out), strict=False)
    payload = json.loads(Path(result.manifest_path).read_text(encoding='utf-8'))
    assert payload['records'][0]['diagnostics']['fallback_region_used'] is True
    assert payload['records'][0]['diagnostics']['region_source'] == 'non_strict_default_temporal_region'


def test_temporal_dataset_rejects_missing_provenance(tmp_path: Path) -> None:
    p = tmp_path / 'missingprov.json'
    p.write_text(json.dumps({'records':[{'previous_frame':[[[0,0,0]]],'current_composed_frame':[[[0,0,0]]],'target_refined_frame':[[[0,0,0]]]}]}), encoding='utf-8')
    with pytest.raises(ValueError):
        TemporalDataset.from_temporal_manifest(str(p), strict=True)
    with pytest.raises(ValueError):
        TemporalDataset.from_temporal_manifest(str(p), strict=False)


def test_temporal_dataset_mixed_valid_and_runtime_generated_keeps_valid(tmp_path: Path) -> None:
    p = tmp_path / 'mixed_manifest.json'
    valid = {'previous_frame':[[[0,0,0]]],'current_composed_frame':[[[0,0,0]]],'target_refined_frame':[[[0,0,0]]],'target_source':'provided_ground_truth_temporal_frame','training_target_quality':'external_or_observed_temporal_target','target_training_role':'supervised_temporal_external','sequence_id':'s1'}
    bad = {'previous_frame':[[[0,0,0]]],'current_composed_frame':[[[0,0,0]]],'target_refined_frame':[[[0,0,0]]],'target_source':'runtime_generated_frame','training_target_quality':'self_generated_runtime_target','target_training_role':'bootstrap','sequence_id':'s1'}
    p.write_text(json.dumps({'records':[valid,bad]}), encoding='utf-8')
    ds = TemporalDataset.from_temporal_manifest(str(p), strict=False)
    assert len(ds) == 1
    assert ds.diagnostics['skipped_records'] >= 1
    assert ds.diagnostics['rejected_self_generated_targets'] >= 1


def test_temporal_refinement_synthetic_metadata_honest(tmp_path: Path) -> None:
    trainer = TemporalTrainer()
    result = trainer.train(TrainingConfig(epochs=1, train_size=2, val_size=1, checkpoint_dir='artifacts/checkpoints', learned_dataset_path=''))
    payload = json.loads(Path(result.checkpoint_path).read_text(encoding='utf-8'))
    assert payload['training_source'] == 'synthetic_temporal_bootstrap'
    assert payload['supervised_temporal_record_count'] == 0
    assert payload['target_policy']['target_source'] == 'synthetic_temporal_bootstrap'


def test_temporal_refinement_reload_from_result_checkpoint_relative_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = Path('temporal_manifest.json')
    repo_root = Path(__file__).resolve().parents[1]
    build_temporal_manifest_from_observed_sequences(observed_sequences_path=str(repo_root / 'examples' / 'temporal_sequences.example.json'), output_path=str(manifest), strict=True)
    trainer = TemporalTrainer()
    result = trainer.train(TrainingConfig(epochs=1, learned_dataset_path=str(manifest), checkpoint_dir='artifacts/checkpoints', renderer_target_role_policy='supervised_only'))
    backend = TrainableTemporalConsistencyBackend.from_checkpoint_policy(checkpoint_path=result.checkpoint_path, strict_checkpoint=True, strict_mode=True)
    assert backend.checkpoint_status()['temporal_checkpoint_loaded'] is True
