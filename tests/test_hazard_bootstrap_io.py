from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import sys
import zipfile
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('hazard_bootstrap_io',ROOT/'notebooks/sidewalk_data_3_support/hazard_bootstrap_io.py')
run_io=importlib.util.module_from_spec(spec);spec.loader.exec_module(run_io)


def test_drive_download_is_anonymous_and_hash_checked(tmp_path,monkeypatch):
    payload=b'pretend dataset archive'
    digest=hashlib.sha256(payload).hexdigest()
    calls=[]
    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs['use_cookies'] is False and kwargs['resume'] is True
        Path(kwargs['output']).write_bytes(payload)
        return kwargs['output']
    monkeypatch.setitem(sys.modules,'gdown',SimpleNamespace(download=download))
    path=tmp_path/'data.tar'
    assert run_io.download_drive_archive('https://drive.google.com/file/d/abc/view',path,digest)==path
    assert path.read_bytes()==payload and not path.with_name('data.tar.download').exists()
    # A verified cached archive needs neither a link nor another request.
    run_io.download_drive_archive('',path,digest)
    assert len(calls)==1


def test_bad_download_is_not_accepted_or_reused(tmp_path,monkeypatch):
    def download(**kwargs):
        Path(kwargs['output']).write_bytes(b'not the expected archive');return kwargs['output']
    monkeypatch.setitem(sys.modules,'gdown',SimpleNamespace(download=download))
    path=tmp_path/'data.tar'
    with pytest.raises(ValueError,match='SHA-256'):
        run_io.download_drive_archive('https://drive.google.com/file/d/abc/view',path,'0'*64)
    assert not path.exists() and not path.with_name('data.tar.download').exists()


def test_drive_download_failure_does_not_fall_back_to_login(tmp_path,monkeypatch):
    def download(**kwargs):raise RuntimeError('private file')
    monkeypatch.setitem(sys.modules,'gdown',SimpleNamespace(download=download))
    with pytest.raises(RuntimeError,match='No Google sign-in'):
        run_io.download_drive_archive('https://drive.google.com/file/d/abc/view',tmp_path/'data.tar','0'*64)


@pytest.mark.parametrize('url',['https://drive.google.com/drive/folders/abc','http://drive.google.com/file/d/abc','https://example.com/file'])
def test_rejects_wrong_link_type(tmp_path,url):
    with pytest.raises(ValueError):run_io.download_drive_archive(url,tmp_path/'data.tar','0'*64)


def make_run(root):
    root.mkdir()
    (root/'config.json').write_text('{"settings":{"n_members":10}}')
    (root/'predictions.csv').write_text('image,p_hazard\nx,0.8\n')
    (root/'members/head_01').mkdir(parents=True)
    (root/'members/head_01/head.pt').write_bytes(b'saved weights')
    (root/'features').mkdir()
    (root/'features/pitt-train.pt').write_bytes(b'cached embeddings')
    (root/'inference/backbone').mkdir(parents=True)
    (root/'inference/backbone/model.safetensors').write_bytes(b'frozen encoder')
    (root/'inference/ensemble.pt').write_bytes(b'all ten head weights')
    return root


def test_zip_round_trip_keeps_weights_features_and_provenance(tmp_path):
    root=make_run(tmp_path/'run1')
    (root/'incomplete.tmp').write_text('skip')
    archive=run_io.make_results_zip(root,tmp_path/'downloads')
    assert archive.exists() and archive.with_suffix('.zip.sha256').exists()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert 'run1/inference/ensemble.pt' in z.namelist()
        assert 'run1/features/pitt-train.pt' in z.namelist()
        assert 'run1/incomplete.tmp' not in z.namelist()
    restored=run_io.restore_results_zip(archive,tmp_path/'restored')
    for relative,info in json.loads((root/'archive_manifest.json').read_text()).items():
        assert (restored/relative).read_bytes()==(root/relative).read_bytes()
        assert run_io.file_sha(restored/relative)==info['sha256']
    with pytest.raises(FileExistsError):run_io.restore_results_zip(archive,tmp_path/'restored')
    (root/'predictions.csv').write_text('updated')
    refreshed=run_io.make_results_zip(root,tmp_path/'downloads')
    with zipfile.ZipFile(refreshed) as z:assert z.read('run1/predictions.csv')==b'updated'


def test_zip_cannot_include_itself_or_symlinked_files(tmp_path):
    root=make_run(tmp_path/'run1')
    with pytest.raises(AssertionError):run_io.make_results_zip(root,root/'downloads')
    outside=tmp_path/'credential';outside.write_text('private fixture')
    (root/'outside').symlink_to(outside)
    with pytest.raises(AssertionError):run_io.make_results_zip(root,tmp_path/'downloads')


def test_restore_rejects_path_traversal(tmp_path):
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('../outside','bad')
    with pytest.raises(ValueError,match='Unsafe path'):run_io.restore_results_zip(archive,tmp_path/'restore')
    assert not (tmp_path/'outside').exists()


def test_restore_rejects_tampered_results(tmp_path):
    root=make_run(tmp_path/'run1')
    source=run_io.make_results_zip(root,tmp_path/'downloads')
    altered=tmp_path/'tampered.zip'
    with zipfile.ZipFile(source) as z,zipfile.ZipFile(altered,'w') as out:
        for info in z.infolist():
            content=b'wrong weights' if info.filename.endswith('head.pt') else z.read(info.filename)
            out.writestr(info,content)
    with pytest.raises(ValueError,match='integrity verification'):
        run_io.restore_results_zip(altered,tmp_path/'restore')
    assert not (tmp_path/'restore/run1').exists()
