"""Exercise migrated APIs and algorithms directly from the teaching notebooks."""
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
from IPython.core.inputtransformer2 import TransformerManager

NBS = Path(__file__).resolve().parents[1] / 'nbs'


def cell(prefix, index):
    path = next(NBS.glob(prefix + '*.ipynb'))
    return ''.join(json.loads(path.read_text())['cells'][index]['source'])


def execute(prefix, indices, namespace=None):
    ns = {} if namespace is None else namespace
    for index in indices:
        exec(TransformerManager().transform_cell(cell(prefix, index)), ns)
    return ns


@pytest.mark.parametrize('module', [
    'numpy','scipy','sklearn','matplotlib','pandas','numba','torch','moviepy',
    'PIL','imageio_ffmpeg','skimage','fbpca','psutil','tqdm','joblib','tables',
    'ipywidgets','ipykernel','jupyterlab',
])
def test_dependencies_import(module):
    importlib.import_module(module)


def test_torch_cpu_manual_and_autograd():
    import torch
    ns = dict(torch=torch, np=np, device=torch.device('cpu'), m=8, n=10, d=3,
              mu=1e-5, lam=100, lr=1e-3)
    execute('2.', [101,102], ns)
    M = torch.rand(8,10)
    ns['upd_t'](M, ns['t_W'], ns['t_H'], ns['lr'])
    assert torch.isfinite(ns['t_W']).all()
    ns.update(t_vectors=M)
    execute('2.', [122,123,124], ns)
    loss = ns['loss']()
    loss.backward()
    assert ns['pW'].grad is not None
    assert torch.isfinite(ns['pW'].grad).all()
    execute('2.', [83], ns)
    np.testing.assert_allclose(ns['penalty'](np.array([[-1., 2.], [3., -4.]]), 0), [[-1,0],[0,-4]])


def test_video_resize_and_robust_pca(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = execute('3.', [6,7,10,14,15,71,72,73,74,75,76,77,78])
    matrix = ns['create_data_matrix_from_video'](ns['video'], k=1, scale=10)
    assert matrix.shape == (24*32,12)
    # PCP's examples refer to frame 140, so provide enough frames.
    background = np.ones((12,160))
    foreground = np.zeros_like(background); foreground[2:4,30:40] = 3
    L,S,_ = ns['pcp'](background+foreground, maxiter=30, k=1)
    assert np.linalg.norm(background+foreground-L-S)/np.linalg.norm(background+foreground) < 0.02
    ns['video'].close()


def test_qr_implicit_product():
    ns = execute('8.', [52,58], {'np':np})
    A = np.random.default_rng(42).normal(size=(5,5))
    R,V = ns['householder'](A)
    Q = np.column_stack([ns['implicit_Qx'](V,x) for x in np.eye(5)])
    np.testing.assert_allclose(Q@R,A,atol=1e-12)
    np.testing.assert_allclose(Q.T@Q,np.eye(5),atol=1e-12)


def test_sparse_pagerank_and_redirects():
    from scipy import sparse
    from bz2 import BZ2File
    ns = execute('7.', [45,48,49,99], {'np':np, 'tqdm_notebook':lambda x,**kw:x})
    ns['get_lines'] = lambda _: iter([
        [b'<http://dbpedia.org/resource/a>',b'p',b'<http://dbpedia.org/resource/b>',b'.'],
        [b'<http://dbpedia.org/resource/b>',b'p',b'<http://dbpedia.org/resource/c>',b'.']])
    assert ns['get_redirects']('unused') == {b'a':b'c',b'b':b'c'}
    A = sparse.csr_matrix([[0.,1.,1.],[1.,0.,1.],[1.,1.,0.]])
    before=A.copy()
    scores=ns['power_method'](A,max_iter=10)
    np.testing.assert_allclose(scores,np.ones(3)/np.sqrt(3))
    np.testing.assert_array_equal(A.toarray(),before.toarray())


def test_pandas_hdf_and_scalar_assignment(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.chdir(tmp_path)
    ns={'pd':pd, 'df':pd.DataFrame(index=['QR'], columns=['Time','Error'], dtype=float)}
    ns['df'].at['QR','Time']=0.1
    execute('6.',[74],ns)
    result=pd.read_hdf('least_squares_results.h5','df')
    assert result.at['QR','Time']==0.1


def test_git_filter_preserves_lesson_outputs(tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    root = NBS.parent
    repo = tmp_path/'checkout with spaces'
    (repo/'scripts').mkdir(parents=True)
    for name in ['pyproject.toml','uv.lock','.python-version','.gitattributes']:
        shutil.copy(root/name,repo/name)
    shutil.copy(root/'scripts/setup-git.sh',repo/'scripts/setup-git.sh')
    # Existing interpreter is reused: no downloads or synchronization during Git.
    try:
        (repo/'.venv').symlink_to(Path(sys.prefix), target_is_directory=True)
    except OSError:
        pytest.skip('This platform does not permit the temporary environment symlink.')
    env = dict(os.environ, UV_CACHE_DIR=str(tmp_path/'uv-cache'))
    def run(*args, input=None):
        return subprocess.run(args, cwd=repo, env=env, input=input,
                              text=True, capture_output=True, check=True).stdout
    run('git','init','-q')
    for _ in range(2):
        subprocess.run(['bash',str(repo/'scripts/setup-git.sh')],cwd=tmp_path,env=env,check=True,capture_output=True)
    import nbformat
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        source='print(42)', execution_count=7,
        metadata={'collapsed':True,'execution':{'iopub.status.busy':'yesterday'}},
        outputs=[nbformat.v4.new_output('stream',name='stdout',text='42\n')])])
    original_id=nb.cells[0].id
    oid=run('git','hash-object','-w','--stdin','--path=lesson.ipynb',input=nbformat.writes(nb)).strip()
    cleaned=nbformat.reads(run('git','cat-file','-p',oid),as_version=4)
    assert cleaned.cells[0].outputs == nb.cells[0].outputs
    assert cleaned.cells[0].execution_count is None
    assert cleaned.cells[0].id == original_id
    assert cleaned.cells[0].metadata['collapsed'] is True
    assert 'execution' not in cleaned.cells[0].metadata


def test_library_and_teaching_svd_do_not_shadow_each_other():
    from scipy import linalg
    from sklearn.utils.extmath import randomized_svd as sklearn_randomized_svd
    vectors = np.random.default_rng(42).normal(size=(20,15))
    ns = {'np':np, 'linalg':linalg, 'vectors':vectors,
          'sklearn_randomized_svd':sklearn_randomized_svd}
    execute('2.',[157,159],ns)
    teaching_svd = ns['randomized_svd']
    for i in [145,149,177,178]:
        exec(cell('2.',i).replace('%time ',''),ns)
        assert ns['u'].shape==(20,5)
        assert ns['s'].shape==(5,)
        assert ns['v'].shape==(5,15)
    assert ns['randomized_svd'] is teaching_svd
