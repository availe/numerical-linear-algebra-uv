# Complete notebook source comparison

Original: `baccd5e44049a7265cc4708cce662f4942f14f09`. Output and runtime metadata differences are intentionally excluded; trailing whitespace in the displayed diffs is omitted for readability. Cell numbers below are one-based positions in the original notebook; stable IDs preserve that mapping. See [modernization decisions](modernization.md) for reasons.

## 0. Course Logistics.ipynb

35 original cells; 35 current cells.

No source/type changes.

## 1. Why are we here.ipynb

93 original cells; 93 current cells.

No source/type changes.

## 2. Topic Modeling with NMF and SVD.ipynb

180 original cells; 180 current cells.

### Original cell 8 (`cell-007`): code → code

```diff
--- original
+++ current
@@ -1,5 +1,9 @@
+import course_config
+course_config.configure_randomness()
+
 import numpy as np
 from sklearn.datasets import fetch_20newsgroups
 from sklearn import decomposition
+from sklearn.utils.extmath import randomized_svd as sklearn_randomized_svd
 from scipy import linalg
 import matplotlib.pyplot as plt
```

### Original cell 14 (`cell-013`): code → code

```diff
--- original
+++ current
@@ -1,4 +1,4 @@
 categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
 remove = ('headers', 'footers', 'quotes')
-newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=remove)
-newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=remove)
+newsgroups_train = fetch_20newsgroups(subset='train', data_home='data/sklearn', categories=categories, remove=remove)
+newsgroups_test = fetch_20newsgroups(subset='test', data_home='data/sklearn', categories=categories, remove=remove)
```

### Original cell 25 (`cell-024`): code → code

```diff
--- original
+++ current
@@ -1,3 +1,3 @@
 vectorizer = CountVectorizer(stop_words='english')
-vectors = vectorizer.fit_transform(newsgroups_train.data).todense() # (documents, vocab)
+vectors = vectorizer.fit_transform(newsgroups_train.data).toarray() # (documents, vocab)
 vectors.shape #, vectors.nnz / vectors.shape[0], row_means.shape
```

### Original cell 27 (`cell-026`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-vocab = np.array(vectorizer.get_feature_names())
+vocab = np.array(vectorizer.get_feature_names_out())
```

### Original cell 60 (`cell-059`): code → code

```diff
--- original
+++ current
@@ -1,4 +1,4 @@
-clf = decomposition.NMF(n_components=d, random_state=1)
+clf = decomposition.NMF(n_components=d, random_state=course_config.random_state(1))

 W1 = clf.fit_transform(vectors)
 H1 = clf.components_
```

### Original cell 84 (`cell-083`): code → code

```diff
--- original
+++ current
@@ -1,2 +1,2 @@
 def penalty(M, mu):
-    return np.where(M>=mu,0, np.min(M - mu, 0))
+    return np.minimum(M - mu, 0)
```

### Original cell 96 (`cell-095`): markdown → markdown

```diff
--- original
+++ current
@@ -1 +1 @@
-**Note about GPUs**: If you are not using a GPU, you will need to remove the `.cuda()` from the methods below. GPU usage is not required for this course, but I thought it would be of interest to some of you.  To learn how to create an AWS instance with a GPU, you can watch the [fast.ai setup lesson](http://course.fast.ai/lessons/aws.html).
+**Note about GPUs**: The code selects CUDA when available and otherwise runs on CPU. No edits are required to run without a GPU. GPU acceleration is optional; CPU execution of the optimization loops may take longer.
```

### Original cell 97 (`cell-096`): code → code

```diff
--- original
+++ current
@@ -1,3 +1,5 @@
 import torch
-import torch.cuda as tc
-from torch.autograd import Variable
+
+# CPU works on every supported platform; CUDA is optional.
+device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
+course_config.configure_torch(torch)
```

### Original cell 98 (`cell-097`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-def V(M): return Variable(M, requires_grad=True)
+def V(M): return M.detach().clone().requires_grad_(True)
```

### Original cell 99 (`cell-098`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-v=vectors_tfidf.todense()
+v=vectors_tfidf.toarray()
```

### Original cell 100 (`cell-099`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-t_vectors = torch.Tensor(v.astype(np.float32)).cuda()
+t_vectors = torch.as_tensor(v, dtype=torch.float32, device=device)
```

### Original cell 102 (`cell-101`): code → code

```diff
--- original
+++ current
@@ -4,7 +4,7 @@
         W.t().mm(R) + penalty_t(H, mu)*lam) # dW, dH

 def penalty_t(M, mu):
-    return (M<mu).type(tc.FloatTensor)*torch.clamp(M - mu, max=0.)
+    return torch.clamp(M - mu, max=0.)

 def upd_t(M, W, H, lr):
     dW,dH = grads_t(M,W,H)
```

### Original cell 103 (`cell-102`): code → code

```diff
--- original
+++ current
@@ -1,4 +1,2 @@
-t_W = tc.FloatTensor(m,d)
-t_H = tc.FloatTensor(d,n)
-t_W.normal_(std=0.01).abs_();
-t_H.normal_(std=0.01).abs_();
+t_W = torch.empty((m, d), device=device).normal_(std=0.01).abs_()
+t_H = torch.empty((d, n), device=device).normal_(std=0.01).abs_()
```

### Original cell 111 (`cell-110`): markdown → markdown

```diff
--- original
+++ current
@@ -2,4 +2,4 @@

 The approach we use below is very general, and would work for almost any optimization problem.

-In PyTorch, Variables have the same API as tensors, but Variables remember the operations used on to create them.  This lets us take derivatives.
+Modern PyTorch tensors with `requires_grad=True` remember the operations used to create them. This lets us take derivatives. The original lecture used the older `Variable` wrapper; tensors now provide this functionality directly.
```

### Original cell 114 (`cell-113`): code → code

```diff
--- original
+++ current
@@ -1,2 +1,2 @@
-x = Variable(torch.ones(2, 2), requires_grad=True)
+x = torch.ones(2, 2, requires_grad=True)
 print(x)
```

### Original cell 115 (`cell-114`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-print(x.data)
+print(x.detach())
```

### Original cell 123 (`cell-122`): code → code

```diff
--- original
+++ current
@@ -1,4 +1,2 @@
-pW = Variable(tc.FloatTensor(m,d), requires_grad=True)
-pH = Variable(tc.FloatTensor(d,n), requires_grad=True)
-pW.data.normal_(std=0.01).abs_()
-pH.data.normal_(std=0.01).abs_();
+pW = torch.empty((m, d), device=device).normal_(std=0.01).abs_().requires_grad_()
+pH = torch.empty((d, n), device=device).normal_(std=0.01).abs_().requires_grad_()
```

### Original cell 124 (`cell-123`): code → code

```diff
--- original
+++ current
@@ -1,9 +1,10 @@
 def report():
-    W,H = pW.data, pH.data
-    print((M-pW.mm(pH)).norm(2).data[0], W.min(), H.min(), (W<0).sum(), (H<0).sum())
+    W, H = pW.detach(), pH.detach()
+    print((M-W.mm(H)).norm(2).item(), W.min().item(), H.min().item(),
+          (W<0).sum().item(), (H<0).sum().item())

 def penalty(A):
-    return torch.pow((A<0).type(tc.FloatTensor)*torch.clamp(A, max=0.), 2)
+    return torch.clamp(A, max=0.).square()

 def penalize(): return penalty(pW).mean() + penalty(pH).mean()

```

### Original cell 125 (`cell-124`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-M = Variable(t_vectors).cuda()
+M = t_vectors.detach()
```

### Original cell 129 (`cell-128`): code → code

```diff
--- original
+++ current
@@ -1,2 +1,2 @@
-h = pH.data.cpu().numpy()
+h = pH.detach().cpu().numpy()
 show_topics(h)
```

### Original cell 146 (`cell-145`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-%time u, s, v = decomposition.randomized_svd(vectors, 5)
+%time u, s, v = sklearn_randomized_svd(vectors, 5)
```

### Original cell 150 (`cell-149`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-%time u, s, v = decomposition.randomized_svd(vectors, 5)
+%time u, s, v = sklearn_randomized_svd(vectors, 5)
```

Execution tag on original cell 168: `Complete the preceding error-versus-topic-count exercise first.`

### Original cell 178 (`cell-177`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-%time u, s, v = decomposition.randomized_svd(vectors, 5)
+%time u, s, v = sklearn_randomized_svd(vectors, 5)
```

### Original cell 179 (`cell-178`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-%time u, s, v = decomposition.randomized_svd(vectors.todense(), 5)
+%time u, s, v = sklearn_randomized_svd(vectors, 5, random_state=course_config.random_state())
```

## 3. Background Removal with Robust PCA.ipynb

187 original cells; 187 current cells.

Order change: original cell 54 (`cell-053`, sparse-component plot) now follows original cell 83 (`cell-082`, PCP computation); all other relative order is preserved.

### Original cell 7 (`cell-006`): code → code

```diff
--- original
+++ current
@@ -1,3 +1,3 @@
-import moviepy.editor as mpe
-# from IPython.display import display
-from glob import glob
+import moviepy as mpe
+from PIL import Image
+from course_data import ensure_data
```

### Original cell 8 (`cell-007`): code → code

```diff
--- original
+++ current
@@ -1,3 +1,6 @@
+import course_config
+course_config.configure_randomness()
+
 import sys, os
 import numpy as np
 import scipy
```

### Original cell 11 (`cell-010`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-video = mpe.VideoFileClip("data/Video_003.avi")
+video = mpe.VideoFileClip(str(ensure_data("Video_003.avi")))
```

### Original cell 12 (`cell-011`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-video.subclip(0,50).ipython_display(width=300)
+video.subclipped(0, min(50, video.duration)).display_in_notebook(width=300)
```

### Original cell 15 (`cell-014`): code → code

```diff
--- original
+++ current
@@ -1,3 +1,12 @@
 def create_data_matrix_from_video(clip, k=5, scale=50):
-    return np.vstack([scipy.misc.imresize(rgb2gray(clip.get_frame(i/float(k))).astype(int),
-                      scale).flatten() for i in range(k * int(clip.duration))]).T
+    size = tuple(max(1, int(d * scale / 100)) for d in clip.size)
+    frames = []
+    for i in range(k * int(clip.duration)):
+        gray = rgb2gray(clip.get_frame(i / float(k))).astype(int)
+        # Match the uint8 contrast scaling of the removed scipy.misc.imresize.
+        span = max(int(gray.max() - gray.min()), 1)
+        pixels = ((gray - gray.min()) * (255.0 / span)).clip(0, 255)
+        pixels = (pixels + 0.5).astype(np.uint8)
+        frames.append(np.asarray(Image.fromarray(pixels).resize(
+            size, Image.Resampling.BILINEAR), dtype=float).ravel())
+    return np.vstack(frames).T
```

### Original cell 17 (`cell-016`): code → code

```diff
--- original
+++ current
@@ -8,6 +8,6 @@
             sp.axis('Off')
             pixels = mat[:,i]
             if isinstance(pixels, scipy.sparse.csr_matrix):
-                pixels = pixels.todense()
+                pixels = pixels.toarray()
             plt.imshow(np.reshape(pixels, dims), cmap='gray')
     return f
```

### Original cell 21 (`cell-020`): code → code

```diff
--- original
+++ current
@@ -1,2 +1,2 @@
 scale = 25   # Adjust scale to change resolution of image
-dims = (int(240 * (scale/100)), int(320 * (scale/100)))
+dims = (int(video.h * scale/100), int(video.w * scale/100))
```

### Original cell 34 (`cell-033`): code → code

```diff
--- original
+++ current
@@ -1 +1,2 @@
 from sklearn import decomposition
+from sklearn.utils.extmath import randomized_svd
```

### Original cell 35 (`cell-034`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-u, s, v = decomposition.randomized_svd(M, 2)
+u, s, v = randomized_svd(M, 2)
```

### Original cell 44 (`cell-043`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-u, s, v = decomposition.randomized_svd(M, 1)
+u, s, v = randomized_svd(M, 1)
```

### Original cell 52 (`cell-051`): code → code

```diff
--- original
+++ current
@@ -1 +1,4 @@
-plt.imshow(np.reshape(M[:,140] - low_rank[:,140], dims)[50:150,100:270], cmap='gray');
+# The lecture's crop coordinates are in the original (unscaled) frame.
+roi = np.s_[int(50*scale/100):int(150*scale/100),
+            int(100*scale/100):int(270*scale/100)]
+plt.imshow(np.reshape(M[:,140] - low_rank[:,140], dims)[roi], cmap='gray');
```

### Original cell 54 (`cell-053`): code → code

```diff
--- original
+++ current
@@ -1 +1,2 @@
-plt.imshow(np.reshape(S[:,140], dims)[50:150,100:270], cmap='gray')
+# Compare with the robust-PCA foreground, after S has been computed below.
+plt.imshow(np.reshape(S[:,140], dims)[roi], cmap='gray')
```

### Original cell 79 (`cell-078`): code → code

```diff
--- original
+++ current
@@ -32,9 +32,9 @@
         Z = X - L - S
         Y += mu*Z; mu *= rho

-        examples.extend([S[140,:], L[140,:]])
+        examples.extend([S[140,:], L[140,:]] if trans else [S[:,140], L[:,140]])

-        if m > mu_bar: m = mu_bar
+        mu = min(mu, mu_bar)
         if converged(Z, d_norm): break

     if trans: L=L.T; S=S.T
```

### Original cell 84 (`cell-083`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-plots(examples, dims, rows=5)
+plots(examples, dims, rows=len(examples)//2)
```

### Original cell 100 (`cell-099`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-A = np.array([[2,1,1,0],[4,3,3,1],[8,7,9,5],[6,7,9,8]]).astype(np.float)
+A = np.array([[2,1,1,0],[4,3,3,1],[8,7,9,5],[6,7,9,8]]).astype(float)
```

Execution tag on original cell 115: `Define L1 and U1 in the preceding LU exercise first.`

Execution tag on original cell 116: `Define L1 and U1 in the preceding LU exercise first.`

### Original cell 138 (`cell-137`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-A = np.array([[2,1,1,0],[4,3,3,1],[8,7,9,5],[6,7,9,8]]).astype(np.float)
+A = np.array([[2,1,1,0],[4,3,3,1],[8,7,9,5],[6,7,9,8]]).astype(float)
```

Execution tag on original cell 139: `Implement LU_pivot in the preceding exercise first.`

Execution tag on original cell 141: `Implement LU_pivot in the preceding exercise first.`

Execution tag on original cell 142: `Implement LU_pivot in the preceding exercise first.`

Execution tag on original cell 143: `Implement LU_pivot in the preceding exercise first.`

## 4. Compressed Sensing of CT Scans with Robust Regression.ipynb

126 original cells; 126 current cells.

### Original cell 6 (`cell-005`): code → code

```diff
--- original
+++ current
@@ -1,3 +1,8 @@
+import course_config
+course_config.configure_randomness()
+
+import numpy as np
+
 a = np.array([1.0, 2.0, 3.0])
 b = 2.0
 a * b
```

### Original cell 42 (`cell-041`): code → code

```diff
--- original
+++ current
@@ -1,3 +1,5 @@
 %matplotlib inline
 import numpy as np, matplotlib.pyplot as plt, math
 from scipy import ndimage, sparse
+from pathlib import Path
+Path("data/figures").mkdir(parents=True, exist_ok=True)
```

### Original cell 68 (`cell-067`): code → code

```diff
--- original
+++ current
@@ -2,7 +2,7 @@
     x = np.ravel(x)
     floor_x = np.floor((x - orig) / dx)
     alpha = (x - orig - floor_x * dx) / dx
-    return np.hstack((floor_x, floor_x + 1)), np.hstack((1 - alpha, alpha))
+    return np.hstack((floor_x, floor_x + 1)).astype(np.intp), np.hstack((1 - alpha, alpha))


 def _generate_center_coordinates(l_x):
```

### Original cell 75 (`cell-074`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-proj_t = np.reshape(proj_operator.todense().A, (l//7,l,l,l))
+proj_t = np.reshape(proj_operator.toarray(), (l//7,l,l,l))
```

### Original cell 88 (`cell-087`): code → code

```diff
--- original
+++ current
@@ -1,4 +1,4 @@
 plt.figure(figsize=(5,5))
 plt.imshow(data, cmap=plt.cm.gray)
 plt.axis('off')
-plt.savefig("images/data.png")
+plt.savefig("data/figures/data.png")
```

### Original cell 91 (`cell-090`): code → code

```diff
--- original
+++ current
@@ -1,4 +1,4 @@
 plt.figure(figsize=(5,5))
 plt.imshow(data + proj_t[17,40], cmap=plt.cm.gray)
 plt.axis('off')
-plt.savefig("images/data_xray.png")
+plt.savefig("data/figures/data_xray.png")
```

### Original cell 109 (`cell-108`): code → code

```diff
--- original
+++ current
@@ -1,4 +1,4 @@
 plt.figure(figsize=(7,7))
 plt.imshow(np.resize(proj, (l//7,l)), cmap='gray')
 plt.axis('off')
-plt.savefig("images/proj.png")
+plt.savefig("data/figures/proj.png")
```

### Original cell 114 (`cell-113`): code → code

```diff
--- original
+++ current
@@ -1,3 +1,3 @@
 plt.figure(figsize=(12,12))
 plt.title("A: Projection Operator")
-plt.imshow(proj_operator.todense().A, cmap='gray')
+plt.imshow(proj_operator.toarray(), cmap='gray')
```

### Original cell 123 (`cell-122`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-18 x 128 x 128 x 128
+18 * 128 * 128 * 128
```

## 5. Health Outcomes with Linear Regression.ipynb

91 original cells; 91 current cells.

### Original cell 3 (`cell-002`): code → code

```diff
--- original
+++ current
@@ -1,3 +1,6 @@
+import course_config
+course_config.configure_randomness()
+
 from sklearn import datasets, linear_model, metrics
 from sklearn.model_selection import train_test_split
 from sklearn.preprocessing import PolynomialFeatures
```

### Original cell 9 (`cell-008`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-trn,test,y_trn,y_test = train_test_split(data.data, data.target, test_size=0.2)
+trn,test,y_trn,y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=course_config.random_state())
```

### Original cell 25 (`cell-024`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-', '.join(poly.get_feature_names(feature_names))
+', '.join(poly.get_feature_names_out(feature_names))
```

### Original cell 39 (`cell-038`): code → code

```diff
--- original
+++ current
@@ -1,3 +1,2 @@
 import math, numpy as np, matplotlib.pyplot as plt
-from pandas_summary import DataFrameSummary
 from scipy import ndimage
```

### Original cell 79 (`cell-078`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-reg_regr = linear_model.LassoCV(n_alphas=10)
+reg_regr = linear_model.LassoCV(alphas=10)
```

## 6. How to Implement Linear Regression.ipynb

142 original cells; 142 current cells.

### Original cell 4 (`cell-003`): code → code

```diff
--- original
+++ current
@@ -1,3 +1,6 @@
+import course_config
+course_config.configure_randomness()
+
 from sklearn import datasets, linear_model, metrics
 from sklearn.model_selection import train_test_split
 from sklearn.preprocessing import PolynomialFeatures
```

### Original cell 8 (`cell-007`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-trn,test,y_trn,y_test = train_test_split(data.data, data.target, test_size=0.2)
+trn,test,y_trn,y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=course_config.random_state())
```

### Original cell 71 (`cell-070`): code → code

```diff
--- original
+++ current
@@ -8,7 +8,7 @@
             for name in row_names:
                 fcn = name2func[name]
                 t = timeit.timeit(fcn + '(A,b)', number=5, globals=globals())
-                df.set_value(name, (m,n), t)
+                df.at[name, (m,n)] = t
                 coeffs = locals()[fcn](A, b)
                 reg_met = regr_metrics(b, A @ coeffs)
-                df_error.set_value(name, (m,n), reg_met[0])
+                df_error.at[name, (m,n)] = reg_met[0]
```

### Original cell 74 (`cell-073`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-store = pd.HDFStore('least_squares_results.h5')
+# A context manager closes the HDF5 file even if writing fails.
```

### Original cell 75 (`cell-074`): code → code

```diff
--- original
+++ current
@@ -1 +1,2 @@
-store['df'] = df
+with pd.HDFStore('least_squares_results.h5') as store:
+    store['df'] = df
```

### Original cell 106 (`cell-105`): code → code

```diff
--- original
+++ current
@@ -2,5 +2,5 @@
     fcn = name2func[name]
     t = timeit.timeit(fcn + '(A,b)', number=5, globals=globals())
     coeffs = locals()[fcn](A, b)
-    df.set_value(name, 'Time', t)
-    df.set_value(name, 'Error', regr_metrics(b, A @ coeffs)[0])
+    df.at[name, 'Time'] = t
+    df.at[name, 'Error'] = regr_metrics(b, A @ coeffs)[0]
```

### Original cell 128 (`cell-127`): code → code

```diff
--- original
+++ current
@@ -2,5 +2,5 @@
     fcn = name2func[name]
     t = timeit.timeit(fcn + '(A,b)', number=5, globals=globals())
     coeffs = locals()[fcn](A, b)
-    df.set_value(name, 'Time', t)
-    df.set_value(name, 'Error', np.abs(1 - coeffs[-1]))
+    df.at[name, 'Time'] = t
+    df.at[name, 'Error'] = np.abs(1 - coeffs[-1])
```

### Original cell 137 (`cell-136`): code → code

```diff
--- original
+++ current
@@ -1,6 +1,11 @@
 for name in row_names:
     fcn = name2func[name]
-    t = timeit.timeit(fcn + '(A,b)', number=5, globals=globals())
-    coeffs = locals()[fcn](A, b)
-    df.set_value(name, 'Time', t)
-    df.set_value(name, 'Error', regr_metrics(b, A @ coeffs)[0])
+    try:
+        t = timeit.timeit(fcn + '(A,b)', number=5, globals=globals())
+        coeffs = locals()[fcn](A, b)
+        df.at[name, 'Time'] = t
+        df.at[name, 'Error'] = regr_metrics(b, A @ coeffs)[0]
+    except np.linalg.LinAlgError as exc:
+        # This rank-deficient example illustrates why naive inversion fails.
+        print(f"{name}: {exc}")
+        df.loc[name] = np.nan
```

## 7. PageRank with Eigen Decompositions.ipynb

207 original cells; 207 current cells.

### Original cell 6 (`cell-005`): code → code

```diff
--- original
+++ current
@@ -1 +1,5 @@
+import course_config
+course_config.configure_randomness()
+
+import os
 import psutil
```

### Original cell 35 (`cell-034`): code → code

```diff
--- original
+++ current
@@ -3,9 +3,9 @@
 from datetime import datetime
 from pprint import pprint
 from time import time
-from tqdm import tqdm_notebook
+from tqdm.auto import tqdm as tqdm_notebook
 from scipy import sparse

-from sklearn.decomposition import randomized_svd
-from sklearn.externals.joblib import Memory
+from sklearn.utils.extmath import randomized_svd
+from joblib import Memory
 from urllib.request import urlopen
```

### Original cell 39 (`cell-038`): code → code

```diff
--- original
+++ current
@@ -1,8 +1,6 @@
+from course_data import ensure_data
+
 PATH = 'data/dbpedia/'
-URL_BASE = 'http://downloads.dbpedia.org/3.5.1/en/'
 filenames = ["redirects_en.nt.bz2", "page_links_en.nt.bz2"]
-
 for filename in filenames:
-    if not os.path.exists(PATH+filename):
-        print("Downloading '%s', please wait..." % filename)
-        open(PATH+filename, 'wb').write(urlopen(URL_BASE+filename).read())
+    ensure_data('dbpedia/' + filename)
```

### Original cell 50 (`cell-049`): code → code

```diff
--- original
+++ current
@@ -1,5 +1,4 @@
 def get_redirects(redirects_filename):
-    redirects={}
-    lines = get_lines(redirects_filename)
-    return {src[SLICE]:get_redirect(targ[SLICE], redirects)
-                for src,_,targ,_ in tqdm_notebook(lines, leave=False)}
+    redirects = {src[SLICE]: targ[SLICE]
+                 for src, _, targ, _ in tqdm_notebook(get_lines(redirects_filename), leave=False)}
+    return {src: get_redirect(targ, redirects) for src, targ in redirects.items()}
```

### Original cell 55 (`cell-054`): code → code

```diff
--- original
+++ current
@@ -1,7 +1,9 @@
 # Computing the integer index map
 index_map = dict() # links->IDs
 lines = get_lines(page_links_filename)
-source, destination, data = [],[],[]
+# Packed arrays store the same edge lists without Python-int overhead.
+from array import array
+source, destination, data = array('I'), array('I'), array('f')
 for l, split in tqdm_notebook(enumerate(lines), total=limit):
     if l >= limit: break
     add_item(source, redirects, index_map, split[0])
```

### Original cell 56 (`cell-055`): code → code

```diff
--- original
+++ current
@@ -1 +1,2 @@
-n=len(data); n
+n = len(index_map)  # adjacency dimensions count pages, not links
+n, len(data)
```

### Original cell 60 (`cell-059`): code → code

```diff
--- original
+++ current
@@ -1 +1,4 @@
-index_map.popitem()
+# Inspect the same page as the lecture, without deleting its entry.
+example_page = b'1940_Cincinnati_Reds_Team_Issue'
+example_index = index_map[example_page]
+example_page, example_index
```

### Original cell 62 (`cell-061`): markdown → markdown

```diff
--- original
+++ current
@@ -1 +1 @@
-1940_Cincinnati_Reds_Team_Issue has index $9991173$.  This only shows up once in the destination list:
+1940_Cincinnati_Reds_Team_Issue has index `example_index`. This only shows up once in the source list:
```

### Original cell 63 (`cell-062`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-[i for i,x in enumerate(source) if x == 9991173]
+[i for i,x in enumerate(source) if x == example_index]
```

### Original cell 64 (`cell-063`): code → code

```diff
--- original
+++ current
@@ -1 +1,2 @@
-source[119077649], destination[119077649]
+example_source, example_target = source[119077649], destination[119077649]
+example_source, example_target
```

### Original cell 65 (`cell-064`): markdown → markdown

```diff
--- original
+++ current
@@ -1 +1 @@
-Now, we want to check which page is the source (has index $9991050$).  Note: usually you should not access a dictionary by searching for its values.  This is inefficient and not how dictionaries are intended to be used.
+Now, we want to check which page has index `example_target`.  Note: usually you should not access a dictionary by searching for its values.  This is inefficient and not how dictionaries are intended to be used.
```

### Original cell 66 (`cell-065`): code → code

```diff
--- original
+++ current
@@ -1,3 +1,3 @@
 for page_name, index in index_map.items():
-    if index == 9991050:
+    if index == example_target:
         print(page_name)
```

### Original cell 68 (`cell-067`): code → code

```diff
--- original
+++ current
@@ -1 +1 @@
-test_inds = [i for i,x in enumerate(source) if x == 9991050]
+test_inds = [i for i,x in enumerate(source) if x == example_target]
```

### Original cell 72 (`cell-071`): markdown → markdown

```diff
--- original
+++ current
@@ -1 +1 @@
-Now, we want to check which page is the source (has index 9991174):
+Now, we want to check which pages have the destination indices in `test_dests`:
```

### Original cell 100 (`cell-099`): code → code

```diff
--- original
+++ current
@@ -1,6 +1,6 @@
 def power_method(A, max_iter=100):
     n = A.shape[1]
-    A = np.copy(A)
+    A = A.astype(np.float64, copy=True)
     A.data /= np.take(A.sum(axis=0).A1, A.indices)

     scores = np.ones(n, dtype=np.float32) * np.sqrt(A.sum()/(n*n)) # initial guess
```

Execution tag on original cell 147: `Implement practical_qr in the preceding exercise / Homework 3 first.`

Execution tag on original cell 149: `Implement practical_qr in the preceding exercise / Homework 3 first.`

### Original cell 170 (`cell-169`): code → markdown

```diff
(Source unchanged; cell type corrected.)
```

Execution tag on original cell 192: `Implement practical_qr in the preceding exercise / Homework 3 first.`

Execution tag on original cell 194: `Implement practical_qr in the preceding exercise / Homework 3 first.`

Execution tag on original cell 195: `Implement practical_qr in the preceding exercise / Homework 3 first.`

Execution tag on original cell 197: `Implement practical_qr in the preceding exercise / Homework 3 first.`

Execution tag on original cell 198: `Implement practical_qr in the preceding exercise / Homework 3 first.`

## 8. Implementing QR Factorization.ipynb

96 original cells; 96 current cells.

### Original cell 6 (`cell-005`): code → code

```diff
--- original
+++ current
@@ -1,3 +1,6 @@
+import course_config
+course_config.configure_randomness()
+
 import numpy as np

 np.set_printoptions(suppress=True, precision=4)
```

### Original cell 42 (`cell-041`): code → code

```diff
--- original
+++ current
@@ -1 +1,2 @@
+R, V, F = householder_lots(A)
 R
```

### Original cell 59 (`cell-058`): code → code

```diff
--- original
+++ current
@@ -1,4 +1,6 @@
-def implicit_Qx(V,x):
-    n = len(x)
-    for k in range(n-1,-1,-1):
-        x[k:n] -= 2*np.matmul(v[-k], np.matmul(v[-k], x[k:n]))
+def implicit_Qx(V, x):
+    x = np.array(x, dtype=float, copy=True)
+    for k in range(len(V)-1, -1, -1):
+        v = V[k].ravel()
+        x[k:] -= 2 * v * (v @ x[k:])
+    return x
```

### Original cell 74 (`cell-073`): code → code

```diff
--- original
+++ current
@@ -1,6 +1,6 @@
 plt.figure(figsize=(10,10))
-plt.semilogy(np.diag(S), 'r.', basey=2, label="True Singular Values")
-plt.semilogy(np.diag(RM), 'go', basey=2, label="Modified Gram-Shmidt")
-plt.semilogy(np.diag(RC), 'bx', basey=2, label="Classic Gram-Shmidt")
+plt.semilogy(np.diag(S), 'r.', base=2, label="True Singular Values")
+plt.semilogy(np.diag(RM), 'go', base=2, label="Modified Gram-Shmidt")
+plt.semilogy(np.diag(RC), 'bx', base=2, label="Classic Gram-Shmidt")
 plt.legend()
 rcParams.update({'font.size': 18})
```

## Homework 1.ipynb

9 original cells; 9 current cells.

No source/type changes.

## Homework 2.ipynb

9 original cells; 9 current cells.

### Original cell 4 (`cell-003`): code → code

```diff
--- original
+++ current
@@ -1,3 +1,5 @@
+import numpy as np
+
 def LU(A):
     U = np.copy(A)
     m, n = A.shape
```

## Homework 3.ipynb

4 original cells; 4 current cells.

No source/type changes.

## convolution-intro.ipynb

44 original cells; 44 current cells.

### Original cell 3 (`cell-002`): code → code

```diff
--- original
+++ current
@@ -3,10 +3,9 @@
 from numpy.linalg import norm
 from PIL import Image
 from matplotlib import pyplot as plt, rcParams, rc
-from scipy.ndimage import imread
 from skimage.measure import block_reduce
 import pickle as pickle
-from scipy.ndimage.filters import correlate, convolve
+from scipy.ndimage import correlate, convolve
 rc('animation', html='html5')
 rcParams['figure.figsize'] = 3, 6
 %precision 4
```

### Original cell 6 (`cell-005`): code → code

```diff
--- original
+++ current
@@ -1,2 +1,2 @@
-from sklearn.datasets import fetch_mldata
-mnist = fetch_mldata('MNIST original')
+from sklearn.datasets import fetch_openml
+mnist = fetch_openml(data_id=554, as_frame=False, data_home='data/sklearn')  # MNIST original, version 1
```

### Original cell 31 (`cell-030`): code → code

```diff
--- original
+++ current
@@ -1,2 +1,6 @@
+def pool(im):
+    return block_reduce(im, block_size=(7, 7), func=np.max)
+
+pool8 = [np.array([pool(correlate(im, rot)) for im in eights]) for rot in rots]
 filts8 = np.array([ims.mean(axis=0) for ims in pool8])
 filts8 = normalize(filts8)
```

## gradient-descent-intro.ipynb

14 original cells; 14 current cells.

### Original cell 2 (`cell-001`): code → code

```diff
--- original
+++ current
@@ -1,11 +1,15 @@
+import course_config
+course_config.configure_randomness()
+
 %matplotlib inline
 import math,sys,os,numpy as np
 from numpy.random import random
 from matplotlib import pyplot as plt, rcParams, animation, rc
-from __future__ import print_function, division
 from ipywidgets import interact, interactive, fixed
 from ipywidgets.widgets import *
 rc('animation', html='html5')
 rcParams['figure.figsize'] = 3, 3
 %precision 4
 np.set_printoptions(precision=4, linewidth=100)
+import imageio_ffmpeg
+rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
```

## Non-notebook teaching material

All original files were compared byte-for-byte. Differences below do not include added environment, helper, test, and documentation files.

- Changed: `.gitignore`
- Changed: `README.md`

82 original non-notebook files are byte-identical (including all images, spreadsheet, and project ideas).

## README changes against the user-authored version

```diff
--- user README
+++ current README
@@ -12,18 +12,20 @@
 Install [uv](https://docs.astral.sh/uv/), then from the project root run:

 ```bash
-uv sync
+uv sync --locked
 ```

 You may use whichever method you wish to run the notebooks. If you prefer JupyterLab, you can launch it from the terminal via the following command:

 ```bash
-uv run jupyter lab
+uv run --locked jupyter lab
 ```
+
+Repeatability is controlled by `REPRODUCIBLE` in `nbs/course_config.py`. See [modernization notes](docs/modernization.md) for the exact changes, original reference outputs, data sources, and validation.

 ## Optional: cleaner Git diffs for Jupyter notebooks

-This tells Git to ignore noisy Jupyter execution metadata such as execution counts and execution timing while preserving notebook outputs.
+This keeps notebook outputs and execution metadata local, so running cells does not change what Git records. Code, Markdown, cell order, and attachments are preserved.

 This is optional and is **not required to set up the repository**.

```
