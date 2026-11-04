"""Calibration-only channel smoothing and floating outlier compensation.

The native replay consumes W8A8 codes directly. simulate() is a numerical
reference using float BLAS; it is never evidence of INT8 inference speed.
The AWQ-style search uses mean-absolute-activation channel scaling adapted to
W8A8. It is not the official weight-only AWQ implementation or its accuracy claim.
Source: https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_scale.py
"""
import numpy as np


def prepare(weights, activation_max, alpha=None, outlier_count=0, method="smoothquant", activation_mean=None):
    w=np.asarray(weights,dtype=np.float32)
    a=np.asarray(activation_max,dtype=np.float32)
    if w.ndim!=2 or a.shape!=(w.shape[1],) or not np.isfinite(w).all() or not np.isfinite(a).all() or (a<0).any():
        raise ValueError("finite weight matrix and nonnegative channel maxima required")
    if alpha is not None and not 0<=alpha<=1:raise ValueError("alpha must be in [0,1]")
    if not 0<=outlier_count<w.shape[1]:raise ValueError("invalid outlier count")
    if method not in ("smoothquant","awq"):raise ValueError("unknown scaling method")
    if method=="awq":
        mean=np.asarray(activation_mean,dtype=np.float32)
        if mean.shape!=a.shape or not np.isfinite(mean).all() or (mean<0).any():
            raise ValueError("AWQ-style scaling requires calibration mean absolute activations")
        smooth=np.maximum(np.maximum(mean,1e-8)**(0 if alpha is None else alpha),1e-4)
        smooth/=np.sqrt(smooth.max()*smooth.min())
    else:
        smooth=np.ones_like(a) if alpha is None else np.clip(np.maximum(a,1e-5)**alpha/np.maximum(np.max(np.abs(w),axis=0),1e-5)**(1-alpha),1e-3,1e3)
    order=np.argsort(-(a/smooth),kind="stable")
    indices=np.sort(order[:outlier_count]).astype(np.int32)
    inverse=(1/smooth).astype(np.float32);inverse[indices]=0
    scaled=w*smooth
    scaled[:,indices]=0
    scales=np.max(np.abs(scaled),axis=1)/np.float32(127)
    scales=np.where(scales>0,scales,1).astype(np.float32)
    codes=np.clip(np.rint(scaled/scales[:,None]),-127,127).astype(np.int8)
    return dict(codes=codes,scales=scales,inverse=inverse,indices=indices,
                extra=np.ascontiguousarray(w[:,indices],dtype=np.float16),
                alpha=alpha,outlier_count=outlier_count,method=method)


def simulate(inputs, prepared):
    x=np.asarray(inputs,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=prepared["codes"].shape[1] or not np.isfinite(x).all():
        raise ValueError("finite compatible input matrix required")
    scaled=x*prepared["inverse"]
    scale=np.max(np.abs(scaled),axis=1)/np.float32(127)
    scale=np.where(scale>0,scale,1).astype(np.float32)
    codes=np.clip(np.rint(scaled/scale[:,None]),-127,127).astype(np.int8)
    # Dequantized float GEMM is a reference, not the measured native path.
    result=(codes.astype(np.float32)*scale[:,None]) @ (prepared["codes"].astype(np.float32)*prepared["scales"][:,None]).T
    if prepared["outlier_count"]:
        result+=x[:,prepared["indices"]].astype(np.float16).astype(np.float32) @ prepared["extra"].astype(np.float32).T
    return result


def error_metrics(reference,candidate):
    ref=np.asarray(reference,np.float64);got=np.asarray(candidate,np.float64)
    if ref.shape!=got.shape or ref.ndim!=2 or not np.isfinite(ref).all() or not np.isfinite(got).all():
        raise ValueError("finite matching matrices required")
    err=np.linalg.norm(got-ref);norm=np.linalg.norm(ref)
    rn=np.linalg.norm(ref,axis=1);gn=np.linalg.norm(got,axis=1)
    cosine=np.sum(ref*got,axis=1)/np.maximum(rn*gn,1e-30)
    cosine[(rn==0)&(gn==0)]=1
    return dict(relative_l2=float(err/max(norm,1e-30)),minimum_token_cosine=float(np.min(cosine)),
                mean_token_cosine=float(np.mean(cosine)),max_abs=float(np.max(np.abs(got-ref))))
