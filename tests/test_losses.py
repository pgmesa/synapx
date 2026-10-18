import pytest

import torch
import synapx

from utils import check_tensors


def test_MSELoss():
    true = [1.0, 0]
    pred = [[2.0], [-1.0]]
    
    # synapx
    y_true = synapx.tensor(true)
    inp = synapx.tensor(pred, requires_grad=True)
    
    y_pred = synapx.sigmoid(inp)
    y_pred = y_pred.squeeze()
    loss = synapx.nn.MSELoss(reduction='mean')(y_pred, y_true)
    loss.backward()
    
    print(loss)
    
    # torch
    y_true_t = torch.tensor(true)
    inp_t = torch.tensor(pred, requires_grad=True)
    
    y_pred_t = torch.sigmoid(inp_t)
    y_pred_t = y_pred_t.squeeze()
    loss_t = torch.nn.MSELoss(reduction='mean')(y_pred_t, y_true_t)
    loss_t.backward()
    
    print(loss_t)
    
    assert check_tensors(loss, loss_t)
    assert check_tensors(inp.grad, inp_t.grad)
    
    
def test_NLLLoss():
    pred = [[0.14, 0.3],[-0.2, 0.9],[-3, 0.1]]
    label = [0, 1, 0]
    
    # synapx
    ypred = synapx.tensor(pred, requires_grad=True)
    log_soft = synapx.log_softmax(ypred, dim=1)
    ylabel = synapx.tensor(label, dtype=torch.long)
    loss = synapx.nn.NLLLoss()(log_soft, ylabel)
    loss.backward()

    # torch
    ypred_t = torch.tensor(pred, requires_grad=True)
    log_soft_t = torch.log_softmax(ypred_t, dim=1)
    ylabel_t = torch.tensor(label, dtype=torch.long)
    loss_t = torch.nn.NLLLoss()(log_soft_t, ylabel_t)
    loss_t.backward()

    print(ypred_t.grad)
    print(ypred.grad)
    
    assert check_tensors(loss, loss_t)
    assert check_tensors(ypred.grad, ypred_t.grad)

    
def check_loss(loss_fn, loss_fn_t, pred, label, label_dtype=torch.float32):
    ypred = synapx.tensor(pred, requires_grad=True)
    ylabel = synapx.tensor(label, dtype=label_dtype)
    loss = loss_fn(ypred, ylabel)
    loss.sum().backward()  # sum() so that reduction='none' works too

    ypred_t = torch.tensor(pred, requires_grad=True)
    ylabel_t = torch.tensor(label, dtype=label_dtype)
    loss_t = loss_fn_t(ypred_t, ylabel_t)
    loss_t.sum().backward()

    assert check_tensors(loss, loss_t)
    assert check_tensors(ypred.grad, ypred_t.grad)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_BCELoss(reduction):
    # Includes probabilities of exactly 0 and 1, where the log is clamped
    pred = [0.3, 0.4, 0.999, 1.0, 0.0, 1e-8]
    label = [0, 1, 0, 1, 1, 0]
    check_loss(synapx.nn.BCELoss(reduction=reduction), torch.nn.BCELoss(reduction=reduction),
               pred, label)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_BCEWithLogitsLoss(reduction):
    pred = [0.3, 0.4, 0.7, -2.0, -1.0, 40.0, -40.0]
    label = [0, 1, 0, 1, 1, 0, 1]
    check_loss(synapx.nn.BCEWithLogitsLoss(reduction=reduction),
               torch.nn.BCEWithLogitsLoss(reduction=reduction), pred, label)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_CrossEntropyLoss(reduction):
    pred = [[0.14, 0.3, -1.0], [-0.2, 0.9, 2.0], [-3.0, 0.1, 0.5], [1.0, 1.0, 1.0]]
    label = [0, 1, 2, 0]
    check_loss(synapx.nn.CrossEntropyLoss(reduction=reduction),
               torch.nn.CrossEntropyLoss(reduction=reduction), pred, label, torch.long)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_NLLLoss_reductions(reduction):
    pred = [[-0.5, -1.2, -2.0], [-0.1, -3.0, -1.5], [-2.2, -0.3, -0.9]]
    label = [2, 0, 1]
    check_loss(synapx.nn.NLLLoss(reduction=reduction),
               torch.nn.NLLLoss(reduction=reduction), pred, label, torch.long)
