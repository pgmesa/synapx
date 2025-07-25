
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

    
# def test_BCELoss():
#     pred = [0.3, 0.4, 0.999, 1, 0]
#     label = [0, 1, 0, 1, 1]
    
#     # synapx
#     ypred = Tensor(pred, requires_grad=True)
#     ylabel = Tensor(label)
#     loss = nn.BCELoss()(ypred, ylabel)
#     loss.backward()

#     # torch
#     ypred_t = torch.tensor(pred, requires_grad=True)
#     ylabel_t = torch.tensor(label, dtype=torch.float)
#     loss_t = torch.nn.BCELoss()(ypred_t, ylabel_t)
#     loss_t.backward()

#     print(loss, "\n", loss_t)
#     print(ypred_t.grad)
#     print(ypred.grad)
    
#     assert check_tensors(loss, loss_t)
#     assert check_tensors(ypred.grad, ypred_t.grad)


# def test_BCEWithLogitsLoss():
#     pred = [0.3, 0.4, 0.7, -2, -1]
#     label = [0, 1, 0, 1, 1]
    
#     # synapx
#     ypred = Tensor(pred, requires_grad=True)
#     ylabel = Tensor(label)
#     loss = nn.BCEWithLogitsLoss()(ypred, ylabel)
#     loss.backward()

#     # torch
#     ypred_t = torch.tensor(pred, requires_grad=True)
#     ylabel_t = torch.tensor(label, dtype=torch.float)
#     loss_t = torch.nn.BCEWithLogitsLoss()(ypred_t, ylabel_t)
#     loss_t.backward()
    
#     print(loss, "\n", loss_t)
#     print(ypred_t.grad)
#     print(ypred.grad)
    
#     assert check_tensors(loss, loss_t)
#     assert check_tensors(ypred.grad, ypred_t.grad)
    
    
# def test_CrossEntropyLoss():
#     pred = [[0.14, 0.3],[-0.2, 0.9],[-3, 0.1]]
#     label = [0, 1, 0]
    
#     # synapx
#     ypred = Tensor(pred, requires_grad=True)
#     ylabel = Tensor(label, dtype=np.int8)
#     loss = nn.CrossEntropyLoss()(ypred, ylabel)
#     loss.backward()

#     # torch
#     ypred_t = torch.tensor(pred, requires_grad=True)
#     ylabel_t = torch.tensor(label).type(torch.LongTensor)
#     loss_t = torch.nn.CrossEntropyLoss()(ypred_t, ylabel_t)
#     loss_t.backward()

#     print(ypred_t.grad)
#     print(ypred.grad)
    
#     assert check_tensors(loss, loss_t)
#     assert check_tensors(ypred.grad, ypred_t.grad)