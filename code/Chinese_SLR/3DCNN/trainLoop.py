import torch
from sklearn.metrics import accuracy_score


def train_epoch(model, loss_fn, optimizer, dataloader, device, epoch, log_interval, writer):
    model.train()
    losses = []
    score = 0
    # all_label = []
    # all_pred = []

    for batch_idx, data in enumerate(dataloader):
        # get the inputs and labels
        inputs, labels = data['data'].to(device), data['label'].to(device)

        optimizer.zero_grad()
        # forward
        outputs = model(inputs)
        if isinstance(outputs, list):
            outputs = outputs[0]
        # compute the loss
        loss = loss_fn(outputs, labels.squeeze())
        losses.append(loss.item())

        # compute the accuracy
        prediction = torch.max(outputs, 1).indices
        # all_label.extend(labels.squeeze())
        # all_pred.extend(prediction)

        score += accuracy_score(labels.squeeze().cpu().data.squeeze(
        ).numpy(), prediction.cpu().data.squeeze().numpy())
        # backward & optimize
        loss.backward()
        optimizer.step()
        # break

        if (batch_idx + 1) % log_interval == 0:
            print("epoch {:3d} | iteration {:5d} | Loss {:.6f} | Acc {:.2f}%".format(
                epoch, batch_idx+1, loss.item(), (score/log_interval)*100))
            score = 0

    # Compute the average loss & accuracy
    # training_loss = sum(losses)/len(losses)
    # all_label = torch.stack(all_label, dim=0)
    # all_pred = torch.stack(all_pred, dim=0)
    # training_acc = accuracy_score(all_label.squeeze().cpu(
    # ).data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
