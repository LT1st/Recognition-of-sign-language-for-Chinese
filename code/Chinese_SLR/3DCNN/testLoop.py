import torch
from sklearn.metrics import accuracy_score


def test_epoch(model, dataloader, device, writer):
    model.eval()
    dataNum = len(dataloader)
    score = 0
    # all_label = []
    # all_pred = []

    for batch_idx, data in enumerate(dataloader):
        # get the inputs and labels
        inputs, labels = data['data'].to(device), data['label'].to(device)
        labels = labels.squeeze()
        # forward
        outputs = model(inputs)
        if isinstance(outputs, list):
            outputs = outputs[0]

        prediction = torch.max(outputs, dim=1).indices
        # print("prediction :", prediction)
        # print("label :", labels)

        score += accuracy_score(labels.cpu().data.squeeze(
        ).numpy(), prediction.cpu().data.squeeze().numpy())

    print(f"Test num: {5000} Test acc: {score*100/dataNum}%")
