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

        # forward
        print(inputs.shape)
        outputs = model(inputs)
        if isinstance(outputs, list):
            outputs = outputs[0]

        prediction = torch.max(outputs, 1).indices

        score += accuracy_score(labels.squeeze().cpu().data.squeeze(
        ).numpy(), prediction.cpu().data.squeeze().numpy())

    print(f"Test num: {dataNum} Test acc: {score/dataNum}%")
