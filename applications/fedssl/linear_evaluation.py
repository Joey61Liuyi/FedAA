import argparse
from collections import defaultdict

import numpy
import numpy as np
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt
from easyfl.datasets.data import CIFAR100
from eval_dataset import get_data_loaders
from model import get_encoder_network
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def tSNE(h, label, name):
    color_dict = {'blue': 0, 'green': 1, 'yellow': 2, 'red': 3, 'black': 4, 'gray': 5, 'gold': 6, 'pink': 7,
                  'purple': 8, 'orange': 9}
    data = np.array(h)
    target = np.array(label)
    tsne = TSNE(n_components=2, n_iter=500)
    data_tsne = tsne.fit_transform(data)
    x, y = data_tsne[:, 0], data_tsne[:, 1]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    # current_axes = plt.axes()
    # current_axes.xaxis.set_visible(False)
    # current_axes.yaxis.set_visible(False)

    color_target = [list(color_dict.keys())[list(color_dict.values()).index(c)] for c in target]
    plt.scatter(x, y, c=color_target, s=10)
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14, rotation=0)
    fig.tight_layout()
    plt.savefig(name, dpi=1000)
    plt.show()

def inference(loader, model, device):
    feature_vector = []
    labels_vector = []

    if isinstance(model, list):
        models = model
        for resnet in models:
            resnet.eval()

        for step, (x, y) in enumerate(loader):
            x = x.cuda()
            feature_list = []
            for resnet in models:
                with torch.no_grad():
                    h = resnet(x)
                    h = h.squeeze()
                    h = h.detach()
                feature_list.append(h)
            h = sum(feature_list)/len(feature_list)
            feature_vector.extend(h.cpu().detach().numpy())
            labels_vector.extend(y.numpy())
            if step % 5 == 0:
                print(f"Step [{step}/{len(loader)}]\t Computing features...")

    else:
        model.eval()
        for step, (x, y) in enumerate(loader):
            x = x.to(device)

            # get encoding


            with torch.no_grad():
                h = model(x)

            h = h.squeeze()
            h = h.detach()

            # tep = torch.randn(x.shape).to(device)
            # tep1 = torch.randn(h.shape)
            #
            # h_tep = model(tep)
            # h_tep = h_tep.squeeze()
            # h_tep = h_tep.detach()
            # h_tep = torch.matmul(h_tep.t(), h_tep)
            # h_tep = h_tep.cpu().numpy()
            #
            # # sns.heatmap(h_tep, linewidth = 0.5)
            # h_tep = numpy.clip(h_tep, 0, 1000)
            # im = plt.imshow(h_tep)
            # # fig, ax = plt.subplots()
            # # heatmap = ax.pcolor(h_tep, cmap=plt.cm.Blues)
            # plt.colorbar(im)
            # plt.savefig('random_input.png')
            # plt.show()
            # plt.close()
            #
            # h = torch.matmul(h.t(), h)
            # h = h.cpu().numpy()
            # h_tep = numpy.clip(h, 0, 500)
            # im = plt.imshow(h)
            # # sns.heatmap(h.cpu().numpy(), linewidths=0.5)
            # # fig, ax = plt.subplots()
            # # heatmap = ax.pcolor(h.cpu().numpy(), cmap=plt.cm.Blues)
            # plt.colorbar(im)
            # plt.savefig('feature.png')
            # plt.show()
            # plt.close()
            #
            # tep1 = torch.matmul(tep1.t(), tep1)
            #
            #
            # # fig, ax = plt.subplots()
            # # heatmap = ax.pcolor(tep1.numpy(), cmap=plt.cm.Blues)
            #
            #
            # tep1 = tep1.cpu().numpy()
            # tep1 = numpy.clip(tep1, 0, 500)
            # im = plt.imshow(tep1)
            #
            # plt.colorbar(im)
            # plt.savefig('random.png')
            # plt.show()
            # plt.close()
            #
            #
            # # tep = torch.randn
            # # plt.imshow(tep.cpu().numpy())
            # print('pause')
            feature_vector.extend(h.cpu().detach().numpy())
            labels_vector.extend(y.numpy())

            if step % 5 == 0:
                print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, model, device)
    test_X, test_y = inference(test_loader, model, device)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def test_result(test_loader, logreg, device, model_path):
    # Test fine-tuned model
    print("### Calculating final testing performance ###")
    logreg.eval()
    metrics = defaultdict(list)
    for step, (h, y) in enumerate(test_loader):
        h = h.to(device)
        y = y.to(device)

        outputs = logreg(h)

        # calculate accuracy and save metrics
        accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
        metrics["Accuracy/test"].append(accuracy)

    print(f"Final test performance: " + model_path)
    for k, v in metrics.items():
        print(f"{k}: {np.array(v).mean():.4f}")
    return np.array(metrics["Accuracy/test"]).mean()


if __name__ == "__main__":
    set_random_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--model_path", default='./saved_models/100_cifar10_fedu_weights_agg__0.01/100_cifar10_fedu_weights_agg__0.01_global_model_r_89.pth', type=str, help="Path to pre-trained model (e.g. model-10.pt)")
    parser.add_argument('--model', default='byol', type=str, help='name of the network')
    parser.add_argument("--image_size", default=32, type=int, help="Image size")
    parser.add_argument("--learning_rate", default=3e-3, type=float, help="Initial learning rate.")
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size for training.")
    parser.add_argument("--num_epochs", default=200, type=int, help="Number of epochs to train for.")
    parser.add_argument("--encoder_network", default="resnet18", type=str, help="Encoder network architecture.")
    parser.add_argument("--num_workers", default=8, type=int, help="Number of data workers (caution with nodes!)")
    parser.add_argument("--personalized", default=False, type=bool, help="Number of data workers (caution with nodes!)")
    parser.add_argument("--fc", default="identity", help="options: identity, remove")
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # get data loaders


    model_name = ['f0000000.pth','f0000001.pth', 'f0000002.pth', 'f0000003.pth', 'f0000004.pth']

    model_dict ={
        './saved_models/cifar100_byol_local_QR_semantic_0.01/cifar100_byol_local_QR_semantic_0.01_global_model_r_99_f0000000.pth': 'resnet18',
        './saved_models/cifar100_byol_local_QR_semantic_0.01/cifar100_byol_local_QR_semantic_0.01_global_model_r_99_f0000001.pth': 'vgg9',
        './saved_models/cifar100_byol_local_QR_semantic_0.01/cifar100_byol_local_QR_semantic_0.01_global_model_r_99_f0000002.pth': 'alexnet',
        './saved_models/cifar100_byol_local_QR_semantic_0.01/cifar100_byol_local_QR_semantic_0.01_global_model_r_99_f0000003.pth': 'resnet34',
    }

    models = []
    if args.personalized:
        # name_list = args.model_path.split('_')
        # for name in model_name:
        #     name_list.pop()
        #     name_list.append(name)
        #     model = '_'.join(name_list)
        #     resnet = get_encoder_network(args.model, args.encoder_network)
        #     resnet.load_state_dict(torch.load(model, map_location = device))
        #     num_features = list(resnet.children())[-1].in_features
        #     if args.fc == "remove":
        #         resnet = nn.Sequential(*list(resnet.children())[:-1])  # throw away fc layer
        #     else:
        #         resnet.fc = nn.Identity()
        #     resnet = resnet.to(device)
        #     models.append(resnet)
        # resnet = models
        acc_list = []
        for model in model_dict:
            train_loader, test_loader = get_data_loaders(args.dataset, args.image_size, args.batch_size,
                                                         args.num_workers)
            online_encoder = get_encoder_network(args.model, model_dict[model])
            online_encoder.load_state_dict(torch.load(model, map_location=device))
            online_encoder = online_encoder.to(device)
            num_features = online_encoder.feature_dim
            online_encoder.fc = nn.Identity()
            n_classes = 10
            if args.dataset == CIFAR100:
                n_classes = 100

            logreg = nn.Sequential(nn.Linear(num_features, n_classes))
            logreg = logreg.to(device)

            # loss / optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(params=logreg.parameters(), lr=args.learning_rate)

            print("Creating features from pre-trained model")
            (train_X, train_y, test_X, test_y) = get_features(
                online_encoder, train_loader, test_loader, device
            )

            train_loader, test_loader = create_data_loaders_from_arrays(
                train_X, train_y, test_X, test_y, 2048
            )
            logreg.train()
            for epoch in range(args.num_epochs):
                metrics = defaultdict(list)
                for step, (h, y) in enumerate(train_loader):
                    h = h.to(device)
                    y = y.to(device)

                    outputs = logreg(h)

                    loss = criterion(outputs, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # calculate accuracy and save metrics
                    accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
                    metrics["Loss/train"].append(loss.item())
                    metrics["Accuracy/train"].append(accuracy)

                print(f"Epoch [{epoch}/{args.num_epochs}]: " + "\t".join(
                    [f"{k}: {np.array(v).mean()}" for k, v in metrics.items()]))

                if epoch % 100 == 0:
                    print("======epoch {}======".format(epoch))
                    test_result(test_loader, logreg, device, args.model_path)

            acc = test_result(test_loader, logreg, device, args.model_path)
            acc_list.append(acc)
        print('===========')
        print(acc_list)
        print(sum(acc_list)/len(acc_list))
        print('===========')

    else:
        train_loader, test_loader = get_data_loaders(args.dataset, args.image_size, args.batch_size, args.num_workers)
        resnet = get_encoder_network(args.model, args.encoder_network)
        resnet.load_state_dict(torch.load(args.model_path, map_location=device))
        resnet = resnet.to(device)
        num_features = list(resnet.children())[-1].in_features
        if args.fc == "remove":
            resnet = nn.Sequential(*list(resnet.children())[:-1])  # throw away fc layer
        else:
            resnet.fc = nn.Identity()

        n_classes = 10
        if args.dataset == CIFAR100:
            n_classes = 100

        # fine-tune model
        logreg = nn.Sequential(nn.Linear(num_features, n_classes))
        logreg = logreg.to(device)

        # loss / optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=logreg.parameters(), lr=args.learning_rate)

        # compute features (only needs to be done once, since it does not backprop during fine-tuning)
        print("Creating features from pre-trained model")
        (train_X, train_y, test_X, test_y) = get_features(
            resnet, train_loader, test_loader, device
        )

        train_loader, test_loader = create_data_loaders_from_arrays(
            train_X, train_y, test_X, test_y, 2048
        )

        # Train fine-tuned model
        logreg.train()
        for epoch in range(args.num_epochs):
            metrics = defaultdict(list)
            for step, (h, y) in enumerate(train_loader):
                # tSNE(h, y, 'poor.pdf')
                h = h.to(device)
                y = y.to(device)

                outputs = logreg(h)

                loss = criterion(outputs, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # calculate accuracy and save metrics
                accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
                metrics["Loss/train"].append(loss.item())
                metrics["Accuracy/train"].append(accuracy)

            print(f"Epoch [{epoch}/{args.num_epochs}]: " + "\t".join(
                [f"{k}: {np.array(v).mean()}" for k, v in metrics.items()]))

            if epoch % 100 == 0:
                print("======epoch {}======".format(epoch))
                test_result(test_loader, logreg, device, args.model_path)
        test_result(test_loader, logreg, device, args.model_path)
