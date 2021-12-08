import argparse
import time
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
import timm
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import io
import tensorflow as tf
import json
from urllib.parse import urlparse
import boto3
import os
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Train SSD networks.')
    parser.add_argument('--file',
                        type=str,
                        default='s3://w210-poverty-mapper/modeling/model_run_files/binary_within_1.json',
                        help='File with experiments to run.')
    parser.add_argument("-p", "--prcurve", action="store_true")
    args = parser.parse_args()
    return args

def trntransforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def tsttransforms():
    return A.Compose([
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

class PovertyDataset(Dataset):
    def __init__(self, df, mode, transform=None):
        self.data = df
        self.img_dir = f'./'
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        fname = self.data.iloc[idx]['filename'] + ".png"
        img_path = f'{self.img_dir}/data/{fname}'
        image = cv2.imread(img_path)
        if self.transform is not None:
            image = self.transform(image = image)['image']
        image = image.float() / 255.
        label = self.data.iloc[idx]['label']

        return image, label

class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, n_classes):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2048, 1024),
      nn.ReLU(),
      nn.Linear(1024, 512),
      nn.ReLU(),
      nn.Linear(512, n_classes)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4).numpy()
    # Add the batch dimension
    #image = tf.expand_dims(image, 0).numpy()
    return image


def image_grid(images, labels, predictions, batch_size):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10, 10))
    for i in range(min(batch_size, 25)):
    # Start next subplot.
        plt.subplot(5, 5, i + 1, title="lbl:" + str(labels[i].item()) + " prd:" + str(predictions[i]))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        imgviz = (images[i] * 255).transpose(0, 2).numpy().astype(np.uint8)
        img = Image.fromarray(imgviz)
        plt.imshow(img)
    return figure


def train_model(model, trndf, optimizer, criterion, epoch, trnloader, num_classes):
    model.train()
    running_loss = 0.0
    train_loss = 0.0
    loader_total = int(len(trnloader))
    y_pred = [] # save predction
    tk0 = tqdm(trnloader, total=loader_total)
    for step, batch in enumerate(tk0):
        inputs = batch[0].cuda().float()
        labels = batch[1].cuda().long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        y_pred.append(outputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_loss = (running_loss / (step + 1))
        tk0.set_postfix(train_loss=train_loss)
        # if step % (loader_total/5) == 1:
        #     predictions = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        #     figure = image_grid(batch[0], batch[1], predictions, len(batch[1]))
        #     plot_file_name = 'plots/train_' +  str(((epoch * loader_total) + step)) + ".png"
        #     figure.savefig(plot_file_name)
        #     plt.close('all')
    preds = torch.cat(y_pred).argmax(1).detach().cpu().numpy()
    train_acc = (trndf.label.values == preds).mean()
    cf_matrix = np.array(confusion_matrix(preds, trndf.label.values, labels=range(num_classes))).tolist()
    return train_acc, train_loss, cf_matrix

def validate_model(model, criterion, valdf, epoch, valloader, num_classes):
    valpreds = []
    model.eval()
    running_loss = 0.0
    valid_loss = 0.0
    loader_total = int(len(valloader))
    tkval = tqdm(valloader, total=loader_total)
    for step, batch in enumerate(tkval):
        inputs = batch[0].cuda().float()
        labels = batch[1].cuda().long()
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        valpreds.append(outputs)
        running_loss += loss.item()
        valid_loss = (running_loss / (step + 1))
        tkval.set_postfix(valid_loss=valid_loss)
        # if step % (loader_total/5) == 1:
        #     predictions = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        #     figure = image_grid(batch[0], batch[1], predictions, len(batch[1]))
        #     plot_file_name = 'plots/val_' +  str(((epoch * loader_total) + step)) + ".png"
        #     figure.savefig(plot_file_name)
        #     plt.close('all')
    preds = torch.cat(valpreds).argmax(1).detach().cpu().numpy()
    val_acc = (valdf.label.values == preds).mean()
    cf_matrix = np.array(confusion_matrix(preds, valdf.label.values, labels=range(num_classes))).tolist()
    return val_acc, valid_loss, cf_matrix


def test_model(model, tstdf, tstloader, num_classes):
    tstpreds = []
    class_probs = []
    test_total = int(len(tstloader))
    tktst = tqdm(tstloader, total=test_total)
    for step, batch in enumerate(tktst):
        inputs = batch[0].cuda().float()
        labels = batch[1].cuda().long()
        with torch.no_grad():
            outputs = model(inputs)
            tstpreds.append(outputs)
            class_probs_batch = [F.softmax(el, dim=0) for el in outputs]
            class_probs.append(class_probs_batch)
        # if step % (test_total/5) == 1:
        #     predictions = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        #     figure = image_grid(batch[0], batch[1], predictions, len(batch[1]))
        #     plot_file_name = 'plots/test_' +  str(((epoch * test_total) + step)) + ".png"
        #     figure.savefig(plot_file_name)
        #     plt.close('all')
    predicted_labels = torch.cat(tstpreds).argmax(1).detach().cpu().numpy()
    test_acc = (tstdf.label.values == predicted_labels).mean()
    print(f'Test accuracy {test_acc:.4f}')
    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_label = tstdf.label.values
    cf_matrix = np.array(confusion_matrix(predicted_labels,tstdf.label.values, labels=range(num_classes))).tolist()
    return test_acc, test_probs, test_label, cf_matrix

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model(architecture, freeze_layers, num_classes):
    device = torch.device("cuda:0")
    model = timm.create_model(architecture, pretrained = True)
    if freeze_layers == 'yes':
        set_parameter_requires_grad(model, True)
    model.fc = MLP(num_classes)
    print("Model fc layer = " + str(model.fc))
    model = model.to(device)
    return model

def get_s3json(s3, url):
    url_parts = urlparse(url, allow_fragments=False)
    response = s3.get_object(Bucket=url_parts.netloc, Key=url_parts.path.strip("/"))
    content = response['Body']
    json_content = json.loads(content.read())
    return json_content

def save_json_to_s3(s3, results, results_path):
    url_parts = urlparse(results_path, allow_fragments=False)
    s3.put_object(Bucket=url_parts.netloc,
                  Key=url_parts.path.strip("/"),
                  Body=(bytes(json.dumps(results).encode('UTF-8'))))

def save_file_to_s3(s3, file, path):
    url_parts = urlparse(path, allow_fragments=False)
    s3.put_object(Bucket=url_parts.netloc,
                  Key=url_parts.path.strip("/"),
                  Body=open(file, 'rb'))

def get_file_name_from_path(path):
    url_parts = urlparse(path, allow_fragments=False)
    return os.path.basename(url_parts.path)

if __name__ == '__main__':
    args = parse_args()
    s3 = boto3.client('s3')
    filename = args.file
    print(f"Poverty Mapper Trainer v2: Processing file {filename}")
    run_spec_file = get_s3json(s3, filename)
    model_specs = run_spec_file['model_specs']

    for model_spec in model_specs:
        results = {}
        results['model_spec'] = model_spec
        run_configuration = get_s3json(s3, model_spec)
        results['model_spec_content'] = run_configuration
        experiment_name = run_configuration['split_name']
        results_path = run_configuration['results_path']
        num_classes = run_configuration['num_classes']
        output_dir = run_configuration['model_artifacts_path']
        writer = SummaryWriter('runs/' + experiment_name)
        print(f"### Executing model spec: {model_spec}")
        trndf = pd.read_csv(run_configuration['train'])
        valdf = pd.read_csv(run_configuration['val'])
        tstdf = pd.read_csv(run_configuration['test'])
        print(f'File shapes -- train : {trndf.shape}, valid : {valdf.shape}, test : {tstdf.shape}')
        trndataset = PovertyDataset(trndf, 'train', trntransforms())
        valdataset = PovertyDataset(valdf, 'valid', tsttransforms())
        tstdataset = PovertyDataset(tstdf, 'test', tsttransforms())
        loaderargs = {'num_workers': run_configuration['num_workers'], 'batch_size': run_configuration['batch_size'],
                      'pin_memory': False,
                      'drop_last': False}
        trnloader = DataLoader(trndataset, shuffle=True, **loaderargs)
        valloader = DataLoader(valdataset, shuffle=False, **loaderargs)
        tstloader = DataLoader(tstdataset, shuffle=False, **loaderargs)
        model = get_model(run_configuration['pretrained'], run_configuration['freeze_layers'], num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=run_configuration['learning_rate'])
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = lr_scheduler.StepLR(optimizer, step_size=run_configuration['step_size'], gamma=run_configuration['gamma'])
        num_epochs = run_configuration['epochs']
        start_time = time.time()
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        train_cfs = []
        val_cfs = []

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            train_acc, train_loss, train_cf = train_model(model, trndf, optimizer, criterion, epoch, trnloader,num_classes)
            writer.add_scalar("Loss/train", train_loss, epoch+1)
            writer.add_scalar("Acc/train", train_acc, epoch + 1)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            train_cfs.append(train_cf)
            last_lr = scheduler.get_last_lr()
            scheduler.step()
            val_acc, val_loss, val_cf = validate_model(model, criterion, valdf, epoch, valloader,num_classes)
            writer.add_scalar("Loss/val", val_loss, epoch + 1)
            writer.add_scalar("Acc/val", val_acc, epoch + 1)
            writer.add_hparams({'lr': last_lr[0], 'batchsize': run_configuration['batch_size']}, {'hparam/valaccuracy': val_acc, 'hparam/valloss': val_loss, 'hparam/trainloss': train_loss})
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_cfs.append(val_cf)
            print(f'Valid accuracy {val_acc:.4f} Valid Loss {val_loss:.4f} Last LR: {last_lr}')
        test_acc, test_probs, test_label, test_cf = test_model(model, tstdf, tstloader,num_classes)
        writer.add_scalar("Acc/test", test_acc, 0)
        results['train_losses'] = train_losses
        results['train_accs'] = train_accs
        results['train_cfs'] = train_cfs
        results['val_losses'] = val_losses
        results['val_accs'] = val_accs
        results['val_cfs'] = val_cfs
        results['test_acc'] = test_acc
        results['test_cf'] = test_cf
        save_json_to_s3(s3, results, results_path)

        base_experiment_file = get_file_name_from_path(model_spec)
        model_file_name = base_experiment_file.replace(".json", ".pth")
        base_experiment_file = base_experiment_file.replace(".json", "")
        print("Saving model file", model_file_name)
        torch.save(model.state_dict(), model_file_name)
        save_file_to_s3(s3, model_file_name, output_dir + model_file_name)

        # for plot_filename in glob.glob('./plots/*.png'):
        #     base_file_name = os.path.basename(plot_filename)
        #     save_file_to_s3(s3, plot_filename, output_dir + base_experiment_file + base_file_name)
        #     os.remove(plot_filename)
    print("### Program done executing all experiments!")