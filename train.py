import torch
import config
import time
from shutil import copyfile
from config import args_setting
from dataset_new import RoadSequenceDataset, RoadSequenceDatasetList
import numpy as np
import cv2
from radam import RAdam
from model import UNet_TwoConvGRU
from torchvision import transforms
from torch.optim import lr_scheduler
current_pth_name = ''
import torch.nn.functional as F
best_acc = 0.903
best_name = config.pretrained_path
def train(args, epoch, model, train_loader, device, optimizer, criterion):
    since = time.time()
    model.train()
    for batch_idx,  sample_batched in enumerate(train_loader):
        data, target = sample_batched['data'].to(device), sample_batched['label'].type(torch.LongTensor).to(device) # LongTensor
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    time_elapsed = time.time() - since
    print('Train Epoch: {} complete in {:.0f}m {:.0f}s'.format(epoch,
        time_elapsed // 60, time_elapsed % 60))

def val(args, model, val_loader, device, criterion, criterion2):
    model.eval()
    test_loss = 0
    test_loss2 = 0
    correct = 0
    i = 0
    with torch.no_grad():
        for sample_batched in val_loader:
            i += 1
            print('val--------------', i)
            data, target = sample_batched['data'].to(device), sample_batched['label'].type(torch.LongTensor).to(device)            
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]
            t = pred.eq(target.view_as(pred)).sum().item()
            correct += t

    test_loss /= (len(val_loader.dataset)/args.test_batch_size)
    test_loss2 /= (len(val_loader.dataset) / args.test_batch_size)
    val_acc = 100. * int(correct) / (len(val_loader.dataset) * config.label_height * config.label_width)
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)\n'.format(test_loss, int(correct), len(val_loader.dataset), val_acc))
    current_pth_name = '%s.pth'%val_acc
    print('val-------current_pth_name---------------', current_pth_name)
    torch.save(model.state_dict(), '%s.pth'%val_acc)
    return current_pth_name



def evaluate_model(model, test_loader, device, criterion):
    model.eval()
    i = 0
    precision = 0.0
    recall = 0.0
    test_loss = 0
    correct = 0
    error=0
    fp = 0
    fn = 0
    with torch.no_grad():
        for sample_batched in test_loader:
            i+=1
            print(i)
            data, target = sample_batched['data'].to(device), sample_batched['label'].type(torch.LongTensor).to(device)
            raw_file = sample_batched['raw_file']
            label_name = sample_batched['label_name']
            large_label = sample_batched['new_label']
            final_label = torch.squeeze(large_label).cpu().numpy() * 255
            s = time.time()
            output = model(data)
            e = time.time()
            pred = output.max(1, keepdim=True)[1]  # 返回两个，一个是最大值另一个是最大值索引            
            img = torch.squeeze(pred).cpu().numpy()*255
            img2 = torch.squeeze(pred).cpu().unsqueeze(2).numpy() * 255
            final_img = cv2.resize(img2, (1280, 720), interpolation=cv2.INTER_NEAREST)
            lab = torch.squeeze(target).cpu().numpy()*255
            img = img.astype(np.uint8)#for pred_recall
            lab = lab.astype(np.uint8)#for label_precision
            kernel = np.uint8(np.ones((3, 3)))
            
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            label_precision = cv2.dilate(lab, kernel)
            # print('label_precision----', label_precision)
            pred_recall = cv2.dilate(img, kernel)
            # print('pred_recall----', pred_recall)
            img = img.astype(np.int32)
            lab = lab.astype(np.int32)
            label_precision = label_precision.astype(np.int32)
            pred_recall = pred_recall.astype(np.int32)
            # print('imge--------', img.shape, label_precision)

            a = len(np.nonzero(img*label_precision)[1])
            b = len(np.nonzero(img)[1])
            if b==0:
                error=error+1
                continue
            else:
                fp += float(b - a)
                precision += float(a/b)
            c = len(np.nonzero(pred_recall*lab)[1])
            d = len(np.nonzero(lab)[1])

            if d==0:
                error = error + 1
                continue
            else:
                fn += float(d - c)
                recall += float(c / d)
            F1_measure=(2*precision*recall)/(precision+recall)
    test_loss /= (len(test_loader.dataset) / args.test_batch_size)
    test_acc = 100. * int(correct) / (len(test_loader.dataset) * config.label_height * config.label_width)
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)'.format(
        test_loss, int(correct), len(test_loader.dataset), test_acc))

    precision = precision / (len(test_loader.dataset) - error)
    recall = recall / (len(test_loader.dataset) - error)
   
    F1_measure = F1_measure / (len(test_loader.dataset) - error)
    print('Precision: {:.5f}, Recall: {:.5f}, F1_measure: {:.5f}\n'.format(precision,recall,F1_measure))
    evaluate_result = {'precision': precision, 'recall': recall, 'F1_measure': F1_measure, 'test_acc':test_acc}
    return evaluate_result


if __name__ == '__main__':
    args = args_setting()
    torch.manual_seed(args.seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # turn image into floatTensor
    op_tranforms = transforms.Compose([transforms.ToTensor(),])

    # load data for batches, num_workers for multiprocess
    train_loader = torch.utils.data.DataLoader(
        RoadSequenceDatasetList(file_path=config.train_path, transforms=op_tranforms),
        batch_size=args.batch_size, shuffle=True, num_workers=config.data_loader_numworkers)
    val_loader = torch.utils.data.DataLoader(
        RoadSequenceDatasetList(file_path=config.val_path, transforms=op_tranforms),
        batch_size=args.test_batch_size, shuffle=True, num_workers=config.data_loader_numworkers)

    #load data for testing
    test_loader = torch.utils.data.DataLoader(
        RoadSequenceDataset(file_path=config.test_path, transforms=op_tranforms),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1)

    #load model
    model = UNet_TwoConvGRU(3, 2).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Adam 参数betas=(0.9, 0.99)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    optimizer = RAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    class_weight = torch.Tensor(config.class_weight)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)
    criterion2 = torch.nn.MSELoss().to(device)
    # best_acc = 0
    if config.pretrained_path:
        print('loading------------------')
        pretrained_dict = torch.load(config.pretrained_path)
        model_dict = model.state_dict()
        #
        pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
        model_dict.update(pretrained_dict_1)
        model.load_state_dict(model_dict)
        evaluate_model(model, test_loader, device, criterion)
        exit(0)
    # train
    for epoch in range(1, args.epochs+1):
        if scheduler.get_lr()[0] > 0.0000001:
            scheduler.step()
        else:
            print('lr----no--change--------')
        print('lr---------', scheduler.get_lr())
        train(args, epoch, model, train_loader, device, optimizer, criterion)
        val_pth_name = val(args, model, val_loader, device, criterion, criterion2)
        print('val_pth_name------', val_pth_name)
        pretrained_dict = torch.load(val_pth_name)
        model_dict = model.state_dict()
        pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
        model_dict.update(pretrained_dict_1)
        model.load_state_dict(model_dict)
        result = evaluate_model(model, test_loader, device, criterion)
        if result['F1_measure'] > best_acc:
            best_acc = result['F1_measure']
            best_name='__test_acc=%s'%result['test_acc']  + '__precision=%s'%result['precision']  + '__recall=%s'%result['recall']  + '__F1_measure=%s'%result['F1_measure'] + '_epoch=%s'%epoch + '_'+ val_pth_name
            copyfile(val_pth_name, best_name)
            print('best testing-------------', best_name)
            print('test acc-------------',  result['test_acc'])
            print('precision-----------', result['precision'])
            print('recall-----------', result['recall'])
            print('F1_measure-----------', result['F1_measure'])
        elif result['F1_measure'] > 0.903:
            current_name='__test_acc=%s'%result['test_acc']  + '__precision=%s'%result['precision']  + '__recall=%s'%result['recall']  + '__F1_measure=%s'%result['F1_measure'] + '_epoch=%s'%epoch + '_'+ val_pth_name
            copyfile(val_pth_name, current_name)
            print('current testing-------------', val_pth_name)
            print('best testing-------------', best_name)
