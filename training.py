from __future__ import print_function
import os
import ArgsHandler
import torch
import torch.optim as optim
import time
from torch.optim.lr_scheduler import StepLR
from BeamDataset import DatasetHandler, calculate_mmse
from Cosine_sim_loss import make_complex, MSE_normalize, norm_amp_loss, spread_complex
import gc

from ModelHandler import model_selector

import csv
import copy
import numpy as np
from CachefileHandler import save_cache, load_cache


def error_logger(batch_idx, data, h_ls, target, output, log_name, step_type="train"):
    os.makedirs("error", exist_ok=True)
    with open(os.path.join("error", log_name), "a") as f:
        for i in range(len(data)):
            f.write(f"{step_type} : ")
            f.write(f"{batch_idx},{data[i].tolist()},{h_ls[i].tolist()},{target[i].tolist()},{output[i].tolist()}\n")


def train(args, model, device, train_loader, optimizer, epoch, x_norm, y_norm, do_print=False):
    model.train()
    l = torch.nn.MSELoss(reduction='none')

    batch_len = int(len(train_loader))

    batch_multiply_count = args.batch_multiplier
    optimizer.zero_grad(set_to_none=True)

    continuous_error_counter = 0

    for batch_idx, (data, heur, target) in enumerate(train_loader):
        data, target, heur = data.to(device, non_blocking=True), target.to(device, non_blocking=True), heur.to(device, non_blocking=True)

        if batch_multiply_count == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            batch_multiply_count = args.batch_multiplier

        with torch.no_grad():
            data *= x_norm
            target *= y_norm
            heur *= y_norm

        try:
            output = model(data, heur)
            loss = l(output, target)

            loss = MSE_normalize(loss, target) / args.batch_multiplier
            
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                continuous_error_counter += 1
                error_logger(batch_idx, data, heur, target, output, args.log + ".log")

                if continuous_error_counter > 10:
                    if args.save_model:
                        opt_model_para = copy.deepcopy(model.state_dict())
                        torch.save(opt_model_para, "cache/"+args.log+'_error.pt')
                    exit(1)
            else:
                continuous_error_counter = 0
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if batch_idx % args.log_interval == 0 and do_print:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), batch_len * len(data),
                    100. * batch_idx / batch_len, loss.item()))
        except torch._C._LinAlgError:
            pass

        batch_multiply_count -= 1
        
        if args.dry_run:
            break

        if batch_len < batch_idx:
            break
    
    optimizer.step()


def validation(args, model, device, test_loader, x_norm, y_norm, mmse_para, do_print=False):
    model.eval()
    with torch.no_grad():
        test_loss = torch.tensor(0.0, device=device, requires_grad=False)
        test_heur_loss = torch.tensor(0.0, device=device, requires_grad=False)
        test_mmse_loss = torch.tensor(0.0, device=device, requires_grad=False)

        test_cos_loss = torch.tensor(0.0, device=device, requires_grad=False)
        test_heur_cos_loss = torch.tensor(0.0, device=device, requires_grad=False)
        test_mmse_cos_loss = torch.tensor(0.0, device=device, requires_grad=False)

        test_unable_heur = 0

        batch_len = len(test_loader)

        l = torch.nn.MSELoss(reduction='mean')
    
        for batch_idx, (data, heur, target) in enumerate(test_loader):
            data, target, heur = data.to(device, non_blocking=True), target.to(device, non_blocking=True), heur.to(device, non_blocking=True)
            
            data *= x_norm
            heur *= y_norm
            
            output = model(data, heur)
            output /= y_norm

            mse = l(output, target)
            norm_amp = norm_amp_loss(output, target)

            if torch.isnan(mse).any() or torch.isnan(norm_amp).any():
                error_logger(batch_idx, data, heur, target, output, args.log + ".log", step_type="test")
            
            test_loss += mse.item()
            test_cos_loss += norm_amp.item()

            data /= x_norm
            heur /= y_norm
            
            mmse = calculate_mmse(data, mmse_para[0], mmse_para[1])
            mmse = spread_complex(mmse)

            test_heur_loss += l(heur, target).item()
            test_mmse_loss += l(mmse, target).item()

            test_heur_cos_loss += norm_amp_loss(heur, target).item()
            test_mmse_cos_loss += norm_amp_loss(mmse, target).item()

            if batch_len <= batch_idx:
                break

        test_loss = float(test_loss.cpu().item())
        test_heur_loss = float(test_heur_loss.cpu().item())
        test_mmse_loss = float(test_mmse_loss.cpu().item())

        test_cos_loss = float(test_cos_loss.cpu().item())
        test_heur_cos_loss = float(test_heur_cos_loss.cpu().item())
        test_mmse_cos_loss = float(test_mmse_cos_loss.cpu().item())
        
        test_loss /= batch_len
        test_cos_loss /= batch_len

        test_heur_loss /= batch_len
        test_mmse_loss /= batch_len

        test_heur_cos_loss /= batch_len
        test_mmse_cos_loss /= batch_len

        if do_print:
            print('\nAverage loss: {:.6f}, Huristic Average Loss: {:.6f}, MMSE Average Loss: {:.6f}, Unable heur : {:.2f}%\n'.format(
                test_loss*1000000, test_heur_loss*1000000, test_mmse_loss*1000000, test_unable_heur*100))

    return test_loss, float(test_heur_loss), float(test_mmse_loss), test_cos_loss, float(test_heur_cos_loss), float(test_mmse_cos_loss), test_unable_heur


def training_model(args, model, device, val_data_num, do_print=False, early_stopping_patience=3):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False}

    if use_cuda:
        cuda_kwargs = {'num_workers': args.worker,
                       'pin_memory': False, 
                       'persistent_workers': True}

        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset_handler = DatasetHandler(row_size=args.W, aug_ratio1=args.aug_ratio1, aug_ratio2=args.aug_ratio2, dry_run=args.dry_run, add_noise=args.noise_add, val_noise=args.val_noise_add, postfix=args.dataset_postfix)

    training_dataset = dataset_handler.training_dataset
    if do_print:
        print("Training Dataset : ", len(training_dataset))
    training_test_dataset = dataset_handler.training_test_dataset
    if do_print:
        print("training Test Dataset : ", len(training_test_dataset))
    validation_dataset = dataset_handler.validation_dataset
    if do_print:
        print("Test Dataset : ", len(validation_dataset))

    train_loader = torch.utils.data.DataLoader(training_dataset, **train_kwargs)
    train_test_loader = torch.utils.data.DataLoader(training_test_dataset, **test_kwargs)
    valid_test_loader = torch.utils.data.DataLoader(validation_dataset, **test_kwargs)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load Normalization
    norm_vector = load_cache("cache/"+args.log + '.norm', True)
    if norm_vector is None:
        norm_vector = training_dataset.getNormPara()
        save_cache(norm_vector, "cache/"+args.log + '.norm', True)

    x_norm_vector = norm_vector[0].to(device)
    y_norm_vector = norm_vector[1].to(device)

    # mmse_para = (C_h, C_w)
    mmse_para = load_cache("cache/"+args.log + '.mmse', True)
    if mmse_para is None:
        mmse_para = training_dataset.getMMSEpara()
        save_cache(mmse_para, "cache/"+args.log + '.mmse', True)
    
    mmse_para = (mmse_para[0].to(device), mmse_para[1].to(device))

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    logCSV = None
    dir_name = "result/"+'_'.join(args.log.split("_")[:-2])
    file_name = '_'.join(args.log.split("_")[-2:])+'.csv'
    if args.log is not None:
        os.makedirs(dir_name, exist_ok=True)
        logfile = open('/'.join([dir_name, file_name]), "w")

        logCSV = csv.writer(logfile)
        logCSV.writerow(["epoch", "train loss", "test loss", "train ls loss", "test ls loss", "train mmse", "test mmse", "train cos loss", "test cos loss", "train ls cos loss", "test ls cos loss", "train cos mmse", "test cos mmse", "train unable count", "test unable count"])
    else:
        logfile = None
    
    min_cos_loss = float('inf')
    min_loss = float('inf')
    early_stopping_ctr = early_stopping_patience
    opt_model_para = None

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train(args, model, device, train_loader, optimizer, epoch, x_norm_vector, y_norm_vector, do_print)
        end_time = time.time()

        consumed_time = end_time - start_time

        if do_print:
            print("Training Consumed time: ", consumed_time)

        start_time = time.time()
        if do_print:
            print("<< Test Loader >>")
        test_loss, test_heur_loss, test_mmse, test_cos_loss, test_heur_cos_loss, test_mmse_cos, test_unable = validation(args, model, device, valid_test_loader, x_norm_vector, y_norm_vector, mmse_para, do_print)

        scheduler.step()
        with torch.cuda.device('cuda:'+str(args.gpunum)):
            torch.cuda.empty_cache()

        if do_print:
            print("<< Train Loader >>")
        train_loss, train_heur_loss, train_mmse, train_cos_loss, train_heur_cos_loss, train_mmse_cos, train_unable = validation(args, model, device, train_test_loader, x_norm_vector, y_norm_vector, mmse_para, do_print)
        end_time = time.time()

        consumed_time = end_time - start_time

        if do_print:
            print("Validation Consumed time: ", consumed_time)

        if logCSV is not None:
            logCSV.writerow([epoch, train_loss, test_loss, train_heur_loss, test_heur_loss, train_mmse, test_mmse, train_cos_loss, test_cos_loss, train_heur_cos_loss, test_heur_cos_loss, train_mmse_cos, test_mmse_cos, train_unable, test_unable])

        if epoch is args.epochs:
            break

        gc.collect()
        if args.save_model and min_cos_loss > test_cos_loss:
            min_cos_loss = test_cos_loss
            opt_model_para = copy.deepcopy(model.state_dict())

            if args.save_model:
                torch.save(opt_model_para, "cache/"+args.log+'.pt')
        
        if min_loss > test_loss:
            min_loss = test_loss
            early_stopping_ctr = early_stopping_patience
        else:
            early_stopping_ctr -= 1
            if early_stopping_ctr <= 0:
                print("Early stopping Triggered")
                break

        if args.dry_run:
            break

    if logfile is not None:
        logfile.write("FIN\n")
        logfile.close()


def main():
    ArgsHandler.init_args()
    args = ArgsHandler.args

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    print("Connect GPU : ", args.gpunum)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed+256)
    
    if use_cuda:
        if torch.cuda.device_count() >= (args.gpunum-1):
            torch.cuda.set_device(args.gpunum)
        else:
            print("No gpu number")
            exit(1)

    device = torch.device("cuda:"+str(args.gpunum) if use_cuda else "cpu")

    print(args.model)
    model = model_selector(args.model, args.W).to(device)

    training_model(args, model, device, args.val_data_num, True, early_stopping_patience=args.patience)


if __name__ == '__main__':
    main()
