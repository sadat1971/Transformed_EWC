"""
The code performs the "no-reg", "dropout", "EWC" and "TEWC"regularization. So, it can be used to show
the CF effect and baseline regularization process, as well as the TEWC code
"""


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import time
import tqdm
import pandas as pd
import numpy as np
from transformers import get_linear_schedule_with_warmup
import random
import torch.nn.functional as F
import sys
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import argparse
from torch.autograd import 
from copy import deepcopy

# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, classes, dropout1=0.0, dropout2=0.0):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, classes)
        self.dropout1 = nn.Dropout(p=dropout1)
        self.dropout2 = nn.Dropout(p=dropout2)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        #out = self.dropout(out)
        out = self.fc3(out)
        return out

# def (t, use_cuda=True, **kwargs):
#     if torch.cuda.is_available() and use_cuda:
#         t = t.cuda()
#     return (t, **kwargs)


class perform_TEWC(object):
    """
    This is a class that performs the Transformed-EWC. The code is done with the help of following repositories:

    1. https://github.com/lshug/Continual-Keras
    2. https://github.com/AnatoliiPotapov/MNIST-EWC
    3. https://github.com/TimoFlesch/elastic-weight-consolidation
    4. https://github.com/kuc2477/pytorch-ewc
    5. https://github.com/ContinualAI/continual-learning-baselines
    """
    def __init__(self, model, dataset, cumputation_mode):

        self.model = model
        self.dataset = dataset
        self.cumputation_mode = cumputation_mode
        self.parameters = {idx: p for idx, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self.diag_fish = self.diagonal_fisher_computation()

        for n, p in deepcopy(self.parameters).items():
            self._means[n] = p.data

    def diagonal_fisher_computation(self):
        diag_fisher_infos = {}
        for n, p in deepcopy(self.parameters).items():
            p.data.zero_()
            diag_fisher_infos[n] = p.data

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            input = torch.tensor(input).to(device)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                if self.cumputation_mode=="EWC":
                    diag_fisher_infos[n].data += p.grad.data ** 2 / len(self.dataset) ## EWC
                elif self.cumputation_mode=="TEWC":
                    diag_fisher_infos[n].data += p.grad.data ** .5 / len(self.dataset) ##TEWC
                else:
                    print("Wrong input. Please set either EWC or TEWC")
                    break

        diag_fisher_infos = {idx: p for idx, p in diag_fisher_infos.items()}
        return diag_fisher_infos

    def compute_TEWC_Loss(self, model: nn.Module):
        loss = 0
        for idx, p in model.named_parameters():
            if self.cumputation_mode=="EWC":
                _loss = self.diag_fish[idx] * (p - self._means[idx]) ** 2 #EWC
            if self.cumputation_mode=="TEWC":
                _loss = self.diag_fish[idx] * torch.abs(torch.sqrt((p - self._means[idx]))) ## TEWC
            loss += _loss.sum()
        return loss
    
def create_dataloader(train_text, train_labels, batch_size=16):
    # Create the DataLoader for our training set
    '''
    This function will create a dataloader for our training set. The dataloader will help to feed the randomly 
    sampled data on each batch. The batch size is selected to be 16
    '''
    train_data = TensorDataset(torch.tensor(train_text), torch.tensor(train_labels))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader

def initialize_model(model, epochs, train_dataloader,
                     learning_rate=5e-5,decay=0):
    """Initialize the model, optimizer detailts, learning rate, and weight decay rate
    It can be also used to perform the learning rate scheduling. But we didnot use
    the scheduling in our experiment 
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
        # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return optimizer, scheduler

def set_seed(seed_value=42):
    """Set seed for reproducibility. This part is important to ensure reproducibility
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def do_train(model, train_dataloader, epochs, device, optimizer, scheduler, validation_look_up1, validation_look_up2, validation_look_up3, dataset_loc):
    """Train the model.
    In every step ofthe training, we observe the performance across different datasets/domains
    """
    loss_fn = nn.CrossEntropyLoss() ## default choice
    print("Start training...\n")
    train_loss = []
    val_loss1 = []
    val_loss2 = []
    val_loss3 = []
    acc_score1 = []
    acc_score2 = []
    acc_score3 = []
    f1_score1 = []
    f1_score2 = []
    f1_score3 = []
    for epoch_i in range(epochs):

        #For better visulization
        
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # start timer
        t0_epoch, t0_batch = time.time(), time.time()

        #reset the loss
        total_loss, batch_loss, batch_counts = 0, 0, 0

        #set the training mode
        model.train()

        #start training in batches
        for batch_no, batch in enumerate(train_dataloader):
            batch_counts += 1

            train_X, train_y = tuple(t.to(device) for t in batch)

            #zero out the gradient
            model.zero_grad()

            #forward pass
            output = model(train_X.float())


            #compute the loss
            loss = loss_fn(output, train_y)
            batch_loss += loss.item()
            total_loss += loss.item()

            #performing backward pass
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            #scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (batch_no % 100 == 0 and batch_no != 0) or (batch_no == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {batch_no:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking s
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()
            

        
        train_loss.append(total_loss/len(train_dataloader))
        
        ## Now, get the performance for the first task
        df_test1, test_loss1 = predict(model=model, test_X=validation_look_up1[0], test_y=validation_look_up1[1], batch_size=4, device=device)
        accuracy = accuracy_score(df_test1["GT"], df_test1["prediction"])
        f1_1 = f1_score(df_test1["GT"], df_test1["prediction"])
        acc1 = accuracy_score(df_test1["GT"], df_test1["prediction"])
        print("Task 1 F1", f1_1)
        print("Task 1 acc", acc1)
        val_loss1.append(test_loss1)
        acc_score1.append(acc1)
        f1_score1.append(f1_1)
        print("Task 1 test loss", test_loss1)

        ## Now, get the performance for the second task
        df_test2, test_loss2 = predict(model=model, test_X=validation_look_up2[0], test_y=validation_look_up2[1], batch_size=4, device=device)
        accuracy = accuracy_score(df_test2["GT"], df_test2["prediction"])
        f1_2 = f1_score(df_test2["GT"], df_test2["prediction"])
        acc2 = accuracy_score(df_test2["GT"], df_test2["prediction"])
        print("Task 2 F1", f1_2)
        print("Task 2 acc", acc2)
        val_loss2.append(test_loss2)
        acc_score2.append(acc2)
        f1_score2.append(f1_2)
        print("Task 2 test loss", test_loss2)

        ## Now, get the performance for the second task
        df_test3, test_loss3 = predict(model=model, test_X=validation_look_up3[0], test_y=validation_look_up3[1], batch_size=4, device=device)
        accuracy = accuracy_score(df_test3["GT"], df_test3["prediction"])
        f1_3 = f1_score(df_test3["GT"], df_test3["prediction"])
        acc3 = accuracy_score(df_test3["GT"], df_test3["prediction"])
        print("Task 3 F1", f1_3)
        print("Task 3 acc", acc3)
        val_loss3.append(test_loss3)
        acc_score3.append(acc3)
        f1_score3.append(f1_3)
        print("Task 3 test loss", test_loss3)
    
    record_results = pd.DataFrame()
    record_results["train_loss"] = train_loss
    record_results["val_loss1"] = val_loss1
    record_results["val_loss2"] = val_loss2
    record_results["val_loss3"] = val_loss3
    record_results["acc_score1"] = acc_score1
    record_results["acc_score2"] = acc_score2
    record_results["acc_score3"] = acc_score3
    record_results["f1_score1"] = f1_score1
    record_results["f1_score2"] = f1_score2
    record_results["f1_score3"] = f1_score3


    record_results.to_csv(args.log_dir + "Three_comb/seq_" + str(sequence) + "_"  + str(args.Task1) + \
                           "_next_" + str(args.Task2)+ "_next_" + str(args.Task3)  + str(args.use_L2_or_dropout) +".csv", index=False)

  
    return model

def predict(model, test_X, test_y, batch_size, device, labels_there=True):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    test_dataset = TensorDataset(torch.tensor(test_X), torch.tensor(test_y))
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

    all_logits = []
    total_loss = 0

    # For each batch in our test set...
    for batch in (test_dataloader):
        # Load batch to GPU
        b_input_ids= batch[0].to(device)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids.float())
            total_loss += loss_fn(logits, batch[1].to(device))
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    df = pd.DataFrame()
    if labels_there==True:
        df["GT"] = test_y
    df["probs_1"] = probs[:,1]
    df["prediction"] = df["probs_1"].apply(lambda x:1 if x>=0.5 else 0)
    print("Total test loss: ", total_loss.item()/len(test_dataloader))
    return df, total_loss.item()/len(test_dataloader)

def TEWC_train(model, optimizer, data_loader, ewc, LAMBDA_hyp, task_in_sequence="first"):

    """
    The code peforms one training epoch for the TEWC algorithm

    Input:

    model: The model that was initialized (only once)
    optimizer: The optimizer initialized
    data_loader: The training data loader
    ewc: The EWC object. We can either use EWC or TEWC
    LAMBDA_hyp: The lambda hyperparameter used in the EWC algorithm
    task_in_sequence: If the task in sequence is the first one, we will not apply TEWC

    Output:
    The loss

    """
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = (input), (target)
        optimizer.zero_grad()
        output = model(input)
        if task_in_sequence=="first":
            loss = F.cross_entropy(output, target)
        else:
            loss = F.cross_entropy(output, target) + LAMBDA_hyp * ewc.penalty(model)
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)


set_seed(42)

parser = argparse.ArgumentParser(description='Model arguments')

parser.add_argument("--home_dir", 
                    type=str, 
                    default="/media2/sadat/Sadat/EWC_coursework/",
                     help="Input data path.")

parser.add_argument("--data_dir", 
                    type=str, 
                    default="/media2/sadat/Sadat/EWC_coursework/Data/BERT_Embeddings/",
                     help="Input data path.")
parser.add_argument("--log_dir",
                     type=str, 
                     default="/media2/sadat/Sadat/EWC_coursework/Results/",
                     help="Store result path.")
parser.add_argument("--model_path_dir",
                     type=str, 
                     default="/media2/sadat/Sadat/EWC_coursework/Results/",
                     help="Store result path.")
parser.add_argument("--batch_size", type=int, default=8, help="what is the batch size?")
parser.add_argument("--device", type=str, default="cuda:0", help="what is the device?")
parser.add_argument("--dropout1", type=np.float32, default=0.0, help="what is the first dropout in FC layer?")
parser.add_argument("--dropout2", type=np.float32, default=0.0, help="what is the second dropout in FC layer?")
parser.add_argument("--learning_rate", type=np.float32, default=5e-5, help="what is the learning rate?")
parser.add_argument("--valid_with", type=str, default="test", help="Validate with test data?")
parser.add_argument("--epochs1", type=int, default=4, help="what is the epoch size Task1?")
parser.add_argument("--epochs2", type=int, default=4, help="what is the epoch size Task2?")
parser.add_argument("--epochs3", type=int, default=4, help="what is the epoch size Task3?")
parser.add_argument("--hidden_size1", type=int, default=32, help="what is the first hidden layer size?")
parser.add_argument("--hidden_size2", type=int, default=32, help="what is the second hidden layer size?")
parser.add_argument("--sample", type=str, default="none", help="what kind of sampling you want? Over, Under or none")
parser.add_argument("--model_and_pediction_save", type=str, default="No", help="Do you want to save the model and the results as well?")
parser.add_argument("--fast_track", type=np.float32, default=1.00, help="Do you want a fast track train ?")
parser.add_argument("--Task1", type=str, default="task1", help="First Task")
parser.add_argument("--Task2", type=str, default="task2", help="Second Task")
parser.add_argument("--Task3", type=str, default="task3", help="Third Task")
parser.add_argument("--Task_description", type=str, default="double", help="Is it a single Task or double or triple?")
parser.add_argument("--use_L2_or_dropout", type=str, default="", help="What kind of regularization")
parser.add_argument("--w_decay", type=np.float32, default=0, help="what is the weight decay rate?")
parser.add_argument("--mode", type=str, default="no-reg", help="what is the mode you want to train? Available options are: no-reg, dropout, EWC, TEWC")
parser.add_argument("--Lambda_optimization", type=str, default="no", help="Do you want to perform lambda hyperparameter optimization? yes or no")


def observe_prediction_performance():
    """
    This function will be used to monitor the performance only
    
    """
    ## Now, get the performance for the first task
    df_test1, test_loss1 = predict(model=model, test_X=validation_look_up1[0], test_y=validation_look_up1[1], batch_size=4, device=device)
    accuracy = accuracy_score(df_test1["GT"], df_test1["prediction"])
    f1_1 = f1_score(df_test1["GT"], df_test1["prediction"])
    acc1 = accuracy_score(df_test1["GT"], df_test1["prediction"])
    print("Task 1 F1", f1_1)
    print("Task 1 acc", acc1)
    print("Task 1 test loss", test_loss1)

    ## Now, get the performance for the second task
    df_test2, test_loss2 = predict(model=model, test_X=validation_look_up2[0], test_y=validation_look_up2[1], batch_size=4, device=device)
    accuracy = accuracy_score(df_test2["GT"], df_test2["prediction"])
    f1_2 = f1_score(df_test2["GT"], df_test2["prediction"])
    acc2 = accuracy_score(df_test2["GT"], df_test2["prediction"])
    print("Task 2 F1", f1_2)
    print("Task 2 acc", acc2)
    print("Task 2 test loss", test_loss2)

    ## Now, get the performance for the second task
    df_test3, test_loss3 = predict(model=model, test_X=validation_look_up3[0], test_y=validation_look_up3[1], batch_size=4, device=device)
    accuracy = accuracy_score(df_test3["GT"], df_test3["prediction"])
    f1_3 = f1_score(df_test3["GT"], df_test3["prediction"])
    acc3 = accuracy_score(df_test3["GT"], df_test3["prediction"])
    print("Task 3 F1", f1_3)
    print("Task 3 acc", acc3)
    print("Task 3 test loss", test_loss3)

args = parser.parse_args()

device = args.device
# first, we'll see if we have CUDA available
if torch.cuda.is_available():       
    device = torch.device(device)
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


## Prepare the train and test for the task 1
train1 = np.load(args.data_dir + args.Task1 + "_train.npz")
test1 = np.load(args.data_dir + args.Task1 + "_test.npz")
validation_look_up1 = (test1["feat"], test1["label"])
train_dataloader1 = create_dataloader(train1["feat"], train1["label"], batch_size=args.batch_size) 


## Prepare the train and test for the task 2
train2 = np.load(args.data_dir + args.Task2 + "_train.npz")
test2 = np.load(args.data_dir + args.Task2 + "_test.npz")
validation_look_up2 = (test2["feat"], test2["label"])
train_dataloader2 = create_dataloader(train2["feat"], train2["label"], batch_size=args.batch_size) 

## Prepare the train and test for the task 3
train3 = np.load(args.data_dir + args.Task3 + "_train.npz")
test3 = np.load(args.data_dir + args.Task3 + "_test.npz")
validation_look_up3 = (test3["feat"], test3["label"])
train_dataloader3 = create_dataloader(train3["feat"], train3["label"], batch_size=args.batch_size) 


model = NeuralNet(input_size=768, hidden_size1=args.hidden_size1, hidden_size2=args.hidden_size2, 
                  classes=2, dropout1=args.dropout1, dropout2=args.dropout2).to(device)
optimizer, scheduler = initialize_model(model=model, epochs=args.epochs1, 
                                        train_dataloader=train_dataloader1, learning_rate=args.learning_rate, decay=args.w_decay)

if args.mode=="no--reg" or args.mode=="dropout":
    sequence = 1
    model = do_train(model, train_dataloader1, epochs=args.epochs1, device=device, optimizer=optimizer,
                scheduler=scheduler, validation_look_up1 = validation_look_up1, validation_look_up2 = validation_look_up2,validation_look_up3 = validation_look_up3, 
                    dataset_loc=args.data_dir)

    sequence = 2
    model = do_train(model, train_dataloader2, epochs=args.epochs2, device=device, optimizer=optimizer,
                scheduler=scheduler, validation_look_up1 = validation_look_up1, 
                validation_look_up2 = validation_look_up2,validation_look_up3 = validation_look_up3,  dataset_loc=args.data_dir)

    sequence = 3

    model = do_train(model, train_dataloader3, epochs=args.epochs3, device=device, optimizer=optimizer,
                scheduler=scheduler, validation_look_up1 = validation_look_up1, 
                validation_look_up2 = validation_look_up2, validation_look_up3 = validation_look_up3, dataset_loc=args.data_dir)
    
else:


    sequence = 1
    # For the first task, the subsample is not important, since TEWC will not be in effect
    sz = train1["feat"].shape[0]
    sub_sampled = train1["feat"][random.sample(range(sz), sz*.05), :]
    TEWC_instance = perform_TEWC(model=model, dataset=sub_sampled, cumputation_mode="TEWC")

    for i in range(args.epochs1):

        print(TEWC_train(model, optimizer, train_dataloader2, TEWC_instance, LAMBDA_hyp=1))
        observe_prediction_performance()

    sequence = 2

    sz = train1["feat"].shape[0]
    sub_sampled = train1["feat"][random.sample(range(sz), sz*.05), :]
    TEWC_instance = perform_TEWC(model=model, dataset=sub_sampled, cumputation_mode="TEWC")

    for i in range(args.epochs1):
        ## Now we will perform some 
        if args.Lambda_optimization=="yes":
            lambda_list = np.linspace(1, 100, 50)
            for L in lambda_list:
                print(TEWC_train(model, optimizer, train_dataloader2, TEWC_instance, LAMBDA_hyp=L))
                observe_prediction_performance()
        else:
            print(TEWC_train(model, optimizer, train_dataloader2, TEWC_instance, LAMBDA_hyp=10))
            observe_prediction_performance()

    sequence = 3

    train_prevs = np.concatenate(train1["feat"], train2["feat"], axis=0)
    sz = train_prevs["feat"].shape[0]
    sub_sampled = train_prevs["feat"][random.sample(range(sz), sz*.05), :]
    TEWC_instance = perform_TEWC(model=model, dataset=sub_sampled, cumputation_mode="TEWC")

    for i in range(args.epochs1):
        ## Now we will perform some 
        if args.Lambda_optimization=="yes":
            lambda_list = np.linspace(1, 100, 50)
            for L in lambda_list:
                print(TEWC_train(model, optimizer, train_dataloader2, TEWC_instance, LAMBDA_hyp=L))
                observe_prediction_performance()
        else:
            print(TEWC_train(model, optimizer, train_dataloader2, TEWC_instance, LAMBDA_hyp=10))
            observe_prediction_performance()


