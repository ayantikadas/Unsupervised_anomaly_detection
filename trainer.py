import numpy as np
import torch
from matplotlib import pyplot as plt
import copy
import glob
import os
import random
from collections import namedtuple
from tqdm.notebook import tqdm
import ants
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics 
from torchsummary import summary
import monai
import mlflow

class Trainer:
    def __init__(self, task, max_epochs=1,top_level_task='reconstruction',\
                 loss_components_name_train=['loss'],\
                 loss_components_name_val=['loss'],\
                 experiment_name = 'classification_dummy', \
                 log_freq = 5, logging = False,\
                 pre_trained = False,\
                 hyper_params_tune = False, logger=None):
        self.task = task
        self.max_epochs = max_epochs
        self.logger = logger   
        self._experiment_name = experiment_name
        self.top_level_task = top_level_task
        self.loss_components_name_train = loss_components_name_train
        self.loss_components_name_val = loss_components_name_val
        self.pre_trained = pre_trained
        self.hyper_params_tune = hyper_params_tune
        self.log_freq = log_freq
        self.logging = logging        
        self.seed = 20
        
    
    def load_pretrained_model(self,model):

        Experiment_ID = mlflow.get_experiment_by_name(self._experiment_name).experiment_id
        df = mlflow.search_runs(experiment_ids = Experiment_ID)
        mlf_runid = df.loc[df['metrics.least_loss'].idxmin()]['run_id']
        path = "mlruns/"+str(Experiment_ID)+"/"+str(mlf_runid)+"/checkpoints"
        model_name = os.listdir(path)[0]
        full_path = path+"/"+model_name
        ckpt_keys = torch.load(full_path).keys()
        model.load_state_dict(torch.load(full_path))
        
        
    def train_loop(self,train_loader,current_epoch):
        
        ################### Device allocation
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        ################### Pushing Model to device
        self.task.model.to(device)
        self.task.model.train()
        
        desc = "Epoch = {}/{}  Mini-batch loss = {:.4f}"
        pbar = tqdm(initial=0, leave=False, total=len(train_loader),desc=desc.format(0,0,0))

        ################### Dictionary Epoch loss         
        epoch_loss = {}        
        for no_,comp_name in enumerate(self.loss_components_name_train):
            epoch_loss.update({comp_name:torch.FloatTensor()})
            
        ################### Random batch selection whose ouput images would be stored     
        store_batch = torch.randperm(len(train_loader))[0]
        

        for batch_num, batch_data in enumerate(train_loader): 
            for l_ in range(len(batch_data)):
                batch_data[l_] = batch_data[l_].to(device)
            self.task.optimizer.zero_grad()
            
            if self.top_level_task == 'segmentation' or self.top_level_task == 'classification':
                inputs, labels = batch_data[0], batch_data[1]
                outputs = self.task.model(inputs)
                loss, loss_components = self.task.criterion(outputs, labels,\
                                                            self.task.model,\
                                                            batch_data
                                                           )
                                                           
                if store_batch == batch_num:
                    store_batch_inputs = labels
                    store_batch_outputs = outputs
                
            elif self.top_level_task == 'reconstruction':
                inputs = batch_data[0]              
                outputs = self.task.model(inputs)
                loss, loss_components = self.task.criterion(outputs, inputs,\
                                                           self.task.model,\
                                                            batch_data
                                                           )
                ################## store images for the epoch from one random batch
                if store_batch == batch_num:
                    store_batch_inputs = inputs
                    store_batch_outputs = outputs

            
            loss.backward()
            self.task.optimizer.step()
            
            ################### store component_wise losses
            for no_,comp_name in enumerate(self.loss_components_name_train):
                if batch_num == 0:
                    epoch_loss[comp_name] = loss_components[no_].item()
                else:
                    epoch_loss[comp_name] += loss_components[no_].item()
            
            
            
            
            pbar.desc = desc.format(current_epoch+1,self.max_epochs,loss.item())
            pbar.update(1)
            
        for _,comp_name in enumerate(self.loss_components_name_train):
            epoch_loss[comp_name] = epoch_loss[comp_name]/len(train_loader)
        
        pbar.n = pbar.last_print_n = 0
        pbar.close()
        
          
        return epoch_loss,store_batch_inputs,store_batch_outputs

    
    def val_loop(self,val_loader):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.task.model.to(device)
        
        self.task.model.eval()
#         self.task.model.train()

        store_batch_val = torch.randperm(len(val_loader))[0]

        with torch.no_grad():
            ################### Dictionary Epoch loss         
            val_epoch_loss = {}        
            for no_,comp_name in enumerate(self.loss_components_name_val):
                val_epoch_loss.update({comp_name:torch.FloatTensor()})


            for batch_val,val_data in enumerate(val_loader):
                
                for l_ in range(len(val_data)):
                    val_data[l_] = val_data[l_].to(device)
                if self.top_level_task == 'segmentation' or self.top_level_task == 'classification':                    
                    val_images, val_labels = val_data[0],val_data[1]
                    model_output = self.task.model(val_images)
                    _, val_loss_components = self.task.criterion(model_output, \
                                                                val_labels,\
                                                                self.task.model,\
                                                                val_data
                                                           )
                    ################## store images for the epoch from one random batch                
                    if store_batch_val == batch_val:
                        store_batch_inputs_val = val_labels
                        store_batch_outputs_val = model_output  
                        
                elif self.top_level_task == 'reconstruction':
                    val_images = val_data[0]
                    model_output = self.task.model(val_images)
                    _, val_loss_components = self.task.criterion(model_output,\
                                                                val_images,\
                                                                self.task.model,\
                                                                val_data
                                                           )                    
                    ################## store images for the epoch from one random batch                
                    if store_batch_val == batch_val:
                        store_batch_inputs_val = val_images
                        store_batch_outputs_val = model_output              

                
                ################### store component_wise losses
                for no_,comp_name in enumerate(self.loss_components_name_val):
                    if batch_val == 0:
                        val_epoch_loss[comp_name] = val_loss_components[no_].item()
                    else:
                        val_epoch_loss[comp_name] += val_loss_components[no_].item()
                
                
            for _,comp_name in enumerate(self.loss_components_name_val):
                val_epoch_loss[comp_name] = val_epoch_loss[comp_name]/len(val_loader)    

        
        return val_epoch_loss,store_batch_inputs_val,store_batch_outputs_val
        
    
    def compare_and_log_best_model(self,val_loss,current_epoch,max_epoch,model,no_of_batches):
        
        self._model = copy.deepcopy(model)
        if val_loss < self.best_metric:
            self.best_metric = val_loss
            self.best_metric_epoch = current_epoch+1
            self.best_model = copy.deepcopy(model)
            
        save_folder = "mlruns/"+str(self.experiment_ID)+"/"+str(self.current_run_id)+"/"+"checkpoints"
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        if current_epoch % self.log_freq == 0:
            filename_pattern = "epoch_"+str(current_epoch)+".ckpt"
            path = save_folder+"/"+filename_pattern
            torch.save(self._model.to("cpu").state_dict(),path)

        if current_epoch == max_epoch-1:            
            filename_pattern = "best_epoch_"+str(self.best_metric_epoch)+".ckpt"
            path = save_folder+"/"+filename_pattern
            torch.save(self.best_model.to("cpu").state_dict(),path)
            self.logger.log_metrics({"least_loss":self.best_metric},step=current_epoch)
            
    def mlflow_log_images(self,val_batch_inputs,val_batch_outputs,name_):
        if self.top_level_task == 'reconstruction' or self.top_level_task == 'segmentation':                    
            #### Save val images 
            pred_save = val_batch_inputs[0,0,:,:].detach().cpu().numpy()
            if pred_save.min()!=0 and pred_save.max()!=1: 
                pred_save_norm = (pred_save - pred_save.min())/(pred_save.max() - pred_save.min())
            else:
                pred_save_norm = pred_save
            ########## depends on the nature of output from the model
            if type(val_batch_outputs)==tuple:
                org_save = val_batch_outputs[0][0,0,:,:].detach().cpu().numpy()
            else:
                org_save = val_batch_outputs[0,0,:,:].detach().cpu().numpy()
            if org_save.min()!=0 and org_save.max()!=1: 
                org_save_norm = (org_save - org_save.min())/(org_save.max() - org_save.min())
            else:
                org_save_norm = org_save
                
            mlflow.log_image(np.concatenate((org_save_norm,pred_save_norm),axis=1), name_+".png")

            
    def fit(self,train_loader,val_loader):
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.best_metric = float("inf")
        
        if self.logging:
            if mlflow.active_run() is not None:
                mlflow.end_run()
            self.experiment = mlflow.get_experiment_by_name(self._experiment_name)
        
            if self.experiment is None:
                self.experiment_ID = mlflow.create_experiment(self._experiment_name)
            else:
                self.experiment_ID = self.experiment.experiment_id                
            
            mlflow.start_run(experiment_id=self.experiment_ID)
            self.current_run_id = mlflow.active_run().info.run_id


        if self.pre_trained:
            self.load_pretrained_model(self.task.model)

        if self.hyper_params_tune:
            new_lr = self.hyperparam_tuning(train_loader, val_loader)

            for params in self.task.optimizer.param_groups:
                params["lr"] = new_lr
        if self.task.scheduler is not None:
            self.task.scheduler.optimizer = self.task.optimizer
        
        for epoch in range(self.max_epochs):
            
            avg_train_loss_total = 0
            avg_val_loss_total = 0
            
            avg_train_loss, train_batch_inputs, train_batch_outputs = self.train_loop(train_loader,epoch)            
            for no_,comp_name in enumerate(self.loss_components_name_train):     
                if self.logging:
                    self.logger.log_metrics({'train_'+comp_name:avg_train_loss[comp_name]},step=epoch+1) 
                avg_train_loss_total += avg_train_loss[comp_name]
                
            
            avg_val_loss, val_batch_inputs, val_batch_outputs = self.val_loop(val_loader)    
#             print(val_batch_outputs.size())
            for no_,comp_name in enumerate(self.loss_components_name_val):            
                if self.logging:
                    self.logger.log_metrics({'val_'+comp_name:avg_val_loss[comp_name]},step=epoch+1) 
                avg_val_loss_total += avg_val_loss[comp_name]
            
            
            if self.task.scheduler is not None:
                self.task.scheduler.step(avg_val_loss_total)                
            
            
            tqdm.write("Training Results - Epoch: {}/{}  Avg loss: {:.4f}".format(epoch+1,self.max_epochs,avg_train_loss_total))
            tqdm.write("Validation Results - Epoch: {}/{} Avg loss: {:.4f}".format(epoch+1,self.max_epochs,avg_val_loss_total))
            if self.logging:                
                self.compare_and_log_best_model(avg_val_loss_total,epoch,self.max_epochs,self.task.model,len(train_loader))
                if self.top_level_task == 'reconstruction' or self.top_level_task == 'segmentation':
                    self.mlflow_log_images(val_batch_inputs,val_batch_outputs,name_=str('val_'+str(epoch+1)+'_'))
                    
        
        if self.logging:
            mlflow.end_run()
            return self.task, avg_train_loss_total, avg_val_loss_total,self.experiment_ID, self.current_run_id
        else:
            return self.task, avg_train_loss_total, avg_val_loss_total
        