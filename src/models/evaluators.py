from torch import nn, optim
from torch.utils.data import DataLoader
from torchsummary import summary
from .models import *
from .specialLossFuncs import *
from .utils.averageMeter import *

class EvaluatorNetworks():
    def __init__(self, model_idx, device, train_ds, val_ds, optimizer_name='Adam', learning_rate=0.0001, n_epochs=100, batches_size=(256,64), margin=None, is_improved=False):
        self.model_idx = model_idx
        self.device = device
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.train_batch, self.val_batch = batches_size
        self.margin = margin
        self.is_improved = is_improved
        self.initializeModelAndLoss()
        self.initializeOptmizer()
        self.loadValAndTrainDataloader()
        self.trainNetwork()
    
    def initializeModelAndLoss(self):
        if self.model_idx == 0:
            self.model = AE().to(self.device)
            self.loss_fn = nn.MSELoss()
        elif self.model_idx == 1:
            self.model = CNN().to(self.device)
            self.loss_fn = ContrastiveLoss(margin=self.margin)
        elif self.model_idx == 2:
            self.model = CNN().to(self.device)
            self.loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2, eps=1e-7)
        else:
            print("\nInitialize a valid model!\n")

    def initializeOptmizer(self):
        if self.optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            print("\nInitialize a valid optmizer!\n")
    
    def loadValAndTrainDataloader(self):
        self.train_dl = DataLoader(self.train_ds, batch_size=self.train_batch, shuffle=True, num_workers=0, pin_memory=True)
        if self.model_idx != 0:
            self.val_ds.setRefDatasetBasedOnFlagMode(False)
        self.val_dl = DataLoader(self.val_ds, batch_size=self.val_batch, shuffle=True, num_workers=0, pin_memory=True)
    
    def setTrainMode(self):
        self.model.train()
        self.train_loss = AverageMeter()
        
    def setValMode(self):
        self.model.eval()
        self.val_loss = AverageMeter()
        
    def trainStepAE(self):
        self.setTrainMode()
        for input in iter(self.train_dl):
            input = input.to(self.device).squeeze(1)
            with torch.enable_grad():
                self.optimizer.zero_grad()
                _, decoder_output = self.model(input)
                loss = self.loss_fn(input, decoder_output)
                self.train_loss.update(loss.item(), len(input))
                loss.backward()
                self.optimizer.step()
        self.history['train_loss'].append(self.train_loss.avg)
        
    def valStepAE(self):
        self.setValMode()
        with torch.no_grad():
            for input in iter(self.val_dl):
                input = input.to(self.device).squeeze(1)
                _, decoder_output = self.model(input)
                loss = self.loss_fn(input, decoder_output)
                self.val_loss.update(loss.item(), len(input))
        self.history['val_loss'].append(self.val_loss.avg)
        
    def trainStepSN(self):
        self.setTrainMode()
        for input1, input2, label in iter(self.train_dl):
            input1 = input1.to(self.device).squeeze(1)
            input2 = input2.to(self.device).squeeze(1)
            label = label.to(self.device)
            with torch.enable_grad():
                self.optimizer.zero_grad()
                output1, output2 = self.model(input1), self.model(input2)
                loss = self.loss_fn(output1, output2, label)
                self.train_loss.update(loss.item(), len(output1))
                loss.backward()
                self.optimizer.step()
        self.history['train_loss'].append(self.train_loss.avg)
        
    def valStepSN(self):
        self.setValMode()
        with torch.no_grad():
            for input1, input2, label in iter(self.val_dl):
                input1 = input1.to(self.device).squeeze(1)
                input2 = input2.to(self.device).squeeze(1)
                label = label.to(self.device)
                output1, output2 = self.model(input1), self.model(input2)
                loss = self.loss_fn(output1, output2, label)
                self.val_loss.update(loss.item(), len(output1))
        self.history['val_loss'].append(self.val_loss.avg)      
    
    def trainStepTPN(self):
        self.setTrainMode()
        for anchor, positive, negative in iter(self.train_dl):
            anchor = anchor.to(self.device).squeeze(1)
            positive = positive.to(self.device).squeeze(1)
            negative = negative.to(self.device).squeeze(1)
            with torch.enable_grad():
                self.optimizer.zero_grad()
                a_output, p_output, n_output = self.model(anchor), self.model(positive), self.model(negative)
                if self.is_improved:
                    a_output = torch.mean(a_output, dim=0).unsqueeze(0).repeat(a_output.shape[0], 1)
                loss = self.loss_fn(a_output, p_output, n_output)
                self.train_loss.update(loss.item(), len(a_output))
                loss.backward()
            self.optimizer.step()
        self.history['train_loss'].append(self.train_loss.avg)
   
    def valStepTPN(self):
        self.setValMode()
        with torch.no_grad():
            for anchor, positive, negative in iter(self.val_dl):
                anchor = anchor.to(self.device).squeeze(1)
                positive = positive.to(self.device).squeeze(1)
                negative = negative.to(self.device).squeeze(1)
                a_output, p_output, n_output = self.model(anchor), self.model(positive), self.model(negative)
                if self.is_improved:
                    a_output = torch.mean(a_output, dim=0).unsqueeze(0).repeat(a_output.shape[0], 1)
                loss = self.loss_fn(a_output, p_output, n_output)
                self.val_loss.update(loss.item(), len(a_output))
        self.history['val_loss'].append(self.val_loss.avg)
    
    def trainNetwork(self):
        self.history = {'train_loss': [], 'val_loss': []}
        for epoch in range(self.n_epochs):
            if self.model_idx == 0:
                self.trainStepAE()
                self.valStepAE()
            elif self.model_idx == 1:
                self.trainStepSN()
                self.valStepSN()
            elif self.model_idx == 2:
                self.trainStepTPN()
                self.valStepTPN()
            print(f"Epoch [{epoch + 1}/{self.n_epochs}]")
            print("-" * 35)
            print(f"Train loss: {round(self.history['train_loss'][epoch], 6):<7}")
            print(f"Valid loss: {round(self.history['val_loss'][epoch], 6):<7}\n")
    
    def evaluateSamplesSet(self, samples_set):
        self.model.eval()
        with torch.no_grad():
            all_evaluated_samples_set = [self.model(sample.unsqueeze(0).to(self.device)) if self.model_idx != 0 else self.model(sample.unsqueeze(0).to(self.device))[0] for sample in samples_set]
        return all_evaluated_samples_set
    
    def describesNetworkSummary(self, input_size=(8,60)):
       summary(self.model, input_size=input_size, batch_size=self.train_batch, device=self.device)