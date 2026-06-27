import os
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


class Trainer:

    def __init__(self, model, train_loader, val_loader, config, device=None):
        #deivce
        if device is None :
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else :
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        #hyperparameters 
        self.epochs = config.get("epochs", 30)
        self.lr = config.get("lr", 0.001)
        self.weight_decay = config.get("weight_decay", 1e-4)
        self.patience = config.get("patience", 7)
        self.model_name = config.get("model_name", "model")

        #Checkpoint

        self.checkpoint_dir = config.get("checkpoint_dir", "outputs/checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        #loss, optimizer, scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        #every 10 epochs lr*0.1
        self.scheduler = StepLR(self.optimizer,step_size=10,gamma=0.1)

        #history
        self.history = {
            "train_loss" : [],
            "train_acc" : [],
            "val_loss" : [],
            "val_acc" : [],
        }

        self.best_val_acc = 0.0
        self.no_improve_count = 0

    
    def train(self):

        print(f"\n{'='*55}")
        print(f"  مدل    : {self.model_name}")
        print(f"  Device : {self.device}")
        print(f"  Epochs : {self.epochs}  |  LR: {self.lr}")
        print(f"{'='*55}\n")

        start = time.time()

        for epoch in range(1, self.epochs+1):
            train_loss, train_acc= self._run_epoch(epoch, phase="train")
            val_loss,val_acc= self._run_epoch(epoch, phase="val")

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            self.scheduler.step()

            # epoch summary

            print(f"Epoch [{epoch:02d}/{self.epochs}]"
                  f"Train Loss :{train_loss:.4f}, Accuracy: {train_acc:.2f}"
                  f"Validation loss {val_loss:.4f}, Validation Accuracy :{val_acc:.2f}"
                  )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, tag="best")
                print(f"Best model saved (val_acc:{val_acc:.2f}%)\n")
                self.no_improve_count=0
            else :
                self.no_improve_count += 1
            
            # early stopping 
            if self.patience and self.no_improve_count >= self.patience :
                print(f"early stopping in epoch {epoch}")
                break

        elapsed = time.time() - start
        print(f"\n{'='*55}")
        print(f"  تمام شد  |  زمان: {elapsed/60:.1f} دقیقه")
        print(f"  بهترین Val Acc: {self.best_val_acc:.2f}%")
        print(f"{'='*55}\n")

        self._save_checkpoint(epoch, tag="last")

        return self.history
    

    def _run_epoch(self, epoch, phase):
        is_train = (phase == "train")

        self.model.train() if is_train else self.model.eval()

        loader = self.train_loader if is_train else self.val_loader

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader,
                    desc=f"{'train' if is_train else 'val'}[{epoch:02d}]",
                    leave=False)
        
        ctx= torch.enable_grad() if is_train else torch.no_grad()

        with ctx :
            for images , labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                if is_train :
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item() * images.size(0)
                _,predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                pbar.set_postfix({
                    "loss" : f"{total_loss/total:.4f}",
                    "acc" : f"{100.*correct/total:.1f}%"
                })
        
        return total_loss/total, 100.0* correct/total

    def _save_checkpoint(self, epoch, tag="best") :

        path = os.path.join(self.checkpoint_dir, f"{self.model_name}_{tag}.pth")
        
        torch.save(
            {
                "epoch" : epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_val_acc": self.best_val_acc,
                "history":self.history,
                "config": self.config,
            }, path)
    def load_best(self):
        path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pth")
        if not os.path.exists(path):
            raise FileNotFoundError
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"✓ بارگذاری شد (epoch={ckpt['epoch']}, val_acc={ckpt['best_val_acc']:.2f}%)")
        return ckpt
    
    def evaluate(self, test_loader):
        self.model.eval()
        total_loss,correct,total = 0.0, 0, 0

        with torch.no_grad():
            for images,labels in tqdm(test_loader, desc="Test"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _,predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        test_loss = total_loss / total
        test_acc = 100.0 * correct / total

        print(f"Test loss{test_loss:.4f} | Test Acc{test_acc:.2f}")

        return test_loss, test_acc