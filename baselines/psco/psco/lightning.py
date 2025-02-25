import torch
import torch.optim as optim
import lightning as L
import torch.nn.functional as F
import numpy as np

from psco.models import PsCoBYOL, PsCoMoCo
from sklearn.neighbors import KNeighborsClassifier

class PsCoLightning(L.LightningModule):
    def __init__(self, model_name, input_shape, num_train_batches, lr, wd, num_epochs,
                 N, Q, K, metric, backbone, momentum, temperature, queue_size, prediction, 
                 shot_sampling, temperature2, sinkhorn_iter, random_seed):
        super(PsCoLightning, self).__init__()
                
        if model_name == 'psco':
            model_cls = PsCoMoCo
        elif model_name == 'psco_byol':
            model_cls = PsCoBYOL
        else:
            raise ModuleNotFoundError(f"{model_name} is not supported!")
        
        self.model = model_cls(backbone=backbone,
                               input_shape=input_shape,
                               momentum=momentum,
                               temperature=temperature,
                               queue_size=queue_size,
                               prediction=prediction,
                               num_shots=K,
                               shot_sampling=shot_sampling,
                               temperature2=temperature2,
                               sinkhorn_iter=sinkhorn_iter)
                
        self.num_train_batches = num_train_batches
        
        self.K = K
        self.N = N
        self.Q = Q
        self.metric = metric
        self.lr = lr
        self.wd = wd
        self.num_epochs = num_epochs
        
        self.save_hyperparameters({
            'model_name': model_name,
            'input_shape': input_shape,
            'num_train_batches': num_train_batches,
            'lr': lr,
            'wd': wd,
            'num_epochs': num_epochs,
            'N': N,
            'Q': Q,
            'K': K,
            'metric': metric,
            'backbone': backbone,
            'momentum': momentum,
            'temperature': temperature,
            'queue_size': queue_size,
            'prediction': prediction,
            'shot_sampling': shot_sampling,
            'temperature2': temperature2,
            'sinkhorn_iter': sinkhorn_iter,
            'random_seed': random_seed
        })
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        outputs = self.model([x, y])
        self.log('train/loss', outputs['loss'], 
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.num_train_batches)
        return outputs['loss']
    
    def eval_step(self, batch, batch_idx):
        x = batch
        B = x.shape[0]
        input_shape = x.shape[-3:]
        
        x = x.view(B, self.N, self.K + self.Q, *input_shape)
        
        shots = x[:, :, :self.K].reshape(B * self.N * self.K, *input_shape)
        queries = x[:, :, self.K:].reshape(B * self.N * self.Q, *input_shape)

        if self.metric == 'knn':
            knn = KNeighborsClassifier(n_neighbors=self.K, metric='cosine')
            shots_knn   = F.normalize(self.model(shots,   mode='feature')).detach().cpu().numpy()
            queries_knn = F.normalize(self.model(queries, mode='feature')).detach().cpu().numpy()

            shots_knn = shots_knn.reshape(B, self.N * self.K, -1)
            queries_knn = queries_knn.reshape(B, self.N * self.Q, -1)

            y_shots = np.tile(np.expand_dims(np.arange(self.N), 1), self.K).reshape(-1)
            batch_accuracy = []
            for i in range(B):
                knn.fit(shots_knn[i, ...], y_shots)        
                preds = np.array(knn.predict(queries_knn[i, ...]))
                labels = np.tile(np.expand_dims(np.arange(self.N), 1), self.Q).reshape(-1)
                acc = (preds == labels).mean().item()
                batch_accuracy.append(acc)

            batch_accuracy = np.array(batch_accuracy).mean()
        elif self.metric == 'supcon':
            queries_supcon = F.normalize(self.model(queries, mode='feature', momentum=False, projection=True, prediction=True))
            shots_supcon = F.normalize(self.model(shots,   mode='feature', momentum=False, projection=True))

            queries_supcon = queries_supcon.view(x.shape[0], self.N * self.Q, queries_supcon.shape[-1])
            shots_supcon = shots_supcon.view(x.shape[0], self.N, self.K, shots_supcon.shape[-1])

            prototypes = F.normalize(shots_supcon).mean(dim=2)
            dot_product = torch.bmm(queries_supcon, prototypes.transpose(1, 2))
            preds = dot_product.argmax(dim=2)
            labels = torch.arange(self.N, device=preds.device).repeat_interleave(self.Q).repeat(x.shape[0]).view(x.shape[0], -1)
            batch_accuracy = (preds == labels).float().mean().item()
        else:
            raise NotImplementedError(f"Metric {self.metric} not implemented")
        
        return batch_accuracy
        
    def validation_step(self, batch, batch_idx):
        batch_accuracy = self.eval_step(batch, batch_idx)
        self.log('val/accuracy', batch_accuracy, 
                on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.shape[0])
        
    def test_step(self, batch, batch_idx):
        batch_accuracy = self.eval_step(batch, batch_idx)
        self.log('test/accuracy', batch_accuracy, 
                on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.shape[0])
        

    def configure_optimizers(self):
        optimizer = optim.SGD([p for p in self.model.parameters() if p.requires_grad],
                              lr=self.lr, momentum=0.9, weight_decay=self.wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs * self.num_train_batches)

        return [optimizer], [scheduler]