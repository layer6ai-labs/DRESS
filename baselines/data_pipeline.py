import torch


class TasksDataset:
    def __init__(self, args, num_tasks, meta_split, task_generator):
        self.num_tasks = num_tasks
        self.meta_split = meta_split
        self.args = args
        self.task_generator = task_generator
        
    def __len__(self):
        return self.num_tasks
    
    def __getitem__(self, index):
        task = self.task_generator.sample_task(self.meta_split, self.args)
        support_task = torch.cat([task[0][:self.args.KShot], task[3][:self.args.KShot]], dim=0)
        query_task = torch.cat([task[0][self.args.KShot:], task[3][self.args.KShot:]], dim=0)
        return torch.cat([support_task, query_task], dim=0)