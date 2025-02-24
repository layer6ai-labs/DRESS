import datetime
import os


class ExperimentTracker:
    def __init__(self, results_dir, config_path, name, version_name=None, version_prefix=''):
        self.results_dir = results_dir
        self.name = name
        self.config_file = config_path

        if self.name is None or self.name == "":
            raise Exception("Experiment name must be specified.")

        if version_name is not None:
            self.version = version_name
        else:
            self.version_prefix = "" if version_prefix == "" else f"{version_prefix}-"
            self.version = f'{self.version_prefix}{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
            
        self.path = os.path.join(self.results_dir, self.name, self.version)

        if not os.path.exists(self.path):
            os.makedirs(self.path)
            print('Created new experiment directory at: {}'.format(self.path))

        self.results_csv_path = os.path.join(self.path, 'output.csv')

if __name__ == "__main__":
    exp_tracker = ExperimentTracker(
        results_dir='/home/keyvan/Research/meta-learning/results/diverse/',
        config_path='config.yaml',
        name='test_exp')
    
    print(exp_tracker.name)
    print(exp_tracker.version)
    print(exp_tracker.path)