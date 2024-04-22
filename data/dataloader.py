import functools
from json import JSONDecodeError

import numpy as np
import pytorch_lightning as pl
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset, DataLoader

from data.utils import load_json
from data.utils.batcher import plan_collator_dict


# Define a custom dataset for handling plans
class PlanDataset(Dataset):
    """
    A custom PyTorch Dataset for handling plans and their indices.

    Attributes:
        plans (list): A list of plans.
        idxs (list): A list of indices corresponding to the plans.
    """

    def __init__(self, plans, idxs):
        """
        Initializes the PlanDataset with plans and their indices.

        Args:
            plans (list): A list of plans.
            idxs (list): A list of indices corresponding to the plans.
        """
        self.plans = plans
        self.idxs = [int(i) for i in idxs]
        assert len(self.plans) == len(self.idxs)

    def __len__(self):
        """
        Returns the number of plans in the dataset.

        Returns:
            int: The number of plans.
        """
        return len(self.plans)

    def __getitem__(self, i: int):
        """
        Retrieves the index and plan at the specified index.

        Args:
            i (int): The index of the plan to retrieve.

        Returns:
            tuple: A tuple containing the index and the plan.
        """
        return self.idxs[i], self.plans[i]


# Define a function for inverse log1p transformation
def _inv_log1p(x):
    """
    Performs the inverse of log1p transformation.

    Args:
        x (float): The value to transform.

    Returns:
        float: The inverse log1p transformed value.
    """
    return np.exp(x) - 1


# Define a function to read workload runs from paths
def read_workload_runs(
    workload_run_paths, limit_queries=None, limit_queries_affected_wl=None
):
    """
    Reads several workload runs from the given paths.

    Args:
        workload_run_paths (list): A list of paths to workload run files.
        limit_queries (int, optional): The total limit of queries to read. Defaults to None.
        limit_queries_affected_wl (int, optional): The limit of queries affected by workload. Defaults to None.

    Returns:
        tuple: A tuple containing the plans and database statistics.
    """
    plans = []
    database_statistics = dict()

    total_workloads = len(workload_run_paths)
    limit_per_ds = None
    if limit_queries is not None and limit_queries_affected_wl is not None:
        limit_per_ds = limit_queries // limit_queries_affected_wl

    for i, source in enumerate(workload_run_paths):
        try:
            run = load_json(source)
        except JSONDecodeError as e:
            raise ValueError(f"Error reading {source}") from e
        database_statistics[i] = run.database_stats
        database_statistics[i]["run_kwargs"] = run.run_kwargs  # Corrected typo here

        if (
            limit_per_ds is not None
            and i >= total_workloads - limit_queries_affected_wl
        ):
            print(f"Capping workload {source} after {limit_per_ds} queries")
            plans.extend(run.parsed_plans[:limit_per_ds])
        else:
            plans.extend(run.parsed_plans)

        for plan in plans[-limit_per_ds:]:
            plan.database_id = i

    print(f"No of Plans: {len(plans)}")

    return plans, database_statistics


# Define a function to derive label normalizer based on loss class name
def derive_label_normalizer(loss_class_name, y):
    """
    Derives a label normalizer pipeline based on the specified loss class name.

    Args:
        loss_class_name (str): The name of the loss class.
        y (np.array): The labels to fit the normalizer.

    Returns:
        sklearn.pipeline.Pipeline: The label normalization pipeline.
    """
    if loss_class_name == "MSELoss":
        log_transformer = preprocessing.FunctionTransformer(
            np.log1p, _inv_log1p, validate=True
        )
        scale_transformer = preprocessing.MinMaxScaler()
        pipeline = Pipeline([("log", log_transformer), ("scale", scale_transformer)])
        pipeline.fit(y.reshape(-1, 1))
    elif loss_class_name == "QLoss":
        scale_transformer = preprocessing.MinMaxScaler(feature_range=(1e-2, 1))
        pipeline = Pipeline([("scale", scale_transformer)])
        pipeline.fit(y.reshape(-1, 1))
    else:
        pipeline = None
    return pipeline


# Define a function to create datasets for training, validation, and testing
def create_datasets(
    workload_run_paths,
    cap_training_samples=None,
    val_ratio=0.15,
    limit_queries=None,
    limit_queries_affected_wl=None,
    shuffle_before_split=True,
    loss_class_name=None,
):
    """
    Creates datasets for training, validation, and testing based on the provided parameters.

    Args:
        workload_run_paths (list): A list of paths to workload run files.
        cap_training_samples (int, optional): The maximum number of training samples. Defaults to None.
        val_ratio (float, optional): The ratio of validation samples. Defaults to 0.15.
        limit_queries (int, optional): The total limit of queries to read. Defaults to None.
        limit_queries_affected_wl (int, optional): The limit of queries affected by workload. Defaults to None.
        shuffle_before_split (bool, optional): Whether to shuffle the data before splitting. Defaults to True.
        loss_class_name (str, optional): The name of the loss class. Defaults to None.

    Returns:
        tuple: A tuple containing the label normalization pipeline, training dataset, validation dataset, and database statistics.
    """
    plans, database_statistics = read_workload_runs(
        workload_run_paths,
        limit_queries=limit_queries,
        limit_queries_affected_wl=limit_queries_affected_wl,
    )

    no_plans = len(plans)
    plan_idxs = list(range(no_plans))
    if shuffle_before_split:
        np.random.shuffle(plan_idxs)

    train_ratio = 1 - val_ratio
    split_train = int(no_plans * train_ratio)
    train_idxs = plan_idxs[:split_train]
    # Limit number of training samples. To have comparable batch sizes, replicate remaining indexes.
    if cap_training_samples is not None:
        prev_train_length = len(train_idxs)
        train_idxs = train_idxs[:cap_training_samples]
        replicate_factor = max(prev_train_length // len(train_idxs), 1)
        train_idxs = train_idxs * replicate_factor

    train_dataset = PlanDataset([plans[i] for i in train_idxs], train_idxs)

    val_dataset = None
    if val_ratio > 0:
        val_idxs = plan_idxs[split_train:]
        val_dataset = PlanDataset([plans[i] for i in val_idxs], val_idxs)

    # derive label normalization
    runtimes = np.array([p.plan_runtime / 1000 for p in plans])
    label_norm = derive_label_normalizer(loss_class_name, runtimes)

    return label_norm, train_dataset, val_dataset, database_statistics


# Define a function to create dataloaders for training, validation, and testing
def create_dataloader(
    workload_run_paths,
    test_workload_run_paths,
    statistics_file,
    plan_featurization_name,
    database,
    val_ratio=0.15,
    batch_size=32,
    shuffle=True,
    num_workers=1,
    pin_memory=False,
    limit_queries=None,
    limit_queries_affected_wl=None,
    loss_class_name=None,
):
    """
    Creates dataloaders for batching physical plans to train the model in a distributed fashion.

    Args:
        workload_run_paths (list): A list of paths to workload run files.
        test_workload_run_paths (list): A list of paths to test workload run files.
        statistics_file (str): The path to the statistics file.
        plan_featurization_name (str): The name of the plan featurization.
        database (str): The name of the database.
        val_ratio (float, optional): The ratio of validation samples. Defaults to 0.15.
        batch_size (int, optional): The batch size. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        num_workers (int, optional): The number of worker threads. Defaults to 1.
        pin_memory (bool, optional): Whether to use pinned memory. Defaults to False.
        limit_queries (int, optional): The total limit of queries to read. Defaults to None.
        limit_queries_affected_wl (int, optional): The limit of queries affected by workload. Defaults to None.
        loss_class_name (str, optional): The name of the loss class. Defaults to None.

    Returns:
        tuple: A tuple containing the label normalization pipeline, feature statistics, training dataloader, validation dataloader, and test dataloaders.
    """
    # split plans into train/test/validation
    label_norm, train_dataset, val_dataset, database_statistics = create_datasets(
        workload_run_paths,
        loss_class_name=loss_class_name,
        val_ratio=val_ratio,
        limit_queries=limit_queries,
        limit_queries_affected_wl=limit_queries_affected_wl,
    )

    # postgres_plan_collator does the heavy lifting of creating the graphs and extracting the features and thus requires both
    # database statistics but also feature statistics
    feature_statistics = load_json(statistics_file, namespace=False)

    plan_collator = plan_collator_dict[database]
    train_collate_fn = functools.partial(
        plan_collator,
        db_statistics=database_statistics,
        feature_statistics=feature_statistics,
        plan_featurization_name=plan_featurization_name,
    )
    dataloader_args = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=pin_memory,
    )
    train_loader = DataLoader(train_dataset, **dataloader_args)
    val_loader = DataLoader(val_dataset, **dataloader_args)

    # for each test workload run create a distinct test loader
    test_loaders = None
    if test_workload_run_paths is not None:
        test_loaders = []
        for p in test_workload_run_paths:
            _, test_dataset, _, test_database_statistics = create_datasets(
                [p],
                loss_class_name=loss_class_name,
                val_ratio=0.0,
                shuffle_before_split=False,
            )
            # test dataset
            test_collate_fn = functools.partial(
                plan_collator,
                db_statistics=test_database_statistics,
                feature_statistics=feature_statistics,
                plan_featurization_name=plan_featurization_name,
            )
            # previously shuffle=False but this resulted in bugs
            dataloader_args.update(collate_fn=test_collate_fn)
            test_loader = DataLoader(test_dataset, **dataloader_args)
            test_loaders.append(test_loader)

    return label_norm, feature_statistics, train_loader, val_loader, test_loaders


# Define a LightningDataModule for handling plan data
class PlanDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning Data Module for handling plan data.

    Attributes:
        workload_run_paths (list): A list of paths to workload run files.
        test_workload_run_paths (list): A list of paths to test workload run files.
        statistics_file (str): The path to the statistics file.
        plan_featurization_name (str): The name of the plan featurization.
        database (str): The name of the database.
        batch_size (int): The batch size.
        val_ratio (float): The ratio of validation samples.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): The number of worker threads.
        pin_memory (bool): Whether to use pinned memory.
        limit_queries (int, optional): The total limit of queries to read.
        limit_queries_affected_wl (int, optional): The limit of queries affected by workload.
        loss_class_name (str, optional): The name of the loss class.
    """

    def __init__(
        self,
        workload_run_paths,
        test_workload_run_paths,
        statistics_file,
        plan_featurization_name,
        database,
        batch_size=32,
        val_ratio=0.15,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
        limit_queries=None,
        limit_queries_affected_wl=None,
        loss_class_name=None,
    ):
        """
        Initializes the PlanDataModule with the provided parameters.
        """
        super().__init__()
        self.workload_run_paths = workload_run_paths
        self.test_workload_run_paths = test_workload_run_paths
        self.statistics_file = statistics_file
        self.plan_featurization_name = plan_featurization_name
        self.database = database
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.limit_queries = limit_queries
        self.limit_queries_affected_wl = limit_queries_affected_wl
        self.loss_class_name = loss_class_name

    def setup(self, stage=None):
        """
        Sets up the datasets for training, validation, and testing.

        Args:
            stage (str, optional): The stage for which to set up the datasets. Defaults to None.
        """
        # Called on each GPU separately - replace with your method to prepare data
        label_norm, self.train_dataset, self.val_dataset, database_statistics = (
            create_datasets(
                self.workload_run_paths,
                loss_class_name=self.loss_class_name,
                val_ratio=self.val_ratio,
                limit_queries=self.limit_queries,
                limit_queries_affected_wl=self.limit_queries_affected_wl,
            )
        )
        # Implement logic for test dataset similar to create_datasets function for test_workload_run_paths

    def train_dataloader(self):
        """
        Creates a DataLoader for the training dataset.

        Returns:
            DataLoader: The DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        """
        Creates a DataLoader for the validation dataset.

        Returns:
            DataLoader: The DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        """
        Creates a DataLoader for the test dataset. (To be implemented)
        """
        pass
