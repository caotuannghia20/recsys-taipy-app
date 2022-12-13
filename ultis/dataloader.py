import pandas as pd
import numpy as np
from scipy import sparse


class DataLoader:
    def __init__(self):
        self.__train_data = None
        self.__val_data = None
        self.__test_data = None

    def __create_id_mapping(self):
        if self.__val_data:
            unique_uIds = pd.concat(
                [self.__train_data.u_id, self.__test_data.u_id, self.__val_data.u_id]
            ).unique()
            unique_iIds = pd.concat(
                [self.__train_data.i_id, self.__test_data.i_id, self.__val_data.i_id]
            ).unique()
        else:
            unique_uIds = pd.concat(
                [self.__train_data.u_id, self.__test_data.u_id]
            ).unique()
            unique_iIds = pd.concat(
                [self.__train_data.i_id, self.__test_data.i_id]
            ).unique()

        self.user_dict = {uId: idx for idx, uId in enumerate(unique_uIds)}
        self.item_dict = {iId: idx for idx, iId in enumerate(unique_iIds)}

    def __preprocess(self, data):
        """Map the id of all users and items according to user_dict and item_dict.
        To create the user_dict, all user ID in the training set is first sorted, then the first ID is map to 0 and so on.
        Do the same for item_dict.
        This process is done via `self.__create_id_mapping()`.

        Args:
            data (Dataframe): The dataset that need to be preprocessed.

        Returns:
            ndarray: The array with all id mapped.
        """
        # data['u_id'] = data['u_id'].replace(self.user_dict)
        # data['i_id'] = data['i_id'].replace(self.item_dict)

        data["u_id"] = data["u_id"].map(self.user_dict)
        data["i_id"] = data["i_id"].map(self.item_dict)

        # Tag unknown users/items with -1 (when val)
        data.fillna(-1, inplace=True)

        data["u_id"] = data["u_id"].astype(np.int32)
        data["i_id"] = data["i_id"].astype(np.int32)

        return data[["u_id", "i_id", "rating"]].values

    def load_csv2ndarray(
        self,
        train_data,
        test_data,
        val_path="rating_val.csv",
        use_val=False,
        columns=["u_id", "i_id", "rating", "timestamp"],
    ):
        """
        Load training set, validate set and test set via `.csv` file.
        Each as `ndarray`.

        Args:
            train_path (string): path to the training set csv file inside self.__data_folder
            test_path (string): path to the testing set csv file inside self.__data_folder
            val_path (string): path to the validating set csv file inside self.__data_folder
            use_val (boolean): Denote if loading validate data or not. Defaults to False.
            columns (list): Columns name for DataFrame. Defaults to ['u_id', 'i_id', 'rating', 'timestamp'].

        Returns:
            train, val, test (np.array): Preprocessed data.
        """
        self.__train_data = train_data
        self.__test_data = test_data

        if use_val:
            self.__val_data = self.__read_csv(val_path, columns)

        self.__create_id_mapping()

        self.__train_data = self.__preprocess(self.__train_data)
        self.__test_data = self.__preprocess(self.__test_data)

        if use_val:
            self.__val_data = self.__preprocess(self.__val_data)
            return self.__train_data, self.__val_data, self.__test_data
        else:
            return self.__train_data, self.__test_data

    def load_genome_fromcsv(
        self,
        genome_file="genome_scores.csv",
        columns=["i_id", "g_id", "score"],
        reset_index=False,
    ):
        """
        Load genome scores from file.
        Args:
            genome_file (string): File name that contain genome scores. Must be in csv format.
            columns (list, optional): Columns name for DataFrame. Must be ["i_id", "g_id", "score"] or ["i_id", "score", "g_id"].
            reset_index (boolean): Reset the genome_tag column or not. Defaults to False.

        Returns:
            scores (DataFrame)
        """
        genome = pd.read_csv(
            self.__genome_folder + "/" + genome_file, header=0, names=columns
        )

        if reset_index:
            tag_map = {
                genome.g_id: newIdx
                for newIdx, genome in genome.loc[genome.i_id == 1].iterrows()
            }
            genome["g_id"] = genome["g_id"].map(tag_map)

        genome["i_id"] = genome["i_id"].map(self.item_dict)
        genome.fillna(0, inplace=True)

        return sparse.csr_matrix(
            (genome["score"], (genome["i_id"].astype(int), genome["g_id"].astype(int)))
        ).toarray()
