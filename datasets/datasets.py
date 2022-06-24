from urllib.error import URLError
from sklearn.preprocessing import LabelEncoder
import pickle
import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('logger')



class Davis(DTI_dataset):

    download_link = "https://raw.githubusercontent.com/kexinhuang12345/MolTrans/master/dataset/DAVIS/"

    def __init__(self, *args, **kwars) -> None:
        self.label_encoder = LabelEncoder()
        super().__init__(link=self.download_link, *args, **kwars)

    
    def _load_data(self) -> None:
        logger.debug("Went into _load_data.")
        # Get SMILES and TargetSequences encodings
        self._encode_by_ind()
        # Now load data into self.features
        df = pd.read_csv(os.path.join(self.raw_folder, self.mode + '.csv'), index_col=0)

        self.add_feature(feat_name='SMILES', feat_values=df['SMILES'].values.tolist())
        self.add_feature(feat_name='Sequence', feat_values=df['Target Sequence'].values.tolist())
        self.add_feature(feat_name='Label', feat_values=df['Label'].values.tolist())

        self.add_feature(feat_name='DrugName', feat_values=self.entity_to_ind(self.features['SMILES']))
        self.add_feature(feat_name='DrugInd', feat_values=self.entity_to_ind(self.features['SMILES']))
        self.add_feature(feat_name='ProtName', feat_values=self.entity_to_ind(self.features['Sequence']))
        self.add_feature(feat_name='ProtInd', feat_values=self.entity_to_ind(self.features['Sequence']))
        
        self.return_options = self.features.keys()

        logger.debug("LOADED!")


    def _load_processed_data(self) -> None:
        """Load processed data (if exist) and store features in self.features dictionary."""
        # Not implemented yet
    
    def _updata_processed_data(self) -> None:
        """Update processed data (rewrite files train/val/test.csv"""
        # Not implemented yet

    def download(self, link: str) -> None:
        """Download the data if it doesn't exist already."""

        if self._check_exists(self.raw_folder):
            logger.debug("Raw dataset already downloaded.")
            return
        
        os.makedirs(self.raw_folder, exist_ok=True)

        # download and save all raw datasets (train/val/test)
        for filename in self.filenames.values():
            url = f"{link}{filename}"
            try:
                print(f"Downloading {url}")
                df = pd.read_csv(url, index_col=0)
                df[['SMILES','Target Sequence', 'Label']].to_csv(os.path.join(self.raw_folder, filename))
            except URLError as error:
                print(f"Failed to download {filename}:\n{error}")
                continue
            finally:
                print()

    def _encode_by_ind(self) -> None:
        '''
        Encode drugs and targets by Index
        '''
        # If LabelEncoder already exists -- load encodings
        label_encoder_path = os.path.join(self.raw_folder, 'label_encoder.pkl')
        if os.path.exists(label_encoder_path):
            logger.debug('Loading existing label encoder...')
            with open(label_encoder_path, 'rb') as le_dump_file:
                self.label_encoder = pickle.load(le_dump_file)
            logger.debug('Loaded succesfully.\n')
            #attribute
            self.n_entities = len(self.label_encoder.classes_)
            return

        # If not:
        # load all raw files and encode all unique drugs/prots by some IND
        dataset = pd.DataFrame()
        for filename in self.filenames.values():
            df = pd.read_csv(os.path.join(self.raw_folder, filename))
            dataset = dataset.append(df, ignore_index=True).drop_duplicates(ignore_index=True)
        drugs = dataset['SMILES'].unique()
        prots = dataset['Target Sequence'].unique()
        entities = np.append(drugs, prots)           # all unique drugs and proteins
        # attributes
        self.n_entities = len(entities)
        # save encodings 
        self.label_encoder.fit(entities)
        with open(label_encoder_path, 'wb') as le_dump_file:
            pickle.dump(self.label_encoder, le_dump_file)
        logger.debug('Drugs and Proteins were encoded by IND\'s...')
        # ...now we have encodings stored in label_encoder.pkl and in self.label_encoder

    def ind_to_entity(self, ind: list) -> List:
        '''
        Gets list of drugs/proteins IND's.
        Returns SMILES strings/Target Sequences.
        '''
        # return [self.label_encoder.classes_[i] for i in ind]
        return self.label_encoder.inverse_transform(ind).tolist()

    def entity_to_ind(self, s: list) -> List:
        '''
        Gets list of SMILES strings/Target Sequences.
        Returns their indexes (IND's).
        '''
        return self.label_encoder.transform(s).tolist()

    def __len__(self) -> int:
        return len(self.features['Label'])

    def __getitem__(self, idx):
        '''
        Info about what dataset should return is stated in self.return_type
        For every feature_i in self.return_type:
            1. Takes self.features[feature_i][index]
            2. Concatenates with others self.features[feature_j][index]
            3. Returns them
        '''

        feats_to_return = []

        for feat in self.return_type:
            feat_i = self.features[feat][idx]
            try:
                feat_i = torch.tensor(feat_i, dtype=torch.float32)
            except:
                pass
            feats_to_return.append(feat_i)

        return tuple(feats_to_return)