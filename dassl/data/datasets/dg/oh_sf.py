import os.path as osp
import glob
from ..build import DATASET_REGISTRY
from dassl.utils import listdir_nohidden
from .digits_dg import DigitsDG
from ..base_dataset import DatasetBase, Datum_sf, Datum

@DATASET_REGISTRY.register()
class OfficeHomeDG_SF(DatasetBase):
    """Office-Home.

    Statistics:
        - Around 15,500 images.
        - 65 classes related to office and home objects.
        - 4 domains: Art, Clipart, Product, Real World.
        - URL: http://hemanthdv.org/OfficeHome-Dataset/.

    Reference:
        - Venkateswara et al. Deep Hashing Network for Unsupervised
        Domain Adaptation. CVPR 2017.
    """

    dataset_dir = "office_home_dg"
    domains = ["none", "art", "clipart", "product", "real_world"]
    data_url = "https://drive.google.com/uc?id=1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa"

    def __init__(self, cfg, train_data):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, "office_home_dg.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        test_datasets = []
        val_datasets = []
        for domain in cfg.DATASET.TARGET_DOMAINS:
            test_datasets.append(self._read_data([domain], "all"))
        for domain in cfg.DATASET.SOURCE_DOMAINS:
            val_datasets.append(self._read_data([domain], "val"))
        #train = self._read_train_data(train_data, cfg.DATASET.SOURCE_DOMAINS, "train")

        '''
        train = DigitsDG.read_data(
            self.dataset_dir, cfg.DATASET.SOURCE_DOMAINS, "train"
        )
        val = DigitsDG.read_data(
            self.dataset_dir, cfg.DATASET.SOURCE_DOMAINS, "val"
        )
        test = DigitsDG.read_data(
            self.dataset_dir, cfg.DATASET.TARGET_DOMAINS, "all"
        )
        '''
        train = self._read_train_data(train_data, cfg.DATASET.SOURCE_DOMAINS, "train")
        #val = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "val")
        super().__init__(train_x=train, val = val_datasets, test=test_datasets)
        
    def _read_train_data(self, train_data, input_domains, split):
        items = []
        classnames = train_data["classnames"]
        n_cls = train_data["n_cls"]
        n_style = train_data["n_style"]
        
        def _load_data_from_directory(directory):
            folders = listdir_nohidden(directory)
            folders.sort()
            items_ = []

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(directory, folder, "*.jpg"))

                for impath in impaths:
                    items_.append((impath, label))

            return items_
        
        for domain, dname in enumerate(input_domains):
            if split == "all":
                train_dir = osp.join(self.dataset_dir, dname, "train")
                impath_label_list = _load_data_from_directory(train_dir)
                val_dir = osp.join(self.dataset_dir, dname, "val")
                impath_label_list += _load_data_from_directory(val_dir)
            else:
                split_dir = osp.join(self.dataset_dir, dname, split)
                
                impath_label_list = _load_data_from_directory(split_dir)
        
            for impath, label in impath_label_list: # label=0
                classname = impath.split("/")[-2]
                for idx_style in range(n_style):
                    item = Datum_sf(
                        cls=label, # 0
                        style=idx_style,
                        label=label,
                        classname=classnames[label],
                        impath=impath,
                        domain=domain,
                    )
                    items.append(item)
        
        return items

    def _read_data(self, input_domains, split):
        items = []
        def _load_data_from_directory(directory):
            folders = listdir_nohidden(directory)
            folders.sort()
            items_ = []

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(directory, folder, "*.jpg"))

                for impath in impaths:
                    items_.append((impath, label))

            return items_

        items = []
        for domain, dname in enumerate(input_domains):
            if split == "all":
                train_dir = osp.join(self.dataset_dir, dname, "train")
                impath_label_list = _load_data_from_directory(train_dir)
                val_dir = osp.join(self.dataset_dir, dname, "val")
                impath_label_list += _load_data_from_directory(val_dir)
            else:
                split_dir = osp.join(self.dataset_dir, dname, split)
                
                impath_label_list = _load_data_from_directory(split_dir)

            for impath, label in impath_label_list:
                class_name = impath.split("/")[-2].lower()
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    classname=class_name
                )
                items.append(item)
        
        return items

    